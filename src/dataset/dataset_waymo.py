from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import os
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim, rescale_depth, rescale
from .shims.geometry_shim import depthmap_to_absolute_camera_coordinates
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetWaymoCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    avg_pose: bool
    rescale_to_1cube: bool
    intr_augment: bool
    normalize_by_pts3d: bool
    rescale_to_1cube: bool


@dataclass
class DatasetWaymoCfgWrapper:
    waymo: DatasetWaymoCfg



class DatasetWaymo(Dataset):
    cfg: DatasetWaymoCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0
    
    def __init__(
        self,
        cfg: DatasetWaymoCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        # load data
        self.data_root = cfg.roots[0]
        self.data_list = []
        with open(f"{self.data_root}/{self.data_stage}_index.json", "r") as file:
            data_index = json.load(file)
        
        self.data_list = [
            os.path.join(self.data_root, item) for item in data_index
        ]  # train: 9900 test: 140
        
        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.load_jsons, scene_path) for scene_path in self.data_list]
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        print(f"WAYMO: {self.stage}: loaded {len(self.scene_ids)} scenes")
        
    def convert_intrinsics(self, meta_data):
        store_h, store_w = meta_data["width"], meta_data["height"]
        intr = meta_data["intrinsic"]
        fx, fy, cx, cy = (
            intr[0],
            intr[1],
            intr[2],
            intr[3],
        )
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = float(fx) / float(store_w)
        intrinsics[1, 1] = float(fy) / float(store_h)
        intrinsics[0, 2] = float(cx) / float(store_w)
        intrinsics[1, 2] = float(cy) / float(store_h)
        return intrinsics
        
    def waymo2opencv_c2w(self, pose):
        pose = np.asarray(pose, dtype=np.float32)
        
        pose = pose.reshape(4, 4)
  
        blender2opencv = np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()

    def load_jsons(self, scene_path):
        json_path = os.path.join(scene_path, "calib.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        
        scene_frames = []
        scene_id = scene_path.split("/")[-1].split(".")[0]
        data = data["cameras"]
        for it in data.keys():
            frame_tmp = {}
            frame_tmp["file_path"] = os.path.join(scene_path, data[it]["file_path"])
            frame_tmp["intrinsics"] = self.convert_intrinsics(data[it]).tolist()
            frame_tmp["extrinsics"] = self.waymo2opencv_c2w(data[it]["extrinsic"])
            scene_frames.append(frame_tmp)
        return scene_frames, scene_id

    def load_frames(self, frames):
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, file_path in enumerate(frames):
                file_path = file_path["file_path"].replace("images", "images")
                futures_with_idx.append(
                    (
                        idx,
                        executor.submit(
                            lambda p: self.to_tensor(Image.open(p).convert("RGB")),
                            file_path,
                        ),
                    )
                )
            
            # Pre-allocate list with correct size to maintain order
            torch_images = [None] * len(frames)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
            # Check if all images have the same size
            sizes = set(img.shape for img in torch_images)
            if len(sizes) == 1:
                torch_images = torch.stack(torch_images)
        # Return as list if images have different sizes
        return torch_images

    def load_depth(self, frames):
        """Load depth maps in parallel and pad them to same HxW.
        Returns:
            depths: Tensor (N, H_max, W_max)
            pad_mask: Bool Tensor (N, H_max, W_max)  True means padded region
        """
        depth_list = []
        for frame in frames:
            frame_name = frame["file_path"]
            depth_path = frame_name.replace("images", "depths").replace("jpg", "npy")
            depth = torch.from_numpy(np.load(depth_path))
            positive_depths = depth[depth > 0]
            if len(positive_depths) > 1000000:  # If more than 1M points, sample randomly
                indices = torch.randperm(len(positive_depths))[:1000000]
                positive_depths = positive_depths[indices]
            percentile_95 = torch.quantile(positive_depths, 0.95)
            # Set depth values greater than the 95th percentile to 0
            depth[depth > percentile_95] = 0
            depth_list.append(depth)
        sizes = set(img.shape for img in depth_list)
        if len(sizes) == 1:
            depth_list = torch.stack(depth_list)

        return depth_list

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    

    def resize_images_and_intrinsics(self, imgs_list, depths_list):
        """
        Combine each image (3,H,W) with its depth (H,W) into (4,H,W), resize all to the
        maximum HxW among inputs, adjust intrinsics (assumed normalized by width/height),
        then split back into images (3,H,W) and depths (H,W).

        Note: target_shape argument is accepted for compatibility but ignored; the actual
        target is computed as the largest HxW among inputs.
        """
        import torch.nn.functional as F

        # Normalize inputs to lists of 

        n = len(imgs_list)
        if len(depths_list) != n:
            # if depths shorter/longer, try to align by index; if mismatch, raise
            raise ValueError(f"Number of images ({n}) and depths ({len(depths_list)}) must match")

        # Determine max height and width among all images and depths
        max_h = 0
        max_w = 0
        for img in imgs_list:
            if not isinstance(img, torch.Tensor):
                raise TypeError("image items must be torch.Tensor")
            h, w = int(img.shape[1]), int(img.shape[2])
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        for d in depths_list:
            if not isinstance(d, torch.Tensor):
                raise TypeError("depth items must be torch.Tensor")
            dh, dw = int(d.shape[0]), int(d.shape[1])
            max_h = max(max_h, dh)
            max_w = max(max_w, dw)

        target_h, target_w = max_h, max_w

        resized_imgs = []
        resized_depths = []

        for i, (img, depth) in enumerate(zip(imgs_list, depths_list)):
            # ensure types & shapes
            assert img.dim() == 3 and img.shape[0] == 3, "Expected image tensor (3,H,W)"
            assert depth.dim() == 2, "Expected depth tensor (H,W)"

            old_h, old_w = int(img.shape[1]), int(img.shape[2])

            # combine to 4-channel tensor
            depth_ch = depth.unsqueeze(0)  # (1,H,W)
            combined = torch.cat([img.to(torch.float32), depth_ch.to(torch.float32)], dim=0)  # (4,H,W)

            # resize if needed
            if (old_h, old_w) != (target_h, target_w):
                combined_b = combined.unsqueeze(0)  # (1,4,H,W)
                combined_resized = F.interpolate(
                    combined_b, size=(target_h, target_w), mode="bilinear", align_corners=False
                ).squeeze(0)  # (4, target_h, target_w)
            else:
                combined_resized = combined

            # split back
            img_resized = combined_resized[0:3, :, :]
            depth_resized = combined_resized[3, :, :]

            resized_imgs.append(img_resized)
            resized_depths.append(depth_resized)

        images_tensor = torch.stack(resized_imgs, dim=0)  # (N,3,H,W)
        depths_tensor = torch.stack(resized_depths, dim=0)  # (N,H,W)

        return images_tensor, depths_tensor
    
    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        
        scene = self.scene_ids[index]
        
        example = self.scenes[scene]
        # load poses
        extrinsics = []
        intrinsics = []
        for frame in example:
            extrinsic = frame["extrinsics"]
            intrinsic = frame["intrinsics"]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        
        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        
        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                scene,
                num_context_views,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Skip because the example doesn't have enough frames.
            raise Exception("Not enough frames")
        
        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")
        
        # Load the images.
        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]
        
        
        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)

        context_depth  = self.load_depth(input_frames)
        target_depth = self.load_depth(target_frame)

        if isinstance(context_images, list):
            context_images, context_depth = self.resize_images_and_intrinsics(context_images, context_depth)
        
        if isinstance(target_images, list):
            target_images, target_depth = self.resize_images_and_intrinsics(target_images, target_depth)
        
        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            print(
                f"Skipped bad example {example['key']}. Context shape was "
                f"{context_images.shape} and target shape was "
                f"{target_images.shape}."
            )
            raise Exception("Bad example image shape")
        
        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                print(
                    f"Skipped {scene} because of baseline out of range: "
                    f"{scale:.6f}"
                )
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1
        
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3])) # target pose is not included
            rescale_factor = 1 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "depth": context_depth,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                # "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "depth": target_depth,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": "waymo_"+scene,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        if self.stage == "train" and self.cfg.intr_augment:
            intr_aug = True
        else:
            intr_aug = False
        # example = apply_crop_shim(example, (self.cfg.input_image_shape[0], self.cfg.input_image_shape[1]), intr_aug=intr_aug)
        example['context']['image'] = torch.stack([rescale(image, (self.cfg.input_image_shape[0]*2, self.cfg.input_image_shape[1])) for image in example['context']['image']])
        example['context']['depth'] = torch.stack([rescale_depth(depth, (self.cfg.input_image_shape[0]*2, self.cfg.input_image_shape[1])) for depth in example['context']['depth']])

        image_size = example["context"]["image"].shape[2:]
        context_intrinsics = example["context"]["intrinsics"].clone().detach().numpy()
        context_intrinsics[:, 0] = context_intrinsics[:, 0] * image_size[1]
        context_intrinsics[:, 1] = context_intrinsics[:, 1] * image_size[0]

        target_intrinsics = example["target"]["intrinsics"].clone().detach().numpy()
        target_intrinsics[:, 0] = target_intrinsics[:, 0] * image_size[1]
        target_intrinsics[:, 1] = target_intrinsics[:, 1] * image_size[0]

        context_pts3d_list, context_valid_mask_list = [], []    
        target_pts3d_list, target_valid_mask_list = [], []

        for i in range(len(example["context"]["depth"])):
            context_pts3d, context_valid_mask = depthmap_to_absolute_camera_coordinates(example["context"]["depth"][i].numpy(), context_intrinsics[i], example["context"]["extrinsics"][i].numpy())
            context_pts3d_list.append(torch.from_numpy(context_pts3d).to(torch.float32))
            context_valid_mask_list.append(torch.from_numpy(context_valid_mask))
        context_pts3d = torch.stack(context_pts3d_list, dim=0)
        context_depth_valid_mask = torch.stack(context_valid_mask_list, dim=0)
        context_image_nonzero = (example["context"]["image"] != 0).any(dim=1)  # (N, H, W), True if any channel != 0
        context_valid_mask = (context_depth_valid_mask & context_image_nonzero).bool()  # [N, H, W]


        for i in range(len(example["target"]["depth"])):
            target_pts3d, target_valid_mask = depthmap_to_absolute_camera_coordinates(example["target"]["depth"][i].numpy(), target_intrinsics[i], example["target"]["extrinsics"][i].numpy())
            target_pts3d_list.append(torch.from_numpy(target_pts3d).to(torch.float32))
            target_valid_mask_list.append(torch.from_numpy(target_valid_mask))
        target_pts3d = torch.stack(target_pts3d_list, dim=0)
        target_depth_valid_mask = torch.stack(target_valid_mask_list, dim=0)
        target_image_nonzero = (example["target"]["image"] != 0).any(dim=1)  # (N, H, W), True if any channel != 0
        target_valid_mask = (target_depth_valid_mask & target_image_nonzero).bool()  # [N, H, W]

        
        # normalize by context pts3d
        if self.cfg.normalize_by_pts3d:
            transformed_pts3d = context_pts3d[context_valid_mask]
            scene_factor = transformed_pts3d.norm(dim=-1).mean().clip(min=1e-8)

            context_pts3d /= scene_factor
            example["context"]["depth"] /= scene_factor
            example["context"]["extrinsics"][:, :3, 3] /= scene_factor

            target_pts3d /= scene_factor
            example["target"]["depth"] /= scene_factor
            example["target"]["extrinsics"][:, :3, 3] /= scene_factor
        
        example["context"]["pts3d"] = context_pts3d
        example["target"]["pts3d"] = target_pts3d


        # # Save whole scene context_pts3d as a PLY point cloud (points + colors)
        # try:
        #     out_dir = "tmp"

        #     # context_pts3d: (N, H, W, 3), context_valid_mask: (N, H, W)
        #     pts = context_pts3d[context_valid_mask]  # (M, 3)
        #     if pts.numel() > 0:
        #         # colors from context images: (N,3,H,W) -> (N,H,W,3)
        #         imgs = example["context"]["image"].permute(0, 2, 3, 1)  # (N,H,W,3)
        #         colors = imgs[context_valid_mask]  # (M,3), floats in [0,1] from to_tensor
        #         colors = (colors.clamp(0.0, 1.0) * 255).to(torch.uint8)

        #         pts_np = pts.cpu().numpy()
        #         colors_np = colors.cpu().numpy()

        #         ply_path = f"tmp/{example['scene']}_context_pts.ply"
        #         with open(ply_path, "w") as f:
        #             f.write("ply\n")
        #             f.write("format ascii 1.0\n")
        #             f.write(f"element vertex {pts_np.shape[0]}\n")
        #             f.write("property float x\nproperty float y\nproperty float z\n")
        #             f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        #             f.write("end_header\n")
        #             for (x, y, z), (r, g, b) in zip(pts_np, colors_np):
        #                 f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")
        # except Exception as e:
        #     # Do not fail dataset loading on ply write errors; just log.
        #     print(f"Warning: failed to write ply for scene {example.get('scene', '')}: {e}")


        example["context"]["valid_mask"] = context_valid_mask * -1
        example["target"]["valid_mask"] = target_valid_mask * -1

        return example
        
    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            index = np.random.randint(len(self))
            return self.__getitem__((index, num_context_views, patchsize_h))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.scene_ids)
