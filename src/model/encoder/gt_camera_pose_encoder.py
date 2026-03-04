import torch
from torch import nn
from einops import rearrange
from typing import Optional
  

class GTCameraPoseEncoder(nn.Module):
    """ 
    Encode ground-truth camera poses (extrinsics or pose encodings) into camera tokens
    that can replace the Aggregator's learned camera_token.

    Expected input shapes:
      - extrinsics: (B, S, 4, 4)  -> will be flattened to 12 dims (3x4)
      - or pose_enc: (B, S, P) where P is small (e.g., 9)

    Output:
      - camera_override: (B, 2, E) where E == embed_dim (first-frame token, rest-frames token)
    """

    def __init__(self, embed_dim=256, patch_size=4, nhead=8, num_layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # 1. 局部几何投影：处理每个 Patch 内部的 3D 射线分布
        # 输入维度：3 (XYZ) * Patch宽 * Patch高
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        
        # 2. 全局交互 Transformer
        # 这里会让所有相机的全量 Patch (5 * 16 = 80个) 互相求 Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def _ensure_proj(self, in_dim: int):
        if self.input_proj is None:
            self.input_proj = nn.Linear(in_dim, self.embed_dim)
            # move to same device if needed later

    def forward(self, height: int, width: int, extrinsics: torch.Tensor, intrinsics: Optional[torch.Tensor] = None,) -> torch.Tensor:
        """Encode poses -> camera_override (B,2,embed_dim)

        poses can be extrinsics (B,S,4,4) or (B,S,3,4) or pose encodings (B,S,P).
        """
        B, N_cam, _, _ = intrinsics.shape
        device = intrinsics.device

        # 处理 intrinsics 为空或为归一化值的情况
        if intrinsics is None:
            # 默认内参：主点在中心，焦距取为 0.5 * max(H, W)
            fx = fy = 0.5 * max(width, height)
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            intrinsics = torch.zeros((B, N_cam, 3, 3), device=device, dtype=extrinsics.dtype)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = cx
            intrinsics[..., 1, 2] = cy
            intrinsics[..., 2, 2] = 1.0
        else:
            # 将提供的 intrinsics 移到正确设备/类型并 clone（避免原地修改外部张量）
            intrinsics = intrinsics.to(device=device, dtype=extrinsics.dtype).clone()
            # 如果内参是归一化到 [0,1] 的（例如来自相对坐标），则按像素尺度放大
            intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * width
            intrinsics[..., 0, 2] = intrinsics[..., 0, 2] * width
            intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * height
            intrinsics[..., 1, 2] = intrinsics[..., 1, 2] * height


        # 1. 生成像素网格 (u, v, 1) -> [H, W, 3]
        y, x = torch.meshgrid(torch.arange(height, device=device), 
                              torch.arange(width, device=device), indexing='ij')
        # 深度恒定为 1
        coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() 
        coords = coords.reshape(1, 1, height * width, 3) # [1, 1, N, 3]

        # 2. 变换到相机坐标系: K_inv * p_pixel
        # 这里得到了在相机空间中，深度为1的射线向量
        inv_k = torch.inverse(intrinsics) # [B, N_cam, 3, 3]
        points_cam = torch.matmul(inv_k.view(B, N_cam, 1, 3, 3), coords.unsqueeze(-1)).squeeze(-1)

        # 3. 变换到世界坐标系 (包含旋转 R 和 平移 t)
        # R: [B, N_cam, 3, 3], t: [B, N_cam, 3, 1]
        R = extrinsics[:, :, :3, :3].view(B, N_cam, 1, 3, 3)
        t = extrinsics[:, :, :3, 3].view(B, N_cam, 1, 3, 1)
        
        # P_world = R * P_cam + t
        points_world = torch.matmul(R, points_cam.unsqueeze(-1)) + t
        points_world = points_world.squeeze(-1) # [B, N_cam, N_pix, 3]

        coord_map = points_world.view(B, N_cam, height, width, 3).permute(0, 1, 4, 2, 3)

        # 4. 频率编码 (Sinusoidal Encoding)
        # 将 3D 坐标映射为高维特征
        pe = self._self_attention(coord_map)
        return pe


    def _self_attention(self, coord_map):
        """
        Args:
            coord_map: [B, 5, 3, H, W] (带平移的世界坐标图)
        Returns:
            camera_tokens: [B, 5, embed_dim]
        """
        B, N_cam, C, H, W = coord_map.shape
        P = self.patch_size

        # --- 第一步：分块并投影 ---
        # [B, 5, 3, H, W] -> [B, 5, 16, 3*P*P]
        patches = rearrange(coord_map, 'b n c (h p1) (w p2) -> b n (h w) (c p1 p2)', 
                            p1=P, p2=P)
        
        # [B, 5, 16, 3*P*P] -> [B, 5, 16, embed_dim]
        patch_features = self.patch_proj(patches)
        
        # --- 第二步：跨相机交互 (重点) ---
        # 我们把 (5 * 16) 个 Patch 全部铺平，送入 Transformer
        # 这样相机 A 的 Patch 就能看到相机 B 的 Patch
        all_tokens = rearrange(patch_features, 'b n p e -> b (n p) e') # [B, 80, embed_dim]
        interacted_all = self.transformer(all_tokens) # [B, 80, embed_dim]
        
        # --- 第三步：收缩回相机维度 ---
        # 重新分回 [B, 5, 16, embed_dim]
        interacted_patches = rearrange(interacted_all, 'b (n p) e -> b n p e', n=N_cam)
        
        # 对 16 个 Patch 取均值（或者最大值），聚合为相机的全局代表
        # [B, 5, 16, 256] -> [B, 5, 256]
        camera_tokens = interacted_patches.mean(dim=2)
        
        return camera_tokens