import os
from pathlib import Path

import hydra
import torch
import wandb
import random
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DeepSpeedStrategy
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import get_model
from src.misc.weight_modify import checkpoint_filter_fn
from src.model.model.anysplat import AnySplat

import warnings
warnings.filterwarnings("ignore")

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def load_weights_into_model(model, pretrained_path, device):
    import torch, os

    if not os.path.exists(pretrained_path):
        print(f"Pretrained path {pretrained_path} not found. Skip loading.")
        return model

    # If pretrained_path is a HF-style directory, try from_pretrained to get a source model
    if os.path.isdir(pretrained_path):
        try:
            # Try to use HF mixin to read weights (local_files_only to avoid network)
            src_model = AnySplat.from_pretrained(pretrained_path, local_files_only=True)
            src_state = src_model.state_dict()
            del src_model
        except Exception:
            # fallback to torch.load for files inside directory (e.g., pytorch_model.bin)
            # search for common filenames
            candidates = ["pytorch_model.bin", "model.pt", "model.pth", "weights.pt"]
            file_found = None
            for f in candidates:
                p = os.path.join(pretrained_path, f)
                if os.path.exists(p):
                    file_found = p
                    break
            if file_found is None:
                print(f"No recognized weight file found in {pretrained_path}")
                return model
            ckpt = torch.load(file_found, map_location="cpu")
            src_state = ckpt.get("state_dict", ckpt)
    else:
        # pretrained_path is a file
        ckpt = torch.load(pretrained_path, map_location="cpu")
        src_state = ckpt.get("state_dict", ckpt)

    # strip common prefixes (Lightning, DataParallel...)
    def strip_prefix(state, prefixes=("module.", "model.")):
        new = {}
        for k, v in state.items():
            nk = k
            for p in prefixes:
                if k.startswith(p):
                    nk = k[len(p):]
                    break
            new[nk] = v
        return new

    src_state = strip_prefix(src_state)

    # Try direct strict load first, else fallback to non-strict
    try:
        model.load_state_dict(src_state, strict=True)
        print("Loaded pretrained weights (strict=True).")
    except Exception as e:
        print(f"Strict load failed: {e}\nTrying strict=False (partial load).")
        res = model.load_state_dict(src_state, strict=False)
        # print("Load result:", res)
    model.to(device)
    return model


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    
    cfg.train.output_path = output_dir
    
    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))
        
        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    
    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'
    
    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    
    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
    
    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        # num_sanity_val_steps=0,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        # strategy="deepspeed_stage_1",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    
    model = load_weights_into_model(model, "./model/AnySplat", device="cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        model,
        get_losses(cfg.loss),
        step_tracker
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    
    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
