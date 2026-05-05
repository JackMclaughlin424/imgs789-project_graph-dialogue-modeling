"""
train.py  --config path/to/config.json

Trains the SCFAWithStyleHead pipeline end-to-end (or partially frozen),
reading all hyperparameters from a JSON config file.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


import pandas as pd

from graph_model.graph_model_helpers import (
    run_epoch, build_model
)

import sys


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.fx")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._inductor")

import os, time

from capstone_src.style_prompt_generator.model.train_helpers import (
     load_config, apply_overrides, set_seed,
    build_optimizer_and_scheduler,
    wandb_log, build_dataloaders, wandb_init, load_checkpoint, save_checkpoint, prune_old_checkpoints, wandb_finish
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_log_file = os.environ.get("SLURM_JOB_LOG", "train.log")
_fh = logging.FileHandler(_log_file, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_fh)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)




def train(cfg: Dict[str, Any], resume: bool = True) -> None:
    """Train SCFAWithStyleHead; saves final weights to output_dir/final_model.pt."""
    set_seed(cfg["seed"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, _, _ = build_dataloaders(cfg, log)
    model              = build_model(cfg, device, log)

    total_steps            = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler   = build_optimizer_and_scheduler(model, cfg, total_steps, log)

    start_epoch  = 0
    global_step  = 0

    if resume:
        ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
        if ckpts:
            start_epoch, global_step = load_checkpoint(
                str(ckpts[-1]), log, model, optimizer, scheduler
            )
            start_epoch += 1

    wandb_run = wandb_init(cfg, log)

    patience          = cfg.get("early_stopping_patience", 3)
    min_delta         = cfg.get("early_stopping_min_delta", 1e-4)
    MIN_EPOCH         = 5
    best_loss         = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, cfg["num_epochs"]):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=wandb_run,
            is_train=True, use_tqdm=False,
        )

        # epoch-level summary (step-level metrics are logged inside run_epoch)
        wandb_log({"epoch/train_loss": train_loss}, step=global_step, run=wandb_run)

        if (epoch + 1) % cfg["save_every_n_epochs"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                global_step, train_loss, cfg, out_dir, log,
            )
            prune_old_checkpoints(out_dir, cfg["keep_last_n_ckpts"], log)

        if train_loss < best_loss - min_delta:
            best_loss         = train_loss
            epochs_no_improve = 0
        elif patience > 0 and epoch >= MIN_EPOCH:
            epochs_no_improve += 1

        if patience > 0 and epoch >= MIN_EPOCH and epochs_no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    final_path = out_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    log.info(f"Final model saved: {final_path}")

    wandb_finish(wandb_run)



# Entry point

def main():
    parser = argparse.ArgumentParser(description="Train GraphStyleGenerator from a JSON config.")
    parser.add_argument("--config",   required=True, help="Path to hyperparameter config JSON.")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE",
                        help="Override individual config fields (e.g. --override learning_rate=1e-4).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and train from scratch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override, log)
    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()


