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

import numpy as np
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
    apply_overrides, set_seed,
    build_optimizer_and_scheduler,
    wandb_log, wandb_init, load_checkpoint, save_checkpoint, prune_old_checkpoints, wandb_finish
)
from capstone_src.style_prompt_generator.dataset.ConvoStyleDataset import ConvoStyleDataset, collate_pad
from torch.utils.data import DataLoader

_on_colab = "google.colab" in sys.modules or os.path.isdir("/content")

if _on_colab:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        stream=sys.stdout,
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
log = logging.getLogger(__name__)

# write our logs to a file independent of wandb's stderr capture
_log_file = os.environ.get("SLURM_JOB_LOG", "sweep_run.log")
_fh = logging.FileHandler(_log_file, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_fh)



logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)



def train(cfg: Dict[str, Any], resume: bool = True) -> None:
    """Train the graph model; saves final weights to output_dir/final_model.pt."""
    set_seed(cfg["seed"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    _, test_conv_ids = ConvoStyleDataset.make_fixed_test_split(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        max_len_sec=cfg["max_len_sec"],
        num_turns=cfg["num_turns"],
    )
    meta = pd.read_parquet(cfg["meta_path"], columns=["conv_id"])
    train_ids = set(c for c in meta["conv_id"].unique() if c not in test_conv_ids)

    train_ds = ConvoStyleDataset(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        num_turns=cfg["num_turns"],
        max_len_sec=cfg["max_len_sec"],
        allowed_conv_ids=train_ids,
    )
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g,
        collate_fn=collate_pad, num_workers=cfg["num_workers"], pin_memory=True,
    )
    log.info(f"{len(train_ds)} train chains")

    model = build_model(cfg, device, log)

    total_steps          = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)

    start_epoch = 0
    global_step = 0

    if resume:
        ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
        if ckpts:
            start_epoch, global_step = load_checkpoint(
                str(ckpts[-1]), log, model, optimizer, scheduler
            )
            start_epoch += 1

    wandb_run = wandb_init(cfg, log)

    for epoch in range(start_epoch, cfg["num_epochs"]):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=wandb_run,
            is_train=True, use_tqdm=True, log_handler=log
        )

        wandb_log({"epoch/train_loss": train_loss}, step=global_step, run=wandb_run)

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

    with open(args.config) as f:
        cfg = json.load(f)
        
    apply_overrides(cfg, args.override, log)
    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()


