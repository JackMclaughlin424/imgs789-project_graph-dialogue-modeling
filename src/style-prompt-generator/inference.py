"""
inference.py  --config path/to/config.json
              --checkpoint path/to/ckpt.pt
              [--wandb_run_name name]
              [--override KEY=VALUE ...]

Loads a model checkpoint, runs batch inference on the fixed test split,
computes all test metrics (BERTScore, METEOR, CHrF, ROUGE-L, Tag-F1),
logs results to W&B, and writes predictions to a CSV.
"""

import argparse
import gc
import logging
import os

import pandas as pd
import torch
import wandb

from dataset.ConvoStyleDataset import ConvoStyleDataset
from model.train_helpers import (
    apply_overrides, build_model, load_config, set_seed, wandb_log, eval_test_by_source
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch inference + test metrics from a checkpoint.")
    parser.add_argument("--config",     required=True, help="Base config JSON used during training.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--output_csv", default="inference_predictions.csv",
                        help="Path to write predictions CSV (default: inference_predictions.csv).")
    parser.add_argument("--wandb_run_name", default=None, help="Optional W&B run name.")
    parser.add_argument("--override",   nargs="*", metavar="KEY=VALUE",
                        help="Override config fields, same syntax as train.py.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override, log)
    if args.wandb_run_name:
        cfg["run_name"] = args.wandb_run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    set_seed(cfg["seed"])

    model = build_model(cfg, device, log)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    log.info(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')})")
    model.eval()

    test_chains_by_source, _ = ConvoStyleDataset.make_fixed_test_split(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        max_len_sec=cfg["max_len_sec"],
        num_turns=cfg["num_turns"],
    )


    source_metrics = eval_test_by_source(model, cfg, test_chains_by_source, device, log)

    gc.collect()
    torch.cuda.empty_cache()

    # W&B logging
    run = None
    if cfg.get("use_wandb", True) and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        run = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            name=cfg.get("run_name"),
            config={**cfg, "checkpoint": args.checkpoint},
            job_type="inference",
        )
        log.info(f"W&B run: {run.url}")

    all_rows = []
    for src, src_m in source_metrics.items():
        log.info(
            f"Test/{src}  bertscore_f1={src_m['bertscore_f1']:.4f}  "
            f"meteor={src_m['meteor']:.4f}  chrf={src_m['chrf']:.4f}  "
            f"tag_f1={src_m['tag_f1_overall']:.4f}"
        )
        test_summary = {f"test/{src}/{k}": v for k, v in src_m.items()}
        if run is not None:
            run.summary.update(test_summary)
            wandb_log(test_summary, step=0, run=run)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
