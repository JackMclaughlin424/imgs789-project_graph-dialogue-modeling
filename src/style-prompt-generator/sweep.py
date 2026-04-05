"""
sweep.py  --config path/to/config.json [options]

Runs a W&B hyperparameter sweep where each trial performs N-fold cross-validation.
Conversations are split at the conv_id level (same anti-leakage logic as train.py).

Usage
-----
# Start a new sweep and run one agent:
    python sweep.py --config default_config.json --n_folds 5 --count 20

# Join an existing sweep from a second machine:
    python sweep.py --config default_config.json --sweep_id <SHORT_ID> --count 10
"""

import argparse
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import wandb

from train_helpers import (
    load_config, apply_overrides, set_seed,
    build_model, build_optimizer_and_scheduler,
    wandb_log, compute_bertscore, compute_meteor,
)
from train import run_epoch
from ConvoStyleDataset import ConvoStyleDataset, collate_pad
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Search space 
# Adjust the parameters dict to match the axes you want to explore.
SWEEP_CONFIG = {
    "method": "bayes",  # "bayes" | "random" | "grid"
    "metric": {"name": "cv/mean_val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "batch_size":         {"values": [16, 32]},
        "num_prefix_tokens":  {"values": [10, 20, 40]},
        "warmup_ratio":       {"values": [0.05, 0.1, 0.2]},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "num_mapping_layers": {"values": [4, 8]},
        "dialogue_pooler":    {"values": ["attentive", "last"]},
        "num_turns":    {"values": [0,1,3,5]},
    },
}


# Data helpers 

def _get_conv_ids(meta_path: str) -> np.ndarray:
    return pd.read_parquet(meta_path)["conv_id"].unique()


def _build_fold_loaders(cfg: dict, train_ids: set, val_ids: set):
    effective_turns = max(cfg["num_turns"], 1)
    ds_kwargs = dict(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description"],
        sample_rate=cfg["sample_rate"],
        num_turns=effective_turns,
        max_len_sec=cfg["max_len_sec"],
    )
    train_ds = ConvoStyleDataset(**ds_kwargs, allowed_conv_ids=train_ids)
    val_ds   = ConvoStyleDataset(**ds_kwargs, allowed_conv_ids=val_ids)

    loader_kw = dict(collate_fn=collate_pad, num_workers=cfg["num_workers"], pin_memory=True)
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **loader_kw)
    log.info(f"  {len(train_ds)} train chains  |  {len(val_ds)} val chains")
    return train_loader, val_loader


# Single-fold training 

def _train_fold(
    cfg: dict,
    train_ids: set,
    val_ids: set,
    fold_idx: int,
    sweep_run,              # the W&B run owned by the sweep agent
    device: torch.device,
) -> dict:
    """Train one fold; return dict with best-epoch metrics."""
    set_seed(cfg["seed"] + fold_idx)  # each fold gets a distinct but reproducible seed

    train_loader, val_loader = _build_fold_loaders(cfg, train_ids, val_ids)
    model = build_model(cfg, device, log)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)
    scaler = (
        torch.amp.GradScaler(device=device)
        if cfg["fp16"] and device.type == "cuda" else None
    )

    best: dict = {"val_loss": float("inf"), "epoch": -1}
    global_step = 0
    fp = f"fold_{fold_idx}"  # metric namespace for this fold in W&B

    for epoch in range(cfg["num_epochs"]):
        # pass wandb_run=None so per-step metrics don't spam the sweep run
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, cfg, epoch, global_step, wandb_run=None, is_train=True,
        )

        if (epoch + 1) % cfg["eval_every_n_epochs"] != 0:
            continue

        val_loss, _ = run_epoch(
            model, val_loader, optimizer, scheduler, scaler,
            device, cfg, epoch, global_step, wandb_run=None, is_train=False,
        )

        wandb_log({
            f"{fp}/train_loss": train_loss,
            f"{fp}/val_loss":   val_loss,
            "epoch":            epoch,
        }, step=global_step, run=sweep_run)

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"]    = epoch

    # compute text-quality metrics once, on the final model state
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            audio       = batch["audio"].to(device)
            lengths     = batch["lengths"].to(device)
            text_only   = batch["text_only"].to(device)
            texts       = batch["transcription"]
            speaker_ids = batch["speaker_id"]
            targets     = batch["text_description"]
            if cfg["num_turns"] == 0:
                audio     = torch.zeros_like(audio)
                text_only = torch.ones_like(text_only)
            ctx   = model.scfa(audio, lengths, texts, speaker_ids, text_only)
            vec   = model.pooler(ctx)
            preds = model.style_generator.generate(vec)
            all_preds.extend(preds)
            all_refs.extend([chain[-1] for chain in targets])

    bs  = compute_bertscore(all_preds, all_refs, device=str(device))
    met = compute_meteor(all_preds, all_refs)

    wandb_log({
        f"{fp}/bertscore_f1": bs["bertscore_f1_mean"],
        f"{fp}/meteor":       met["meteor_mean"],
    }, step=global_step, run=sweep_run)

    # free GPU memory before the next fold loads a fresh model
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return {
        "val_loss":     best["val_loss"],
        "bertscore_f1": bs["bertscore_f1_mean"],
        "meteor":       met["meteor_mean"],
    }

def _train_final_and_eval_test(
    cfg: dict,
    trainval_ids: set,
    test_ids: set,
    sweep_run,
    device: torch.device,
) -> dict:
    """Train on all train/val data, evaluate generation metrics on held-out test split."""
    set_seed(cfg["seed"])
    train_loader, test_loader = _build_fold_loaders(cfg, trainval_ids, test_ids)
    model = build_model(cfg, device, log)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)
    scaler = (
        torch.amp.GradScaler(device=device)
        if cfg["fp16"] and device.type == "cuda" else None
    )

    global_step = 0
    for epoch in range(cfg["num_epochs"]):
        _, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, cfg, epoch, global_step, wandb_run=None, is_train=True,
        )

    test_loss, _ = run_epoch(
        model, test_loader, optimizer, scheduler, scaler,
        device, cfg, 0, global_step, wandb_run=None, is_train=False,
    )

    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in test_loader:
            audio       = batch["audio"].to(device)
            lengths     = batch["lengths"].to(device)
            text_only   = batch["text_only"].to(device)
            texts       = batch["transcription"]
            speaker_ids = batch["speaker_id"]
            targets     = batch["text_description"]
            if cfg["num_turns"] == 0:
                audio     = torch.zeros_like(audio)
                text_only = torch.ones_like(text_only)
            ctx   = model.scfa(audio, lengths, texts, speaker_ids, text_only)
            vec   = model.pooler(ctx)
            preds = model.style_generator.generate(vec)
            all_preds.extend(preds)
            all_refs.extend([chain[-1] for chain in targets])

    bs  = compute_bertscore(all_preds, all_refs, device=str(device))
    met = compute_meteor(all_preds, all_refs)

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return {
        "test_loss":     test_loss,
        "bertscore_f1":  bs["bertscore_f1_mean"],
        "meteor":        met["meteor_mean"],
    }


# Sweep function 

def _make_sweep_fn(base_cfg: dict, n_folds: int, 
                   all_conv_ids: np.ndarray, test_ids: set):

    """Return the callable that W&B's agent invokes for each hyperparameter trial."""

    # Precompute folds once so every trial sees the same data curriculum.
    rng = np.random.default_rng(base_cfg["seed"])
    shuffled = all_conv_ids.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)

    def sweep_fn():
        run = wandb.init()
        cfg = deepcopy(base_cfg)

        for key, val in run.config.items():
            if key in cfg:
                cfg[key] = val
        run.config.update({"n_folds": n_folds}, allow_val_change=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fold_metrics = []
        for fold_idx in range(n_folds):
            log.info(f"=== Fold {fold_idx + 1}/{n_folds} ===")
            val_ids   = set(folds[fold_idx])
            train_ids = set(np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx]))
            metrics   = _train_fold(cfg, train_ids, val_ids, fold_idx, run, device)
            fold_metrics.append(metrics)
            log.info(
                f"Fold {fold_idx + 1}  val_loss={metrics['val_loss']:.4f}  "
                f"bertscore_f1={metrics['bertscore_f1']:.4f}  meteor={metrics['meteor']:.4f}"
            )

        # ── aggregate across folds (the sweep optimises this) ─────────────────
        val_losses    = [m["val_loss"]     for m in fold_metrics]
        bert_f1s      = [m["bertscore_f1"] for m in fold_metrics]
        meteor_scores = [m["meteor"]       for m in fold_metrics]

        summary = {
            "cv/mean_val_loss":     float(np.mean(val_losses)),
            "cv/std_val_loss":      float(np.std(val_losses)),
            "cv/mean_bertscore_f1": float(np.mean(bert_f1s)),
            "cv/std_bertscore_f1":  float(np.std(bert_f1s)),
            "cv/mean_meteor":       float(np.mean(meteor_scores)),
            "cv/std_meteor":        float(np.std(meteor_scores)),
        }

        # ── held-out test evaluation (train on all trainval, eval on test) ────
        log.info("=== Final model: training on all train/val folds for test eval ===")
        trainval_ids = set(np.concatenate(folds))
        test_metrics = _train_final_and_eval_test(cfg, trainval_ids, test_ids, run, device)
        log.info(
            f"Test  loss={test_metrics['test_loss']:.4f}  "
            f"bertscore_f1={test_metrics['bertscore_f1']:.4f}  "
            f"meteor={test_metrics['meteor']:.4f}"
        )
        summary.update({
            "test/loss":         test_metrics["test_loss"],
            "test/bertscore_f1": test_metrics["bertscore_f1"],
            "test/meteor":       test_metrics["meteor"],
        })

        run.summary.update(summary)
        wandb_log(summary, step=0, run=run)
        log.info(
            f"CV summary  mean_val_loss={summary['cv/mean_val_loss']:.4f} "
            f"(±{summary['cv/std_val_loss']:.4f})  "
            f"mean_bertscore_f1={summary['cv/mean_bertscore_f1']:.4f}"
        )
        run.finish()


    return sweep_fn


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="W&B hyperparameter sweep with N-fold cross-validation."
    )
    parser.add_argument("--config",   required=True,
                        help="Base config JSON (same format as train.py)")
    parser.add_argument("--n_folds",  type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--sweep_id", default=None,
                        help="Short sweep ID to join an existing sweep instead of creating one. "
                             "entity/project are read from the config.")
    parser.add_argument("--count",    type=int, default=None,
                        help="Max trials this agent will run (default: unlimited)")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE",
                        help="Override base config fields (same syntax as train.py)")
    args = parser.parse_args()

    base_cfg     = load_config(args.config)
    base_cfg     = apply_overrides(base_cfg, args.override)

    all_conv_ids = _get_conv_ids(base_cfg["meta_path"])

    # carve out a 10% held-out test split before any fold construction
    rng_split = np.random.default_rng(base_cfg["seed"])
    shuffled_all = all_conv_ids.copy()
    rng_split.shuffle(shuffled_all)
    n_test       = max(1, int(len(shuffled_all) * 0.10))
    test_ids     = set(shuffled_all[:n_test])
    trainval_ids = shuffled_all[n_test:]
    log.info(
        f"{len(all_conv_ids)} unique conversations → "
        f"{len(trainval_ids)} train/val  |  {len(test_ids)} test (held-out)"
    )

    sweep_fn = _make_sweep_fn(base_cfg, args.n_folds, trainval_ids, test_ids)


    project = base_cfg["wandb_project"]
    entity  = base_cfg.get("wandb_entity")

    if args.sweep_id:
        # join an existing sweep (useful for running multiple agents in parallel)
        qualified = (
            f"{entity}/{project}/{args.sweep_id}" if entity
            else f"{project}/{args.sweep_id}"
        )
        log.info(f"Joining existing sweep: {qualified}")
        sweep_id = qualified
    else:
        sweep_id = wandb.sweep(
            deepcopy(SWEEP_CONFIG),
            project=project,
            entity=entity,
        )
        log.info(f"Created sweep ID: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_fn, count=args.count)


if __name__ == "__main__":
    main()
