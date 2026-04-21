"""
sweep.py  --config path/to/sweep_base_config.json
          --sweep_values path/to/sweep_values.json
          [options]

Runs a W&B hyperparameter sweep where each trial performs N-fold cross-validation.
Conversations are split at the conv_id level (same anti-leakage logic as train.py).

Usage
-----
# Start a new sweep and run one agent:
    python sweep.py --config default_sweep_config.json --sweep_values default_sweep_values.json --n_folds 5 --count 20

# Join an existing sweep from a second machine:
    python sweep.py --config default_sweep_config.json --sweep_values default_sweep_values.json --sweep_id <SHORT_ID> --count 10
"""


import argparse
import logging
from copy import deepcopy
import gc
import numpy as np
import pandas as pd
import torch
import wandb
import json
import os
import time
import itertools

from model.train_helpers import (
    load_config, apply_overrides, set_seed,
    build_model, build_optimizer_and_scheduler,
    wandb_log, compute_bertscore, compute_meteor,
    compute_chrf, compute_rouge, compute_tag_f1, eval_test_by_source,
    assert_no_test_leakage
)

from train import run_epoch
from dataset.ConvoStyleDataset import ConvoStyleDataset, collate_pad
from torch.utils.data import DataLoader

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



# Data helpers 


def _build_fold_loaders(cfg: dict, train_ids: set, val_ids: set):
    ds_kwargs = dict(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        num_turns=cfg["num_turns"],
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
    global_step: int
) -> dict:
    """Train one fold; return dict with best-epoch metrics."""
    set_seed(cfg["seed"] + fold_idx)  # each fold gets a distinct but reproducible seed

    train_loader, val_loader = _build_fold_loaders(cfg, train_ids, val_ids)
    model = build_model(cfg, device, log)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)
    

    best: dict = {"val_loss": float("inf"), "epoch": -1}
    
    fp = f"fold_{fold_idx+1}"

    patience         = cfg.get("early_stopping_patience", 3)
    min_delta        = cfg.get("early_stopping_min_delta", 1e-4)
    MIN_EPOCH        = 5
    epochs_no_improve = 0

    fold_start_time = time.time()
    for epoch in range(cfg["num_epochs"]):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=None, log_handler=log
            , is_train=True, use_tqdm=False
        )

        if (epoch + 1) % cfg["eval_every_n_epochs"] != 0:
            continue

        val_loss, _ = run_epoch(
            model, val_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=None, log_handler=log
            , is_train=False, use_tqdm=False
        )

        wandb_log({
            f"{fp}/train_loss": train_loss,
            f"{fp}/val_loss":   val_loss,
            "epoch":            epoch,
        }, step=global_step, run=sweep_run)

        if val_loss < best["val_loss"] - min_delta:
            best["val_loss"] = val_loss
            best["epoch"]    = epoch
            epochs_no_improve = 0
        elif patience > 0 and epoch >= MIN_EPOCH:
            epochs_no_improve += 1

        if patience > 0 and epoch >= MIN_EPOCH and epochs_no_improve >= patience:
            log.info(f"  Fold {fold_idx+1}: early stop at epoch {epoch+1} (no improvement for {patience} evals)")
            break

        gc.collect()

    elapsed = time.time() - fold_start_time
    fmt = lambda s: f"{int(s)//60:02d}:{int(s)%60:02d}"
    log.info(
        f"Fold {fold_idx+1}: Trained for {epoch+1} epochs. Total time: {fmt(elapsed)}"
        
    )

    # reclaim memory before next fold loads
    del train_loader, val_loader, optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
  
    
    return {
        "val_loss":     best["val_loss"],
        "best_epoch":   best["epoch"],
    }, global_step



def _train_final_and_eval_test(
    cfg: dict,
    trainval_ids: set,
    test_chains_by_source: dict,
    sweep_run,
    device: torch.device,
    global_step: int = 0,
) -> dict:
    """Train on all train/val data, then evaluate per source on the held-out test chains."""
    set_seed(cfg["seed"])

    # Build data loaders
    ds_kwargs = dict(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        num_turns=cfg["num_turns"],
        max_len_sec=cfg["max_len_sec"],
    )
    loader_kw  = dict(collate_fn=collate_pad, num_workers=cfg["num_workers"], pin_memory=True)
    g          = torch.Generator().manual_seed(cfg["seed"])
    train_ds   = ConvoStyleDataset(**ds_kwargs, allowed_conv_ids=trainval_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g, **loader_kw)


    model = build_model(cfg, device, log)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)

    for epoch in range(cfg["num_epochs"]):
        _, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run=None, log_handler=log,
            is_train=True, use_tqdm=False
        )

    log.info("Evaluating generation...")
    model.eval()
    metrics = eval_test_by_source(
        model, cfg, test_chains_by_source, device, log
    )


    del train_loader
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, global_step





# Sweep function 

def _make_sweep_fn(base_cfg: dict, n_folds: int, overrides: list | None = None):


    """Return the callable that W&B's agent invokes for each hyperparameter trial."""

    def sweep_fn():
        gc.collect()
        torch.cuda.empty_cache()
        # sweep.py, line 242
        run = wandb.init(settings=wandb.Settings(console="off", init_timeout=300))

        cfg = deepcopy(base_cfg)

        for key, val in run.config.items():
            cfg[key] = val

        apply_overrides(cfg, overrides)


        # expand shared sweep param into per-encoder keys
        if "num_unfrozen_embedder_layers" in run.config:
            n = run.config["num_unfrozen_embedder_layers"]
            cfg["num_unfrozen_bert"]  = n
            cfg["num_unfrozen_wavlm"] = n

        run.config.update({"n_folds": n_folds}, allow_val_change=True)
        log.info(f"Run config: {json.dumps(cfg, indent=2, default=str)}")


        test_chains_by_source, test_conv_ids = ConvoStyleDataset.make_fixed_test_split(
            h5_path=cfg["h5_path"],
            meta_path=cfg["meta_path"],
            meta_columns=["transcription", "text_description", "source"],
            sample_rate=cfg["sample_rate"],
            max_len_sec=cfg["max_len_sec"],
            num_turns=cfg["num_turns"],
        )



        meta         = pd.read_parquet(cfg["meta_path"], columns=["conv_id"])
        trainval_arr = np.array([c for c in meta["conv_id"].unique() if c not in test_conv_ids])

        # double check no leakage
        assert_no_test_leakage(set(trainval_arr), test_conv_ids)

        n_test = sum(len(v) for v in test_chains_by_source.values())
        log.info(
            f"Split sizes  trainval_conv_ids={len(trainval_arr)}  test_chains={n_test}  "
            + "  ".join(f"{src}={len(c)}" for src, c in test_chains_by_source.items())
        )


        rng      = np.random.default_rng(cfg["seed"])
        shuffled = trainval_arr.copy()
        rng.shuffle(shuffled)

        if n_folds <= 1:
            # 80/20 single-fold fallback — avoids degenerate empty train/val split
            split = int(len(shuffled) * 0.8)
            fold_splits = [(set(shuffled[:split]), set(shuffled[split:]))]
        else:
            chunks = np.array_split(shuffled, n_folds)
            fold_splits = [
                (set(np.concatenate([chunks[j] for j in range(n_folds) if j != i])), set(chunks[i]))
                for i in range(n_folds)
            ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global_step = 0
        fold_metrics = []
        for fold_idx, (train_ids, val_ids) in enumerate(fold_splits):
            log.info(f"=== Fold {fold_idx+1}/{len(fold_splits)} ===")
            metrics, global_step = _train_fold(cfg, train_ids, val_ids, fold_idx, run, device, global_step)
            fold_metrics.append(metrics)
            log.info(f"Fold {fold_idx+1}  val_loss={metrics['val_loss']:.4f}  ")


        # aggregate across folds (the sweep optimises this)
        val_losses    = [m["val_loss"]     for m in fold_metrics]
        best_epochs    = [m["best_epoch"]     for m in fold_metrics]

        
        # from training, average best epoch to stop at for test
        mean_best_epoch = int(round(np.mean([m["best_epoch"] for m in fold_metrics])))
        log.info(f"Mean best epoch across folds: {mean_best_epoch}")

        cv_summary = {
            "cv/mean_val_loss":      float(np.mean(val_losses)),
            "cv/std_val_loss":       float(np.std(val_losses)),
            "cv/mean_best_epoch":    float(np.mean(best_epochs)),
            "cv/std_best_epoch":     float(np.std(best_epochs)),
        }

        wandb_log(cv_summary, step=global_step, run=run)
        log.info(
            f"CV summary  mean_val_loss={cv_summary['cv/mean_val_loss']:.4f} "
            f"(±{cv_summary['cv/std_val_loss']:.4f})  "
            f"mean_best_epoch={cv_summary['cv/mean_best_epoch']:.1f}"
        )

        # held-out test evaluation (train on all trainval, eval on test) 
        log.info("=== Final model: training on all train/val folds for test eval ===")
        trainval_ids = set(shuffled)


        
        test_metrics, global_step = _train_final_and_eval_test(
            cfg, trainval_ids, test_chains_by_source, run, device, global_step
        )

        
        for src, src_m in test_metrics.items():
            log.info(
                f"Test/{src}  bertscore_f1={src_m['bertscore_f1']:.4f}  "
                f"meteor={src_m['meteor']:.4f}  chrf={src_m['chrf']:.4f}  "
                f"tag_f1={src_m['tag_f1_overall']:.4f}"
            )
            
            test_summary = {f"test/{src}/{k}": v for k, v in src_m.items()}
            run.summary.update(test_summary)      # move inside loop
            wandb_log(test_summary, step=global_step, run=run)


        run.finish()


    return sweep_fn




# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="W&B hyperparameter sweep with N-fold cross-validation."
    )
    parser.add_argument("--config",       required=True,
                        help="Base config JSON with fixed hyperparameters (e.g. default_sweep_config.json).")
    parser.add_argument("--sweep_values", required=True,
                        help="W&B sweep search-space JSON (e.g. default_sweep_values.json).")
    parser.add_argument("--n_folds",     type=int, default=5,
                        help="CV folds (default: 5). Use 1 for a single 80/20 holdout split.")

    parser.add_argument("--sweep_id",    default=None,
                        help="Short sweep ID to join an existing sweep instead of creating one.")
    parser.add_argument("--count",       type=int, default=None,
                        help="Max trials this agent will run (default: unlimited).")
    parser.add_argument("--override",    nargs="*", metavar="KEY=VALUE",
                        help="Override base config fields (same syntax as train.py).")
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    with open(args.sweep_values) as f:
        sweep_config = json.load(f)

    sweep_fn = _make_sweep_fn(base_cfg, args.n_folds, args.override)


    project = base_cfg["wandb_project"]
    entity  = base_cfg.get("wandb_entity")

    if args.sweep_id:
        sweep_id = args.sweep_id
        log.info(f"Joining existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(deepcopy(sweep_config), project=project, entity=entity)
        log.info(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_fn, count=args.count,
                project=project, entity=entity)



if __name__ == "__main__":
    main()
