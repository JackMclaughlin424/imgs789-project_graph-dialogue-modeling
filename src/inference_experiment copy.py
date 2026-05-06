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

from model.train_helpers import (
    load_config, apply_overrides, wandb_log, assert_no_test_leakage,
    compute_bertscore, compute_meteor, compute_chrf, compute_rouge, compute_tag_f1,
    compute_dist, compute_pred_semantic_sim, _flatten,
    build_model, load_checkpoint, eval_test_by_source, set_seed,
)

from baseline import load_llm, build_system_prompt, build_user_prompt, batch_query_llm, LLM_REPO
from dataset.ConvoStyleDataset import ConvoStyleDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_log_file = os.environ.get("SLURM_JOB_LOG", "inference_test.log")
_fh = logging.FileHandler(_log_file, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_fh)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def build_fewshot_set(train_ds, shuffled, cfg, num_few_shot):
    conv_id_to_chains: dict = {}
    for chain in train_ds._chains:
        cid = chain[-1].get("conv_id")
        conv_id_to_chains.setdefault(cid, []).append(chain)

    ordered_chains = []
    for conv_id in shuffled:
        if conv_id in conv_id_to_chains:
            ordered_chains.extend(conv_id_to_chains[conv_id])

    chains_by_source: dict = {}
    for chain in ordered_chains:
        src = str(chain[-1].get("source", "unknown")).lower()
        chains_by_source.setdefault(src, []).append(chain)

    total = len(ordered_chains)
    rng = np.random.default_rng(cfg["seed"])
    few_shot_chains = []
    allocated = 0
    sources = sorted(chains_by_source)
    for i, src in enumerate(sources):
        pool = chains_by_source[src]
        if i == len(sources) - 1:
            n = max(0, num_few_shot - allocated)
        else:
            n = round(num_few_shot * len(pool) / total)
        n = min(n, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        few_shot_chains.extend(pool[j] for j in idxs)
        allocated += n

    source_counts = {src: sum(1 for c in few_shot_chains if str(c[-1].get("source", "")).lower() == src) for src in sources}
    log.info(f"Baseline: {len(few_shot_chains)} few-shot chains sampled with seed={cfg['seed']}  per-source={source_counts}  (num_turns={cfg['num_turns']})")
    return few_shot_chains


def run_baseline_for_trial(cfg, shuffled, test_chains_by_source, device):
    num_few_shot         = cfg.get("num_few_shot", 25)
    max_new_tokens       = cfg.get("max_new_tokens", 80)
    inference_batch_size = cfg.get("inference_batch_size", 16)
    llm_repo             = cfg.get("llm_repo", LLM_REPO)
    device_str           = str(device)

    train_ds = ConvoStyleDataset(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "conv_id", "source"],
        num_turns=int(cfg["num_turns"]),
        max_len_sec=float(cfg["max_len_sec"]),
        allowed_conv_ids=set(shuffled),
    )

    few_shot_chains = build_fewshot_set(train_ds, shuffled, cfg, num_few_shot)

    log.info(f"Baseline: loading LLM ({llm_repo}) on {device_str}...")
    tokenizer, llm = load_llm(device_str, repo=llm_repo)
    system_prompt = build_system_prompt(few_shot_chains)

    for src, chains in test_chains_by_source.items():
        run_baseline = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            config={**cfg, "run_type": "baseline", "dataset": src},
            name=f"infer_baseline_{src}",
            settings=wandb.Settings(console="off", init_timeout=300),
        )

        full_prompts  = [f"{system_prompt}\n\n---\n\n{build_user_prompt(c)}" for c in chains]
        ground_truths = [(c[-1].get("text_description") or "").strip() for c in chains]

        log.info(f"Baseline/{src}: running inference on {len(full_prompts)} chains...")
        t0_infer = time.time()
        predictions = batch_query_llm(
            tokenizer, llm, full_prompts, device_str,
            max_new_tokens=max_new_tokens,
            batch_size=inference_batch_size,
        )
        inference_time = time.time() - t0_infer
        log.info(f"Baseline/{src}: inference_time={inference_time:.1f}s")

        for i, (pred, ref, chain) in enumerate(zip(predictions[:3], ground_truths[:3], chains[:3])):
            txt = " | ".join(t.get("transcription", "") for t in chain if t.get("transcription"))
            log.info(f"  [Baseline/{src} Sample {i+1}]")
            log.info(f"    Dialogue : {txt}")
            log.info(f"    Predicted: {pred}")
            log.info(f"    Reference: {ref}")

        bs_metrics    = compute_bertscore(predictions, ground_truths, device=device_str)
        met_metrics   = compute_meteor(predictions, ground_truths)
        chrf_metrics  = compute_chrf(predictions, ground_truths)
        rouge_metrics = compute_rouge(predictions, ground_truths)
        tag_metrics   = compute_tag_f1(predictions, ground_truths, src)
        div_metrics   = compute_dist(predictions)
        psem_metrics  = compute_pred_semantic_sim(predictions, device=device_str)

        all_metrics = {**_flatten(bs_metrics), **_flatten(met_metrics), **_flatten(chrf_metrics), **_flatten(rouge_metrics), **_flatten(tag_metrics), **div_metrics, **psem_metrics}
        all_metrics["inference_time_s"] = inference_time

        log.info(
            f"Baseline/{src}  bertscore_f1={all_metrics['bertscore_f1']:.4f}  "
            f"meteor={all_metrics['meteor']:.4f}  chrf={all_metrics['chrf']:.4f}  "
            f"tag_f1={all_metrics['tag_f1_overall']:.4f}"
        )
        summary = {f"baseline/{src}/{k}": v for k, v in all_metrics.items()}
        run_baseline.summary.update(summary)
        wandb_log(summary, step=0, run=run_baseline)
        run_baseline.finish()

    del llm, tokenizer
    gc.collect()
    if device_str == "cuda":
        torch.cuda.empty_cache()



def run_inference_trial(cfg, checkpoint_path, test_chains_by_source, device):
    """Load a trained checkpoint once, then evaluate each source under its own wandb run."""
    set_seed(cfg["seed"])

    model = build_model(cfg, device, log)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        log.info(f"Loaded checkpoint: {checkpoint_path}  (epoch {ckpt.get('epoch')}, step {ckpt.get('step')})")
    else:
        model.load_state_dict(ckpt)
        log.info(f"Loaded final model state_dict: {checkpoint_path}")
    model.eval()

    for src, chains in test_chains_by_source.items():
        run_test = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            config={**cfg, "checkpoint": checkpoint_path, "run_type": "test", "dataset": src},
            name=f"infer_test_{src}",
            settings=wandb.Settings(console="off", init_timeout=300),
        )

        src_metrics = eval_test_by_source(model, cfg, {src: chains}, device, log)
        src_m = src_metrics[src]
        inference_time = src_m.get("inference_time_s", 0.0)

        log.info(
            f"Test/{src}  bertscore_f1={src_m['bertscore_f1']:.4f}  "
            f"meteor={src_m['meteor']:.4f}  chrf={src_m['chrf']:.4f}  "
            f"tag_f1={src_m['tag_f1_overall']:.4f}"
        )
        summary = {f"test/{src}/{k}": v for k, v in src_m.items()}
        summary["trial/inference_time_s"] = inference_time
        run_test.summary.update(summary)
        wandb_log(summary, step=0, run=run_test)
        run_test.finish()

    del model
    gc.collect()
    torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the test set (no training)."
    )
    parser.add_argument("--config",       required=True,
                        help="Config JSON matching the one used during training.")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to a saved model checkpoint (.pt file).")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the few-shot LLM baseline evaluation.")
    parser.add_argument("--override",     nargs="*", metavar="KEY=VALUE",
                        help="Override base config fields.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Run config: {json.dumps(cfg, indent=2, default=str)}")

    test_chains_by_source, test_conv_ids = ConvoStyleDataset.make_fixed_test_split(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=int(cfg["sample_rate"]),
        max_len_sec=float(cfg["max_len_sec"]),
        num_turns=int(cfg["num_turns"]),
    )

    meta         = pd.read_parquet(cfg["meta_path"], columns=["conv_id"])
    trainval_arr = np.array([c for c in meta["conv_id"].unique() if c not in test_conv_ids])
    assert_no_test_leakage(set(trainval_arr), test_conv_ids)

    n_test = sum(len(v) for v in test_chains_by_source.values())
    log.info(
        f"Split sizes  trainval_conv_ids={len(trainval_arr)}  test_chains={n_test}  "
        + "  ".join(f"{src}={len(c)}" for src, c in test_chains_by_source.items())
    )

    rng      = np.random.default_rng(cfg["seed"])
    shuffled = trainval_arr.copy()
    rng.shuffle(shuffled)

    run_inference_trial(cfg, args.checkpoint, test_chains_by_source, device)

    if not args.skip_baseline:
        run_baseline_for_trial(cfg, shuffled, test_chains_by_source, device)

    gc.collect()
    torch.cuda.empty_cache()




if __name__ == "__main__":
    main()
