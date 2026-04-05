import nltk
from nltk.translate.meteor_score import meteor_score as _meteor
from bert_score import score as _bert_score
import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    Wav2Vec2FeatureExtractor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from ConvoStyleDataset import ConvoStyleDataset, collate_pad
from DialogueEncoder import (
    DualModalityEmbedder,
    SCFA,
    DialoguePooler,
    SelfAttentivePooling,
    ModalityEncoder,
)
from StylePromptGenerator import (
    StylePromptHead,
    StylePromptGenerator,
    SCFAWithStyleHead,
    LLM_REPO,
    LLM_DIM,
)

# defaults

DEFAULTS: Dict[str, Any] = {
    # data
    "h5_path":               None,        # required
    "meta_path":             None,        # required
    "output_dir":            "runs/exp",
    "num_turns":             5,           # 0 means text-only (script-only mode, no audio)
    "max_len_sec":           15,        # trim/pad audio to this duration

    # model dimensions
    "d_model":               768,         # must be divisible by 3 (192, 384, or 768)
    "num_ctx_layers":        2,           # ContextAwareTransformer depth
    "num_spk_layers":        2,           # SpeakerAwareTransformer depth
    "dim_feedforward":       2048,
    "nhead":                 8,           # must divide d_model AND d_model//3

    # prefix / LLM
    "llm_repo":              LLM_REPO,
    "llm_dim":               LLM_DIM,
    "num_prefix_tokens":     20,          # K in StyleCap notation
    "num_mapping_layers":    4,
    "mapping_nhead":         8,
    "system_prompt":         "",        # "" | "Speaking style:" | custom string
    "max_prompt_tokens": 128,   # system prompt text -- used at both train and inference
    "max_style_desc_tokens": 128,  # ground-truth target style descriptions -- training only
    "max_new_tokens": 80,       # generation budget -- inference only

    # freezing
    "num_unfrozen_bert":     0,           # how many BERT encoder layers to unfreeze (from top)
    "num_unfrozen_wavlm":    0,           # how many WavLM encoder layers to unfreeze (from top)

    # pooling
    "dialogue_pooler":       "last", # "attentive" | "last"

    # training
    "batch_size":            16,
    "num_epochs":            10,
    "learning_rate":         5e-5,
    "weight_decay":          1e-3,
    "grad_clip":             1.0,         # max gradient norm; None to disable
    "warmup_ratio":          0.1,         # fraction of total steps used for LR warmup
    "lr_schedule":           "cosine",    # "cosine" | "linear" | "constant"
    "dropout":               0.1,

    # validation / checkpointing
    "val_split":             0.1,         # fraction of data held out for validation
    "eval_every_n_epochs":   1,
    "save_every_n_epochs":   1,
    "keep_last_n_ckpts":     2,           # older checkpoints are deleted
    "early_stopping_patience": 3,   # epochs without improvement before stopping; 0 to disable
    "early_stopping_min_delta": 1e-4,

    # reproducibility / efficiency
    "seed":                  42,
    "num_workers":           4,
    "sample_rate":           16_000,

    # logging
    "log_every_n_steps":     10,
    "run_name":              None,        # optional label shown in log lines

    # wandb -- all optional; set use_wandb=false to disable entirely
    "use_wandb":             True,
    "wandb_project":         "style-prompt-gen",
    "wandb_entity":          "jdm8943-rochester-institute-of-technology",        # your W&B username or team; None uses default
}

REQUIRED = {"h5_path", "meta_path"}

VALID = {
    "d_model":          {192, 384, 768},
    "dialogue_pooler":  {"attentive", "last"},
    "lr_schedule":      {"cosine", "linear", "constant"},
    "num_turns":        set(range(0, 6)),   # 0-5 inclusive
    "num_prefix_tokens": {10, 20, 40},
    "batch_size":       {4, 8, 16, 32},
}


# Config loading + validation

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        user = json.load(f)

    cfg = {**DEFAULTS, **user}

    # check required fields
    for key in REQUIRED:
        if cfg[key] is None:
            raise ValueError(f"Config is missing required field: '{key}'")

    # check enum fields -- only when the value is one we explicitly constrain
    for key, allowed in VALID.items():
        if key in cfg and cfg[key] not in allowed:
            raise ValueError(
                f"Config field '{key}' = {cfg[key]!r} is not one of {sorted(allowed)}"
            )

    # d_model must be divisible by 3 (splits into 3 sub-streams in SCFA)
    if cfg["d_model"] % 3 != 0:
        raise ValueError(f"d_model must be divisible by 3, got {cfg['d_model']}")

    # nhead must divide both d_model and d_model // 3
    d_sub = cfg["d_model"] // 3 # d_sub is dimensionality for the context transformer and 2 speaker transformers
    if cfg["d_model"] % cfg["nhead"] != 0 or d_sub % cfg["nhead"] != 0:
        raise ValueError(
            f"nhead={cfg['nhead']} must divide d_model={cfg['d_model']} "
            f"AND d_model//3={d_sub}"
        )

    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides, log=None):
    if not overrides:
        return cfg
    for item in overrides:
        key, _, raw = item.partition("=")
        if key not in cfg:
            raise ValueError(f"Unknown config key in override: '{key}'")
        # cast to the same type as the default
        default_val = DEFAULTS.get(key)
        if default_val is None:
            cfg[key] = raw  # can't infer type; keep as string
        elif isinstance(default_val, bool):
            cfg[key] = raw.lower() in ("1", "true", "yes")
        elif isinstance(default_val, int):
            cfg[key] = int(raw)
        elif isinstance(default_val, float):
            cfg[key] = float(raw)
        else:
            cfg[key] = raw

        if log is not None:
            log.info(f"Override: {key} = {cfg[key]!r}")
        else:
            print(f"Override: {key} = {cfg[key]!r}")
    return cfg

# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model construction

def _unfreeze_top_n_layers(model: nn.Module, layer_attr: str, n: int, log):
    """Freeze everything first, then selectively unfreeze the top n encoder layers."""
    for p in model.parameters():
        p.requires_grad = False

    if n <= 0:
        return

    obj = model
    for part in layer_attr.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            log.warning(f"Could not find attribute '{layer_attr}' on {type(model).__name__} -- skipping unfreeze")
            return
    layers = obj


    # unfreeze from the top (last layers first, which are most task-relevant)
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def build_text_encoder(cfg: Dict[str, Any], device: torch.device, log):
    """Load BERT, freeze all layers, then optionally unfreeze the top N."""
    from transformers import AutoModel, AutoTokenizer

    BERT_REPO = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    model = AutoModel.from_pretrained(BERT_REPO).to(device)

    _unfreeze_top_n_layers(model, "encoder.layer", cfg["num_unfrozen_bert"], log)

    n = cfg["num_unfrozen_bert"]
    log.info(f"BERT: {n} encoder layer(s) unfrozen")
    return model, tokenizer


def build_audio_encoder(cfg: Dict[str, Any], device: torch.device, log):
    """Load WavLM-base-plus, freeze all layers, then optionally unfreeze the top N."""
    from transformers import WavLMModel, AutoFeatureExtractor

    WAVLM_REPO = "microsoft/wavlm-base-plus"
    processor = AutoFeatureExtractor.from_pretrained(WAVLM_REPO)
    model = WavLMModel.from_pretrained(WAVLM_REPO).to(device)

    _unfreeze_top_n_layers(model, "encoder.layers", cfg["num_unfrozen_wavlm"], log)

    n = cfg["num_unfrozen_wavlm"]
    log.info(f"WavLM: {n} encoder layer(s) unfrozen")
    return model, processor


def build_model(cfg: Dict[str, Any], device: torch.device, log) -> SCFAWithStyleHead:
    log.info("Building model...")

    text_backbone, tokenizer = build_text_encoder(cfg, device, log)
    audio_backbone, processor = build_audio_encoder(cfg, device, log)


    embedder = DualModalityEmbedder(
        text_encoder_model_pretrained=text_backbone,
        audio_encoder_model_pretrained=audio_backbone,
        tokenizer=tokenizer,
        processor=processor,
        SAMPLE_RATE=cfg["sample_rate"],
    )

    scfa = SCFA(
        max_turns=cfg["num_turns"],
        embedder=embedder,
        d_model=cfg["d_model"],
        num_ctx_layers=cfg["num_ctx_layers"],
        num_spk_layers=cfg["num_spk_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        nhead=cfg["nhead"],
        dropout=cfg["dropout"],
    ).to(device)

    # 4 * d_model because SCFA cats [z_audio, z_text, z_audio_fused, z_text_fused]
    pooler = DialoguePooler(
        d_model=cfg["d_model"] * 4,
        mode=cfg["dialogue_pooler"],
    ).to(device)

    # LLM embedding dim is fixed at LLM_DIM
    head = StylePromptHead(
        d_model=cfg["d_model"],
        num_prefix_tokens=cfg["num_prefix_tokens"],
        llm_dim=cfg["llm_dim"],
        num_mapping_layers=cfg["num_mapping_layers"],
        nhead=cfg["mapping_nhead"],
        dropout=cfg["dropout"],
    ).to(device)

    tokenizer_llm = AutoTokenizer.from_pretrained(cfg["llm_repo"])
    llm = AutoModelForCausalLM.from_pretrained(cfg["llm_repo"], dtype=torch.bfloat16).to(device)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token


    generator = StylePromptGenerator(
        style_head=head,
        tokenizer=tokenizer_llm,
        llm=llm,
        system_prompt=cfg["system_prompt"],
        max_new_tokens=cfg["max_new_tokens"],
        max_prompt_tokens=cfg["max_prompt_tokens"],
    ).to(device)

    model = SCFAWithStyleHead(scfa=scfa, pooler=pooler, style_generator=generator)
    return model


# Data

def build_dataloaders(cfg: Dict[str, Any], log):
    # num_turns == 0 means script-only (text only), represented as 1 turn with no audio
    effective_turns = max(cfg["num_turns"], 1)

    train_ds, val_ds = ConvoStyleDataset.train_val_split(
        val_split=cfg["val_split"],
        seed=cfg["seed"],
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description"], # text_description is GT style description
        sample_rate=cfg["sample_rate"],
        num_turns=effective_turns,
        max_len_sec=cfg["max_len_sec"],
    )

    loader_kwargs = dict(
        collate_fn=collate_pad,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    log.info(f"Chains: {len(train_ds)} train  |  {len(val_ds)} val")
    return train_loader, val_loader, train_ds



# Optimizer + scheduler

def build_optimizer_and_scheduler(model: SCFAWithStyleHead, cfg: Dict[str, Any], total_steps: int, log):
    # only optimize parameters that require gradients
    # (LLM is frozen inside StylePromptGenerator; backbone layers may be partially frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    if cfg["lr_schedule"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif cfg["lr_schedule"] == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:  # constant
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, warmup_steps)

    return optimizer, scheduler



# Checkpoint helpers

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, cfg, out_dir: Path, log):
    ckpt = {
        "epoch":     epoch,
        "step":      step,
        "loss":      loss,
        "cfg":       cfg,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = out_dir / f"ckpt_epoch{epoch:04d}_step{step:07d}.pt"
    torch.save(ckpt, path)
    log.info(f"Saved checkpoint: {path}")
    return path


def prune_old_checkpoints(out_dir: Path, keep: int, log):
    ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()
        log.info(f"Removed old checkpoint: {old}")


def load_checkpoint(path: str, log, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Resumed from checkpoint: {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]



# W&B helpers
 
def wandb_init(cfg: Dict[str, Any], log):
    if not cfg.get("use_wandb", True):
        return None

    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed -- skipping W&B tracking.")
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        log.warning("WANDB_API_KEY not set -- skipping W&B tracking.")
        return None

    # explicitly authenticate so wandb never falls through to the interactive prompt
    wandb.login(key=api_key, relogin=False)

    try:
        run = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            name=cfg.get("run_name"),
            config=cfg,
            resume="allow",
        )
        log.info(f"W&B run: {run.url}")
        return run
    except Exception as e:
        log.warning(f"W&B init failed ({e}) -- continuing without tracking.")
        return None
 
 
def wandb_log(metrics: dict, step: int, run):
    """Log a dict of metrics. No-ops if W&B is disabled."""
    if run is None:
        return
    run.log(metrics, step=step)
 
 
def wandb_finish(run):
    if run is not None:
        run.finish()


# Evaluation metrics

def compute_bertscore(
    preds: List[str],
    refs: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute BERTScore (P, R, F1) mean and std over a batch of predictions."""
    
    P, R, F1 = _bert_score(preds, refs, lang="en", device=device, verbose=False)
    return {
        "bertscore_precision_mean": P.mean().item(),
        "bertscore_precision_std":  P.std().item(),
        "bertscore_recall_mean":    R.mean().item(),
        "bertscore_recall_std":     R.std().item(),
        "bertscore_f1_mean":        F1.mean().item(),
        "bertscore_f1_std":         F1.std().item(),
    }


def compute_meteor(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """Compute METEOR mean and std over a batch of predictions."""
   
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    scores = np.array([
        _meteor([ref.split()], pred.split())
        for pred, ref in zip(preds, refs)
    ])
    return {
        "meteor_mean": float(scores.mean()) if len(scores) else 0.0,
        "meteor_std":  float(scores.std())  if len(scores) else 0.0,
    }
