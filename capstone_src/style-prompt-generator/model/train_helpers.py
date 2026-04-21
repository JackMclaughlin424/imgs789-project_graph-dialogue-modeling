import nltk
from nltk.translate.meteor_score import meteor_score as _meteor
from bert_score import score as _bert_score
from sacrebleu.metrics import CHRF as _CHRF
from rouge_score import rouge_scorer as _rouge_scorer


import argparse
import json
import logging
import math
import os
import random
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from dataset.ConvoStyleDataset import ConvoStyleDataset, collate_pad
from model.DialogueEncoder import (
    DualModalityEmbedder,
    SCFA,
    DialoguePooler,
    SelfAttentivePooling,
    ModalityEncoder,
)
from model.StylePromptGenerator import (
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
    "num_turns":             5,           # >=1; anchor (last turn) is always text-only
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
    "log_every_n_steps":     50,
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
    "num_turns":        set(range(1, 6)),   # 1-5 inclusive
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
    llm = AutoModelForCausalLM.from_pretrained(cfg["llm_repo"], torch_dtype=torch.bfloat16).to(device)
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
    train_ds, val_ds = ConvoStyleDataset.train_val_split(
        val_split=cfg["val_split"],
        seed=cfg["seed"],
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description"],
        sample_rate=cfg["sample_rate"],
        num_turns=cfg["num_turns"],
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


def prune_old_checkpoints(out_dir: Path, keep: int, log, wandb_run=None):
    ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
    for old in ckpts[:-keep]:
        # parse epoch from filename to match against artifact metadata
        epoch_from_name = None
        try:
            epoch_from_name = int(old.stem.split("_")[1].replace("epoch", ""))
        except (IndexError, ValueError):
            pass

        old.unlink()
        log.info(f"Removed old checkpoint: {old}")

        if wandb_run is not None and epoch_from_name is not None:
            try:
                import wandb
                api = wandb.Api()
                artifact_name = f"{wandb_run.entity}/{wandb_run.project}/checkpoint-{wandb_run.id}"
                for av in api.artifacts(type_name="model", name=artifact_name):
                    if av.metadata.get("epoch") == epoch_from_name:
                        av.delete(delete_aliases=True)
                        log.info(f"Deleted W&B artifact version for epoch {epoch_from_name}: {av.version}")
                        break
            except Exception as e:
                log.warning(f"Could not delete W&B artifact for epoch {epoch_from_name}: {e}")



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

def compute_chrf(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """Compute CHrF++ mean and std over a batch of predictions."""
    chrf = _CHRF(word_order=2)  # word_order=2 gives CHrF++
    scores = np.array([
        chrf.sentence_score(pred, [ref]).score / 100.0  # normalize to [0, 1]
        for pred, ref in zip(preds, refs)
    ])
    return {
        "chrf_mean": float(scores.mean()) if len(scores) else 0.0,
        "chrf_std":  float(scores.std())  if len(scores) else 0.0,
    }


def compute_rouge(
    preds: List[str],
    refs: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-L mean and std over a batch of predictions."""
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = np.array([
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(preds, refs)
    ])
    return {
        "rougeL_mean": float(scores.mean()) if len(scores) else 0.0,
        "rougeL_std":  float(scores.std())  if len(scores) else 0.0,
    }



# CUSTOM TAG-LEVEL F1 metric

VOCAB_REF: Dict[str, Any] = {
  "expresso": {
    "accent": [
      "american"
    ],
    "gender": [
      "female",
      "male"
    ],
    "intrinsic_tags": [
    #   "american",  excluded since already in another category
      "authoritative",
      "booming",
      "crisp",
      "enunciated",
      "flowing",
      "loud",
      "nasal",
      "shrill",
      "silky",
      "singsong"
    ],
    "situational_tags": [
        'angry',
        'animated',
        'awed',
        'bored',
        'calm',
        'confused',
        'desirous',
        'disgusted',
        'enunciated',
        # 'fast',
        'happy',
        'laughing',
        'loud',
        'passive',
        'saddened',
        'sarcastic',
        'scared',
        'sleepy',
        'sympathetic',
        'whispered'
        ],
    "noise": [
      "clean environment",
      "noisy environment",
      "environment balanced"
    ],
    "pitch": [
      "high-pitched"
    ],
    "speaking_rate": [
      "fast speed",
      "measured speed",
      "slow speed"
    ]
  },
  "styletalk": {
    "emotion": [
      "cheerful tone",
      "excited tone",
      "friendly tone",
      "hopeful tone",
      "neutral tone",
      "sad tone",
      "unfriendly tone"
    ],
    "speaking_rate": [
      "fast speed",
      "normal speed",
      "slow speed"
    ],
    "volume": [
      "loud volume",
      "normal volume",
      "quiet volume"
    ]
  }
}

_PATTERN_CACHE: Dict[str, Any] = {}

def _load_tag_categories(source: str) -> Dict[str, Any]:
    """Returns {category: compiled_regex} for the given source, compiled once."""
    if source not in _PATTERN_CACHE:
        import re
        _PATTERN_CACHE[source] = {
            cat: re.compile(
                r'\b(' + '|'.join(re.escape(t) for t in sorted(tags, key=len, reverse=True)) + r')\b',
                re.IGNORECASE,
            )
            for cat, tags in VOCAB_REF.get(source, {}).items()
        }
    return _PATTERN_CACHE[source]



def _tags_present(pattern, text: str) -> set:
    return set(m.lower() for m in pattern.findall(text))

def _f1_sets(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    p = tp / len(pred)
    r = tp / len(ref)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_tag_f1(
    preds: List[str],
    refs: List[str],
    test_dataset_source: str,
) -> Dict[str, float]:
    """Per-category tag F1 using word-boundary substring matching against the vocab."""
    categories = _load_tag_categories(test_dataset_source)

    cat_scores: Dict[str, List[float]] = {cat: [] for cat in categories}
    overall_scores: List[float] = []

    for pred, ref in zip(preds, refs):
        all_pred, all_ref = set(), set()
        for cat, tags in categories.items():
            pred_tags = _tags_present(tags, pred)
            ref_tags  = _tags_present(tags, ref)
            cat_scores[cat].append(_f1_sets(pred_tags, ref_tags))
            all_pred |= pred_tags
            all_ref  |= ref_tags
        overall_scores.append(_f1_sets(all_pred, all_ref))

    metrics: Dict[str, float] = {}
    for cat, scores in cat_scores.items():
        arr = np.array(scores)
        metrics[f"tag_f1_{cat}_mean"] = float(arr.mean()) if len(arr) else 0.0
        metrics[f"tag_f1_{cat}_std"]  = float(arr.std())  if len(arr) else 0.0

    arr = np.array(overall_scores)
    metrics["tag_f1_overall_mean"] = float(arr.mean()) if len(arr) else 0.0
    metrics["tag_f1_overall_std"]  = float(arr.std())  if len(arr) else 0.0

    return metrics



def _flatten(d: dict) -> dict:
    """Strip _mean suffix → bare name; keep _std keys as-is."""
    return {(k[:-5] if k.endswith("_mean") else k): v for k, v in d.items()}



def eval_test_by_source(
    model,
    cfg: dict,
    test_chains_by_source: dict,   # dict[src, list[chain]] from make_fixed_test_split
    device: torch.device,
    log
) -> dict:
    """Evaluate model on each source's fixed test chains; returns per-source metrics."""
    loader_kw = dict(collate_fn=collate_pad, num_workers=cfg["num_workers"], pin_memory=True)

    source_metrics = {}

    for src, src_chains in test_chains_by_source.items():
        test_ds = ConvoStyleDataset.from_prebuilt_chains(
            chains=src_chains,
            h5_path=cfg["h5_path"],
            meta_columns=["transcription", "text_description"],
            sample_rate=cfg["sample_rate"],
            max_len_sec=cfg["max_len_sec"],
        )
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kw)
        all_preds, all_refs, all_texts, all_vecs = [], [], [], []


        with torch.no_grad():
            for batch in test_loader:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    audio       = batch["audio"].to(device)
                    lengths     = batch["lengths"].to(device)
                    text_only   = batch["text_only"].to(device)
                    texts       = batch["transcription"]
                    speaker_ids = batch["speaker_id"]
                    targets     = batch["text_description"]
                    if cfg["num_turns"] == 0:
                        audio     = torch.zeros_like(audio)
                        text_only = torch.ones_like(text_only)
                    ctx = model.scfa(audio, lengths, texts, speaker_ids, text_only)
                    vec = model.pooler(ctx)
                    del ctx
                    all_vecs.append(vec.float().detach().cpu())  # collect before delete
                    preds = model.style_generator.generate(vec)
                    del vec
                
                all_preds.extend(preds)
                all_refs.extend([chain[-1] for chain in targets])
                all_texts.extend(texts)

        # free GPU memory
        del test_loader
        gc.collect()
        torch.cuda.empty_cache()

        # collapse diagnostics on pooled dialogue vectors
        vecs = torch.cat(all_vecs, dim=0)               # (N, 4*d_model)
        vec_std      = vecs.std(dim=0).mean().item()     # near 0 = collapsed
        vec_norm_cv  = (vecs.norm(dim=-1).std() /
                        vecs.norm(dim=-1).mean()).item()  # coeff of variation of norms

        # pairwise cosine similarity — mean off-diagonal → 1.0 = fully collapsed
        normed   = torch.nn.functional.normalize(vecs, dim=-1)
        sim_mat  = normed @ normed.T
        n        = sim_mat.shape[0]
        off_diag = sim_mat[~torch.eye(n, dtype=torch.bool)].mean().item()

        del all_vecs, vecs, normed, sim_mat

        
        bs   = compute_bertscore(all_preds, all_refs, device=str(device))
        met  = compute_meteor(all_preds, all_refs)
        chrf = compute_chrf(all_preds, all_refs)
        rou  = compute_rouge(all_preds, all_refs)
        tf1  = compute_tag_f1(all_preds, all_refs, src)

        # automate logging of lots of metrics
        source_metrics[src] = {
            **_flatten(bs),
            **_flatten(met),
            **_flatten(chrf),
            **_flatten(rou),
            **_flatten(tf1),
            "vec_std":         vec_std,
            "vec_norm_cv":     vec_norm_cv,
            "mean_cosine_sim": off_diag,
        }



        for i, (pred, ref, txt) in enumerate(zip(all_preds[:3], all_refs[:3], all_texts[:3])):
            log.info(f"  [Test/{src} Sample {i+1}]")
            log.info(f"    Dialogue : {txt}")
            log.info(f"    Predicted: {pred}")
            log.info(f"    Reference: {ref}")

    gc.collect()
    torch.cuda.empty_cache()

    return source_metrics


def assert_no_test_leakage(trainval_ids: set, test_conv_ids: set) -> None:
    """Raise if any test conv_id leaked into the train/val pool."""
    overlap = trainval_ids & test_conv_ids
    assert not overlap, (
        f"Data leakage: {len(overlap)} conv_id(s) appear in both train/val and test sets: "
        f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
    )
