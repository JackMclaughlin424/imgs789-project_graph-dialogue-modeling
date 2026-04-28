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

from model.StylePromptGenerator import (
    SCFAWithStyleHead
)

import sys

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.fx")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._inductor")



from model.train_helpers import *

logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

# suppress huggingface and other builtin messages
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# Loss

def compute_loss(
    model: SCFAWithStyleHead,
    batch: Dict[str, Any],
    device: torch.device,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Teacher-forced cross-entropy over the style prompt tokens.

    The LLM is frozen, so we only flow gradients through the mapping network
    (StylePromptHead), SCFA model, and optionally the unfrozen layers of BERT and WavLM.
    """
    audio      = batch["audio"].to(device)        # (B, T, samples)
    lengths    = batch["lengths"].to(device)      # (B, T)
    text_only  = batch["text_only"].to(device)    # (B, T)
    texts      = batch["transcription"]           # list[B][T]
    speaker_ids = batch["speaker_id"]             # list[B][T]
    targets    = batch["text_description"]            # list[B][T] -- we want the anchor turn's prompt

    # anchor is always the last turn
    anchor_prompts = [chain[-1] for chain in targets]  # list[B]

    # anchor (last turn) is always text-only: predicting style for audio not yet recorded
    audio[:, -1, :] = 0
    lengths[:, -1]  = 0
    text_only[:, -1] = True


    #
    #   RUNNING MODEL
    #

    # Encode turns with context + speaker information
    dialogue_ctx = model.scfa(audio, lengths, texts, speaker_ids, text_only)
    # pool embeddings
    dialogue_vec = model.pooler(dialogue_ctx)
    # get prefix embeddings from the style head
    prefix_embeds = model.style_generator.style_head(dialogue_vec)  # (B, K, TINYLLAMA_DIM)

    if torch.isnan(prefix_embeds).any():
        log.warning(f"NaN in prefix_embeds! dialogue_vec nan={torch.isnan(dialogue_vec).any()}")
        return torch.tensor(float('nan'), device=device, requires_grad=True)

    B = prefix_embeds.shape[0]
    K = prefix_embeds.shape[1]

    # tokenize the target style descriptions
    tokenizer = model.style_generator.tokenizer
    llm       = model.style_generator.llm

    target_tokens = tokenizer(
        anchor_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg["max_style_desc_tokens"],
    )
    input_ids = target_tokens.input_ids.to(device)   # (B, L)
    attn_mask = target_tokens.attention_mask.to(device)

    # embed target tokens for teacher forcing
    token_embeds = llm.get_input_embeddings()(input_ids)  # (B, L, TINYLLAMA_DIM)

    # optionally prepend system prompt, skip if empty string
    if model.style_generator.system_prompt:
        prompt_tokens = tokenizer(
            [model.style_generator.system_prompt] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["max_style_desc_tokens"],
        )
        prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids.to(device))

        input_embeds = torch.cat([prefix_embeds, prompt_embeds, token_embeds], dim=1)
        prefix_mask  = torch.ones(B, K + prompt_embeds.shape[1], dtype=torch.long, device=device)
        full_mask    = torch.cat([prefix_mask, attn_mask], dim=1)
    else:
        input_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask  = torch.ones(B, K, dtype=torch.long, device=device)
        full_mask    = torch.cat([prefix_mask, attn_mask], dim=1)

    # build labels: -100 for prefix positions (not supervised), token ids elsewhere
    prefix_labels = torch.full((B, input_embeds.shape[1] - input_ids.shape[1]), -100, device=device)
    token_labels  = input_ids.masked_fill(attn_mask == 0, -100)
    labels        = torch.cat([prefix_labels, token_labels], dim=1)

    valid_label_count = (labels != -100).sum()
    if valid_label_count == 0:
        log.warning(f"All labels are -100! anchor_prompts sample: {anchor_prompts[0]!r}")

    # cast input embeddings to TinyLlama dtype bf16
    input_embeds = input_embeds.to(llm.dtype)

    if torch.isnan(input_embeds).any() or torch.isinf(input_embeds).any():
        log.warning(f"NaN/Inf in input_embeds after dtype cast. Max abs: {input_embeds.abs().max().item():.2e}")

    outputs = llm(
        inputs_embeds=input_embeds,
        attention_mask=full_mask,
        labels=labels,
    )

    return outputs.loss





# Training loop

def _grad_norm(model: nn.Module) -> float:
    """Compute global L2 norm of all gradients. Useful for diagnosing training stability."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def run_epoch(
    model, loader, optimizer, scheduler, 
    device, cfg, epoch, global_step, wandb_run, log_handler=log
    , is_train=True, use_tqdm=True
) -> tuple[float, int]:
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0
    tag = "TRAIN" if is_train else "VAL"

    n_total = len(loader)
 
    ctx = torch.enable_grad if is_train else torch.no_grad
    epoch_start = time.time()

    with ctx():
        iterable = tqdm(loader, desc=f"{tag} epoch {epoch}", unit="batch", leave=True, dynamic_ncols=True) if use_tqdm else loader

        for batch in iterable:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = compute_loss(model, batch, device, cfg)
 
            if is_train:
                optimizer.zero_grad()
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    if cfg["grad_clip"]:
                        nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            cfg["grad_clip"],
                        )
                    optimizer.step()
                else:
                    log_handler.warning(f"Skipping optimizer step due to NaN/Inf loss at step {global_step}")
                scheduler.step()
                global_step += 1

 
                if global_step % cfg["log_every_n_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    grad_norm = _grad_norm(model)
                    run = f"[{cfg['run_name']}] " if cfg["run_name"] else ""

                    eta_str = ""
                    if not use_tqdm:
                        batches_done = n_batches + 1
                        elapsed = time.time() - epoch_start
                        secs_per_batch = elapsed / batches_done
                        remaining = (n_total - batches_done) * secs_per_batch
                        fmt = lambda s: f"{int(s)//60:02d}:{int(s)%60:02d}"
                        eta_str = f"  {fmt(elapsed)}<{fmt(remaining)}"

                    log_handler.info(
                        f"{run}epoch {epoch} - batch {n_batches + 1}/{n_total} - step {global_step} | "
                        f"loss {loss.item():.4f}  lr {lr:.2e}  grad_norm {grad_norm:.3f}{eta_str}"
                    )
                    # step-level metrics -- logged at every log_every_n_steps
                    wandb_log({
                        "train/loss":      loss.item(),
                        "train/lr":        lr,
                        "train/grad_norm": grad_norm,
                    }, step=global_step, run=wandb_run)
 
            total_loss += loss.item()
            n_batches  += 1

            
 
    avg = total_loss / max(n_batches, 1)
    log_handler.info(f"{tag} epoch {epoch} avg loss: {avg:.4f}")
    return avg, global_step


# tqdm-safe console handler
class TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        tqdm.write(self.format(record))


def train(cfg: Dict[str, Any], resume=True):
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    wandb_run = wandb_init(cfg, log)

    console_handler = TqdmHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(console_handler)

    file_handler = logging.FileHandler(out_dir / "train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    log.info("File logging initialized.")

    test_chains_by_source, test_conv_ids = ConvoStyleDataset.make_fixed_test_split(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        max_len_sec=cfg["max_len_sec"],
        num_turns=cfg["num_turns"],
    )

    meta         = pd.read_parquet(cfg["meta_path"], columns=["conv_id"])
    trainval_ids = set(c for c in meta["conv_id"].unique() if c not in test_conv_ids)
    assert_no_test_leakage(trainval_ids, test_conv_ids)

    ds_kwargs = dict(
        h5_path=cfg["h5_path"],
        meta_path=cfg["meta_path"],
        meta_columns=["transcription", "text_description", "source"],
        sample_rate=cfg["sample_rate"],
        num_turns=cfg["num_turns"],
        max_len_sec=cfg["max_len_sec"],
    )
    loader_kw    = dict(collate_fn=collate_pad, num_workers=cfg["num_workers"], pin_memory=True)
    g            = torch.Generator().manual_seed(cfg["seed"])
    train_ds     = ConvoStyleDataset(**ds_kwargs, allowed_conv_ids=trainval_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=g, **loader_kw)
    log.info(f"Chains: {len(train_ds)} train  |  {sum(len(v) for v in test_chains_by_source.values())} test")

    model = build_model(cfg, device, log)

    if cfg["compile"]:
        model.scfa.ctx_audio = torch.compile(model.scfa.ctx_audio)
        model.scfa.ctx_text  = torch.compile(model.scfa.ctx_text)
        model.scfa.cfa       = torch.compile(model.scfa.cfa)
        model.scfa.ffn_audio = torch.compile(model.scfa.ffn_audio)
        model.scfa.ffn_text  = torch.compile(model.scfa.ffn_text)
        # spk_audio/spk_text take string speaker_ids so skip for now
        model.style_generator.style_head = torch.compile(model.style_generator.style_head)

    total_steps = len(train_loader) * cfg["num_epochs"]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps, log)

    if wandb_run is not None:
        import wandb
        wandb_run.watch(model, log="gradients", log_freq=cfg["log_every_n_steps"] * 5)

    start_epoch = 0
    global_step = 0

    existing = sorted(out_dir.glob("ckpt_epoch*.pt"))
    if resume and existing:
        start_epoch, global_step = load_checkpoint(
            str(existing[-1]), log, model, optimizer, scheduler
        )
        start_epoch += 1

    for epoch in tqdm(
        range(start_epoch, cfg["num_epochs"]), desc="Epochs",
        unit="epoch", initial=start_epoch, total=cfg["num_epochs"]
    ):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg, epoch, global_step, wandb_run, is_train=True,
        )
        wandb_log({"epoch/train_loss": train_loss, "epoch": epoch}, step=global_step, run=wandb_run)

        if (epoch + 1) % cfg["save_every_n_epochs"] == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, train_loss, cfg, out_dir, log
            )
            log.info(f"Keeping last {cfg['keep_last_n_ckpts']} checkpoints, pruning older ones.")
            prune_old_checkpoints(out_dir, cfg["keep_last_n_ckpts"], log, wandb_run=wandb_run)

            if wandb_run is not None:
                import wandb
                artifact = wandb.Artifact(
                    name=f"checkpoint-{wandb_run.id}",
                    type="model",
                    metadata={"epoch": epoch, "step": global_step},
                )
                artifact.add_file(str(ckpt_path))
                wandb_run.log_artifact(artifact)

    log.info("Training complete.")

    log.info("Evaluating generation...")
    model.eval()
    test_metrics = eval_test_by_source(model, cfg, test_chains_by_source, device, log)

    for src, src_m in test_metrics.items():
        log.info(
            f"Test/{src}  bertscore_f1={src_m['bertscore_f1']:.4f}  "
            f"meteor={src_m['meteor']:.4f}  chrf={src_m['chrf']:.4f}  "
            f"tag_f1={src_m['tag_f1_overall']:.4f}"
        )
        wandb_log({f"test/{src}/{k}": v for k, v in src_m.items()}, step=global_step, run=wandb_run)

    del train_loader
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    wandb_finish(wandb_run)




# Entry point

def parse_args():
    p = argparse.ArgumentParser(description="Train SCFAWithStyleHead from a JSON config.")
    p.add_argument("--config", required=True, help="Path to hyperparameter config JSON")
    # allow inline overrides: --override learning_rate=1e-4 batch_size=16
    p.add_argument(
        "--override", nargs="*", metavar="KEY=VALUE",
        help="Override individual config fields (e.g. --override learning_rate=1e-4)"
    )

    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and train from scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    train(cfg, resume=not args.no_resume)

