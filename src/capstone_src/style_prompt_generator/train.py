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
            is_train=True, use_tqdm=True,
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
    parser = argparse.ArgumentParser(description="Train SCFAWithStyleHead from a JSON config.")
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


