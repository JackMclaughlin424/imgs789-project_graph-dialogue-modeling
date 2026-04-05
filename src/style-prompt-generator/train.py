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

from StylePromptGenerator import (
    SCFAWithStyleHead
)

import sys

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.fx")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._inductor")



from train_helpers import *

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

    # num_turns == 0: treat as text-only by zeroing audio
    if cfg["num_turns"] == 0:
        audio = torch.zeros_like(audio)
        text_only = torch.ones_like(text_only)


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
    device, cfg, epoch, global_step, wandb_run, is_train=True,
) -> tuple[float, int]:
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0
    tag = "TRAIN" if is_train else "VAL"
 
    ctx = torch.enable_grad if is_train else torch.no_grad
 
    with ctx():
        pbar = tqdm(loader, desc=f"{tag} epoch {epoch}", unit="batch", leave=True, dynamic_ncols=True)

        for batch in pbar:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = compute_loss(model, batch, device, cfg)
 
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if cfg["grad_clip"]:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg["grad_clip"],
                    )
                optimizer.step()
 
                scheduler.step()
                global_step += 1
 
                if global_step % cfg["log_every_n_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    grad_norm = _grad_norm(model)
                    run = f"[{cfg['run_name']}] " if cfg["run_name"] else ""
                    log.info(
                        f"{run}epoch {epoch}  step {global_step}  "
                        f"loss {loss.item():.4f}  lr {lr:.2e}  grad_norm {grad_norm:.3f}"
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
    log.info(f"{tag} epoch {epoch} avg loss: {avg:.4f}")
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

    # save the resolved config for reproducibility
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
 
    train_loader, val_loader, dataset = build_dataloaders(cfg, log)
    
    model = build_model(cfg, device, log)

    # compile model for faster training
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
 
    # W&B can watch gradients and parameter histograms if you want them
    # log_freq controls how often histograms are computed -- expensive so keep it low
    if wandb_run is not None:
        import wandb
        wandb_run.watch(model, log="gradients", log_freq=cfg["log_every_n_steps"] * 5)

    start_epoch = 0
    global_step = 0
 
    # resume if a checkpoint exists in the output dir
    existing = sorted(out_dir.glob("ckpt_epoch*.pt"))
    if resume and existing:
        start_epoch, global_step = load_checkpoint(
            str(existing[-1]), log, model, optimizer, scheduler
        )
        start_epoch += 1  # resume from the next epoch
 
    best_val_loss = float("inf")
    patience_counter = 0

 
    for epoch in tqdm(
        range(start_epoch, cfg["num_epochs"]), desc="Epochs"
        , unit="epoch", initial=start_epoch, total=cfg["num_epochs"]
    ):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, 
            device, cfg, epoch, global_step, wandb_run, is_train=True,
        )
 
        # epoch-level train loss (separate key from step-level so the chart is clean)
        epoch_metrics = {"epoch/train_loss": train_loss, "epoch": epoch}

        if (epoch + 1) % cfg["eval_every_n_epochs"] == 0:
            val_loss, _ = run_epoch(
                model, val_loader, optimizer, scheduler, 
                device, cfg, epoch, global_step, wandb_run, is_train=False,
            )
            epoch_metrics["epoch/val_loss"] = val_loss

            min_delta = cfg["early_stopping_min_delta"]
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                epoch_metrics["epoch/best_val_loss"] = best_val_loss
            elif cfg["early_stopping_patience"] > 0:
                patience_counter += 1
                log.info(f"No improvement for {patience_counter}/{cfg['early_stopping_patience']} epochs.")
                if patience_counter >= cfg["early_stopping_patience"]:
                    log.info("Early stopping triggered.")
                    wandb_log(epoch_metrics, step=global_step, run=wandb_run)
                    break




        wandb_log(epoch_metrics, step=global_step, run=wandb_run)

        if (epoch + 1) % cfg["save_every_n_epochs"] == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, train_loss, cfg, out_dir, log
            )
            log.info(f"Keeping last {cfg['keep_last_n_ckpts']} checkpoints, pruning older ones.")
            prune_old_checkpoints(out_dir, cfg["keep_last_n_ckpts"], log)
 
            # log the checkpoint as a W&B artifact so you can restore any saved version
            if wandb_run is not None:
                import wandb
                artifact = wandb.Artifact(
                    name=f"checkpoint-{wandb_run.id}",
                    type="model",
                    metadata={"epoch": epoch, "step": global_step, "val_loss": best_val_loss if best_val_loss != float("inf") else None},

                )
                artifact.add_file(str(ckpt_path))
                wandb_run.log_artifact(artifact)
 
        log.info("Training complete.")

    # compute text-quality metrics once on the final model state 
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

            dialogue_ctx = model.scfa(audio, lengths, texts, speaker_ids, text_only)
            dialogue_vec = model.pooler(dialogue_ctx)
            preds = model.style_generator.generate(dialogue_vec)

            all_preds.extend(preds)
            all_refs.extend([chain[-1] for chain in targets])

    bs_metrics  = compute_bertscore(all_preds, all_refs, device=str(device))
    met_metrics = compute_meteor(all_preds, all_refs)
    log.info(
        f"Final val metrics — BERTScore F1: {bs_metrics['bertscore_f1_mean']:.4f} "
        f"(±{bs_metrics['bertscore_f1_std']:.4f})  "
        f"METEOR: {met_metrics['meteor_mean']:.4f} "
        f"(±{met_metrics['meteor_std']:.4f})"
    )
    wandb_log({
        f"final/{k}": v for k, v in {**bs_metrics, **met_metrics}.items()
    }, step=global_step, run=wandb_run)

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

