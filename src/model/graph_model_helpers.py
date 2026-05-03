

from tqdm import tqdm

from capstone_src.style_prompt_generator.model.train_helpers import (
    _flatten,
    compute_bertscore, compute_meteor, compute_chrf
    , compute_rouge, compute_tag_f1, compute_dist, compute_pred_semantic_sim
    ,wandb_log
)

from capstone_src.style_prompt_generator.dataset.ConvoStyleDataset import (
    ConvoStyleDataset, collate_pad
)

from model.DialogueGraph import DialogueGraph
from model.GraphStylePromptGenerator import build_style_generator, GraphStylePrompt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Dict, Any

import gc
import time




def build_text_encoder(cfg: Dict[str, Any], device: torch.device, log):
    """Load BERT, freeze all layers, then optionally unfreeze the top N."""
    from transformers import AutoModel, AutoTokenizer

    BERT_REPO = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    model = AutoModel.from_pretrained(BERT_REPO).to(device)

    return model, tokenizer


def build_audio_encoder(cfg: Dict[str, Any], device: torch.device, log):
    """Load WavLM-base-plus, freeze all layers, then optionally unfreeze the top N."""
    from transformers import WavLMModel, AutoFeatureExtractor

    WAVLM_REPO = "microsoft/wavlm-base-plus"
    processor = AutoFeatureExtractor.from_pretrained(WAVLM_REPO)
    model = WavLMModel.from_pretrained(WAVLM_REPO).to(device)

    return model, processor


def build_model(cfg: Dict[str, Any], device: torch.device, log) -> GraphStylePrompt:
    log.info("Building model...")

    text_backbone, tokenizer = build_text_encoder(cfg, device, log)
    audio_backbone, processor = build_audio_encoder(cfg, device, log)

    dialogue_graph = DialogueGraph(
        bert_model=text_backbone,
        wavlm_model=audio_backbone,
        tokenizer=tokenizer,
        processor=processor,
        sample_rate=cfg["sample_rate"],
        d_feat=cfg["d_feat"],
        d_model=cfg["d_model"],
        d_out=cfg["d_out"],
        attn_dim=cfg["attn_dim"],
        num_gcn_layers=cfg.get("num_gcn_layers", 1),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    model = build_style_generator(
        dialogue_graph=dialogue_graph,
        d_out=cfg["d_out"],
        num_prefix_tokens=cfg["num_prefix_tokens"],
        num_mapping_layers=cfg["num_mapping_layers"],
        nhead=cfg["mapping_nhead"],
        max_new_tokens=cfg["max_new_tokens"],
        max_prompt_tokens=cfg["max_prompt_tokens"],
        system_prompt=cfg["system_prompt"],
        device=device,
        torch_dtype=torch.bfloat16,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable_params:,}")

    return model


def _grad_norm(model: nn.Module) -> float:
    """Compute global L2 norm of all gradients. Useful for diagnosing training stability."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5



def compute_loss(
    model: GraphStylePrompt,
    batch: Dict[str, Any],
    device: torch.device,
    cfg: Dict[str, Any],
    log
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

    dialogue_vec = model.dialogue_graph(audio, lengths, texts, speaker_ids, text_only)
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


def run_epoch(
    model, loader, optimizer, scheduler, 
    device, cfg, epoch, global_step, wandb_run, log_handler
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
                loss = compute_loss(model, batch, device, cfg, log=log_handler)
 
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

        # Warm-up: one forward pass to trigger CUDA kernel compilation and
        # GPU memory allocation before the timed region begins.
        warm_batch = next(iter(test_loader))
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _audio       = warm_batch["audio"].to(device)
                _lengths     = warm_batch["lengths"].to(device)
                _text_only   = warm_batch["text_only"].to(device)
                _texts       = warm_batch["transcription"]
                _speaker_ids = warm_batch["speaker_id"]
                if cfg["num_turns"] == 0:
                    _audio     = torch.zeros_like(_audio)
                    _text_only = torch.ones_like(_text_only)
                
                _ctx = model.dialogue_graph(_audio, _lengths, _texts, _speaker_ids, _text_only)
                model.style_generator.generate(_ctx)
                del _ctx

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0_infer = time.time()
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
                    
                    vec = model.dialogue_graph(audio, lengths, texts, speaker_ids, text_only)
                    all_vecs.append(vec.float().detach().cpu())

                    
                    preds = model.style_generator.generate(vec)
                    del vec

                
                all_preds.extend(preds)
                all_refs.extend([chain[-1] for chain in targets])
                all_texts.extend(texts)
        inference_time = time.time() - t0_infer


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
        div  = compute_dist(all_preds)
        psem = compute_pred_semantic_sim(all_preds, device=str(device))


        # automate logging of lots of metrics
        source_metrics[src] = {
            **_flatten(bs),
            **_flatten(met),
            **_flatten(chrf),
            **_flatten(rou),
            **_flatten(tf1),
            **div,
            **psem,
            "vec_std":           vec_std,
            "vec_norm_cv":       vec_norm_cv,
            "mean_cosine_sim":   off_diag,
            "inference_time_s":  inference_time,
        }





        for i, (pred, ref, txt) in enumerate(zip(all_preds[:3], all_refs[:3], all_texts[:3])):
            log.info(f"  [Test/{src} Sample {i+1}]")
            log.info(f"    Dialogue : {txt}")
            log.info(f"    Predicted: {pred}")
            log.info(f"    Reference: {ref}")

    gc.collect()
    torch.cuda.empty_cache()

    return source_metrics
