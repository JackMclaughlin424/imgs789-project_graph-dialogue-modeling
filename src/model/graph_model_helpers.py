

from capstone_src.style_prompt_generator.model.train_helpers import (
    _flatten,
    compute_bertscore, compute_meteor, compute_chrf
    , compute_rouge, compute_tag_f1, compute_dist, compute_pred_semantic_sim
)

from capstone_src.style_prompt_generator.dataset.ConvoStyleDataset import (
    ConvoStyleDataset, collate_pad
)

from model.DialogueGraph import DialogueGraph

import torch
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


def build_model(cfg: Dict[str, Any], device: torch.device, log) -> DialogueGraph:
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable_params:,}")

    return model

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
                _ctx = model.scfa(_audio, _lengths, _texts, _speaker_ids, _text_only)
                _vec = model.pooler(_ctx)
                del _ctx
                model.style_generator.generate(_vec)
                del _vec
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
                    ctx = model.scfa(audio, lengths, texts, speaker_ids, text_only)
                    vec = model.pooler(ctx)
                    del ctx
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
