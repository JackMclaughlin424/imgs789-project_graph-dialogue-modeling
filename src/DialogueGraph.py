"""
DialogueGraph: multimodal dialogue graph for conversational speech style modeling.

Based on: "Enhancing Speaking Styles in Conversational TTS with Graph-based
Multi-modal Context Modeling" (Li et al., 2022), Figure 2.

Key modifications from the paper:
- WavLM-base-plus with a learned per-layer weighted sum replaces the reference
  encoder + GST attention layer used in the original for style feature extraction.
- BERT-base with self-attentive pooling replaces BERT -> Pre-net -> CBHG.
- Output is a fixed-dimension (B, d_out) style vector intended for downstream
  projection into LLaMA embedding space (not implemented here).

Pipeline (matching Figure 2):
  1. Per-utterance feature extraction: text (BERT) + audio style (WavLM)
  2. Build complete directed graph over context turns with 5 relation types
  3. DialogueGCN (1 iteration per paper) updates each context node
  4. Attention pooling: current/anchor turn is the query over [g_i; h_i] context KV
  5. Linear projection -> (B, d_out)

The last turn in the input sequence is the anchor (current utterance); all
preceding turns form the context graph nodes.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import warnings


# Shared utilities


class _SelfAttentivePooling(nn.Module):
    # Collapses a variable-length hidden sequence to a single vector via learned
    # attention scores. Handles fully-masked rows (e.g. silent audio turns) by
    # zeroing out the output rather than producing NaN from softmax.

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Linear(dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.scorer(hidden).squeeze(-1)  # (B, T)
        if mask is not None:
            scores     = scores.masked_fill(~mask.bool(), float("-inf"))
            all_masked = ~mask.bool().any(dim=-1, keepdim=True)
            scores     = scores.masked_fill(all_masked, 0.0)
        weights = F.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights.masked_fill(all_masked, 0.0)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)


# Per-utterance feature extraction

class _UtteranceFeatureExtractor(nn.Module):
    """
    Extracts fixed-size per-utterance feature vectors for all turns in a batch.

    Text path: BERT-base last_hidden_state -> self-attentive pool -> linear(768, d_feat).
    Audio path: WavLM-base-plus with learned per-layer weighted sum (as in StyleCap sec 3.2)
                -> self-attentive pool -> linear(768, d_feat).
    Final output: cat(text_feat, audio_feat) -> (B, T, 2*d_feat).
    """

    # WavLM-base-plus has 1 CNN embedding layer + 12 transformer layers
    _NUM_WAVLM_LAYERS = 13

    def __init__(
        self,
        bert_model,
        wavlm_model,
        tokenizer,
        processor,
        sample_rate: int,
        d_feat:      int,
    ):
        super().__init__()
        self.bert        = bert_model
        self.wavlm       = wavlm_model
        self.tokenizer   = tokenizer
        self.processor   = processor
        self.sample_rate = sample_rate

        # learned scalar weight per WavLM layer; initialised to uniform (all zeros before softmax)
        self.wavlm_layer_weights = nn.Parameter(torch.zeros(self._NUM_WAVLM_LAYERS))

        self.text_pool  = _SelfAttentivePooling(768)
        self.audio_pool = _SelfAttentivePooling(768)

        self.text_proj  = nn.Linear(768, d_feat)
        self.audio_proj = nn.Linear(768, d_feat)

    def _encode_text_flat(self, flat_texts: List[str]) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        tokens = self.tokenizer(
            flat_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")
            out = self.bert(**tokens)
        return self.text_pool(out.last_hidden_state, tokens["attention_mask"])  # (M, 768)

    def _encode_audio_turn(
        self, audio_t: torch.Tensor, lengths_t: torch.Tensor
    ) -> torch.Tensor:
        # audio_t: (B, samples), lengths_t: (B,) raw sample counts
        device = audio_t.device
        inputs = self.processor(
            list(audio_t.cpu().numpy()),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")
            wavlm_out = self.wavlm(**inputs, output_hidden_states=True)

        # weighted sum over all WavLM layers (StyleCap eq., same approach as DialogueEncoder)
        hidden_states = torch.stack(wavlm_out.hidden_states, dim=0)       # (13, B, T_wav, 768)
        layer_weights = F.softmax(self.wavlm_layer_weights, dim=0)        # (13,)
        hidden        = (layer_weights[:, None, None, None] * hidden_states).sum(dim=0)  # (B, T_wav, 768)

        # downsample the raw-sample mask to match WavLM's subsampled output length
        T_out    = hidden.shape[1]
        raw_mask = (
            torch.arange(audio_t.shape[1], device=device).unsqueeze(0)
            < lengths_t.to(device).unsqueeze(1)
        ).float()
        mask_ds  = F.interpolate(raw_mask.unsqueeze(1), size=T_out, mode="nearest").squeeze(1).bool()

        return self.audio_pool(hidden, mask_ds)  # (B, 768)

    def forward(
        self,
        audio:     torch.Tensor,                    # (B, T, samples)
        lengths:   torch.Tensor,                    # (B, T)
        texts:     List[List[str]],                 # [B][T]
        text_only: Optional[torch.Tensor] = None,   # (B, T) bool
    ) -> torch.Tensor:
        B, T, _ = audio.shape
        d_feat   = self.text_proj.out_features
        device   = audio.device

        # flatten all (batch, turn) pairs into a single tokeniser call for efficiency
        flat_texts = [texts[b][t] for b in range(B) for t in range(T)]
        text_flat  = self._encode_text_flat(flat_texts)              # (B*T, 768)
        text_emb   = self.text_proj(text_flat).view(B, T, d_feat)   # (B, T, d_feat)

        audio_emb = torch.zeros(B, T, d_feat, dtype=audio.dtype, device=device)
        for t in range(T):
            if text_only is not None and text_only[:, t].all():
                continue
            has_audio = (
                (~text_only[:, t].bool()) if text_only is not None
                else torch.ones(B, dtype=torch.bool, device=device)
            )
            raw_emb = self._encode_audio_turn(audio[:, t, :], lengths[:, t])  # (B, 768)
            audio_emb[has_audio, t] = self.audio_proj(raw_emb[has_audio])

        return torch.cat([text_emb, audio_emb], dim=-1)  # (B, T, 2*d_feat)


