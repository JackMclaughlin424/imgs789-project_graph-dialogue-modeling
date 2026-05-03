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
        device = next(iter(self.bert.parameters()), None)
        if device is None:
            raise RuntimeError("bert_model has no parameters — was it initialized correctly?")
        device = device.device

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



# DialogueGCN

class _DialogueGCNLayer(nn.Module):
    """
    Single relation-aware GCN layer with 5 relation types.

    Relation taxonomy follows the complete directed graph construction in the paper:
    each utterance pair gets an intra- or inter-speaker edge in each temporal direction,
    and every node has a self-loop (the paper marks self-loops as future->past).

    Update rule (per relation r, then summed):
        h_v^new += W_r * mean_{u: u->v in r}(h_u)
    Final: h_v^new = ReLU(sum_r contribution_r); LayerNorm(h_new + h)
    """

    _SELF         = 0
    _INTRA_P2F    = 1   # same speaker, earlier turn -> later turn
    _INTRA_F2P    = 2   # same speaker, later turn -> earlier turn
    _INTER_P2F    = 3   # different speaker, earlier turn -> later turn
    _INTER_F2P    = 4   # different speaker, later turn -> earlier turn
    NUM_RELATIONS = 5

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # independent linear map per relation type; no bias follows standard R-GCN
        self.weights = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(self.NUM_RELATIONS)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def build_adjacency(
        speaker_ids: List[List[str]], N: int, device: torch.device
    ) -> torch.Tensor:
        """
        Builds (B, NUM_RELATIONS, N, N) binary adjacency matrices.
        adj[b, r, v, u] = 1 means node u sends a message to node v under relation r.
        """
        B   = len(speaker_ids)
        R   = _DialogueGCNLayer.NUM_RELATIONS
        adj = torch.zeros(B, R, N, N, device=device)
        for b in range(B):
            spk = speaker_ids[b]
            for v in range(N):
                adj[b, _DialogueGCNLayer._SELF, v, v] = 1.0
                for u in range(N):
                    if u == v:
                        continue
                    same = spk[u] == spk[v]
                    if u < v:
                        # u is temporally before v: edge u->v is past-to-future
                        r = _DialogueGCNLayer._INTRA_P2F if same else _DialogueGCNLayer._INTER_P2F
                    else:
                        # u is temporally after v: edge u->v is future-to-past
                        r = _DialogueGCNLayer._INTRA_F2P if same else _DialogueGCNLayer._INTER_F2P
                    adj[b, r, v, u] = 1.0
        return adj

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h:   (B, N, d_model)
        # adj: (B, NUM_RELATIONS, N, N)
        agg = torch.zeros_like(h)
        for r in range(self.NUM_RELATIONS):
            A   = adj[:, r, :, :]                          # (B, N, N)
            # mean-normalise by in-degree so nodes with many neighbours don't dominate
            deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
            msg = (A / deg) @ h                             # (B, N, d_model)
            agg = agg + self.weights[r](msg)
        out = self.drop(F.relu(agg))
        return self.norm(out + h)                           # residual + LN


class _DialogueGCN(nn.Module):
    # Stack of relation-aware GCN layers; the paper uses 1 iteration.

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DialogueGCNLayer(d_model, dropout) for _ in range(num_layers)]
        )

    def forward(self, h: torch.Tensor, speaker_ids: List[List[str]]) -> torch.Tensor:
        # h: (B, N, d_model)
        _, N, _ = h.shape
        adj = _DialogueGCNLayer.build_adjacency(speaker_ids, N, h.device)
        for layer in self.layers:
            h = layer(h, adj)
        return h  # (B, N, d_model)


# Context attention pooling

class _ContextAttentionPooling(nn.Module):
    # Bahdanau-style additive attention where the current (anchor) turn is the
    # query and each context node's [GCN-output; original-embedding] pair is the KV.
    # Returns both the context vector and the attention weights for inspection.

    def __init__(self, query_dim: int, kv_dim: int, attn_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim,    attn_dim, bias=False)
        self.score  = nn.Linear(attn_dim,  1,        bias=False)

    def forward(
        self,
        query: torch.Tensor,                   # (B, query_dim)
        kv:    torch.Tensor,                   # (B, N, kv_dim)
        mask:  Optional[torch.Tensor] = None,  # (B, N) bool, True = valid
    ):
        q      = self.q_proj(query).unsqueeze(1)            # (B, 1, attn_dim)
        k      = self.k_proj(kv)                            # (B, N, attn_dim)
        scores = self.score(torch.tanh(q + k)).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)                 # (B, N)
        context = (weights.unsqueeze(-1) * kv).sum(dim=1)  # (B, kv_dim)
        return context, weights


# Top-level module

class DialogueGraph(nn.Module):
    """
    Full DialogueGraph pipeline (Figure 2).

    Constructor args
    ----------------
    bert_model, wavlm_model, tokenizer, processor, sample_rate
        Pre-loaded HuggingFace model and processor objects; frozen during
        feature extraction (no_grad used inside _UtteranceFeatureExtractor).
    d_feat : int
        Projection dimension for each modality (text and audio) after pooling.
        The extractor concatenates both, so node hidden size entering the GCN
        is 2 * d_feat after input_proj maps it to d_model.
    d_model : int
        Internal dimension for the GCN and attention layers.
    d_out : int
        Dimension of the returned style vector.
    attn_dim : int
        Hidden dimension inside the Bahdanau scoring MLP.
    num_gcn_layers : int
        Number of stacked _DialogueGCNLayer passes (paper uses 1).
    dropout : float
        Dropout applied inside each GCN layer.
    """

    def __init__(
        self,
        bert_model,
        wavlm_model,
        tokenizer,
        processor,
        sample_rate:    int,
        d_feat:         int,
        d_model:        int,
        d_out:          int,
        attn_dim:       int,
        num_gcn_layers: int   = 1,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.extractor    = _UtteranceFeatureExtractor(
            bert_model, wavlm_model, tokenizer, processor, sample_rate, d_feat
        )
        # project concatenated (text || audio) features to GCN working dimension
        self.input_proj   = nn.Linear(2 * d_feat, d_model)
        self.gcn          = _DialogueGCN(d_model, num_gcn_layers, dropout)
        # KV fed to attention is [g_i; h_i], so kv_dim = 2 * d_model
        self.context_attn = _ContextAttentionPooling(d_model, 2 * d_model, attn_dim)
        # final cat is [current_h; context_vec] -> d_model + 2*d_model
        self.out_proj     = nn.Linear(3 * d_model, d_out)

    def forward(
        self,
        audio:        torch.Tensor,                    # (B, T, samples)
        lengths:      torch.Tensor,                    # (B, T) raw sample counts
        texts:        List[List[str]],                 # [B][T]
        speaker_ids:  List[List[str]],                 # [B][T], last entry is anchor
        text_only:    Optional[torch.Tensor] = None,   # (B, T) bool
        context_mask: Optional[torch.Tensor] = None,   # (B, T-1) bool, True = real turn
    ) -> torch.Tensor:
        # --- feature extraction ---
        feats = self.extractor(audio, lengths, texts, text_only)  # (B, T, 2*d_feat)
        h     = self.input_proj(feats)                            # (B, T, d_model)

        # last turn is the anchor; all prior turns form the context graph
        context_h = h[:, :-1, :]   # (B, N, d_model),  N = T - 1
        current_h = h[:, -1,  :]   # (B, d_model)

        if context_h.shape[1] == 0:
            # degenerate case: no context, return projection of current turn only
            zeros = torch.zeros(
                current_h.shape[0], 2 * self.context_attn.k_proj.in_features,
                device=current_h.device, dtype=current_h.dtype,
            )
            return self.out_proj(torch.cat([current_h, zeros], dim=-1))

        # --- DialogueGCN over context nodes ---
        context_speaker_ids = [s[:-1] for s in speaker_ids]  # drop anchor from each item
        g = self.gcn(context_h, context_speaker_ids)          # (B, N, d_model)

        # form key-value pairs as described in paper: concatenate GCN output with
        # the original pre-GCN embedding so the attention can use both representations
        kv = torch.cat([g, context_h], dim=-1)  # (B, N, 2*d_model)

        # --- attention pooling ---
        context_vec, _ = self.context_attn(current_h, kv, context_mask)  # (B, 2*d_model)

        # --- final projection ---
        return self.out_proj(torch.cat([current_h, context_vec], dim=-1))  # (B, d_out)

