import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional

import warnings

class SelfAttentivePooling(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Linear(dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.scorer(hidden).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
            # if an entire row is masked (e.g. zero-length audio, or inter-speaker at turn 0)
            # softmax would return NaN; fall back to uniform attention over all positions
            all_masked = ~mask.bool().any(dim=-1, keepdim=True)  # (B, 1)
            scores = scores.masked_fill(all_masked, 0.0)

        weights = F.softmax(scores, dim=-1)

        if mask is not None:
            weights = weights.masked_fill(all_masked, 0.0)  # zero-out fully-masked rows

        return (weights.unsqueeze(-1) * hidden).sum(dim=1)
    
class ModalityEncoder(nn.Module):

    def __init__(self, backbone, pooler):
        super().__init__()
        self.backbone = backbone
        self.pooler   = pooler

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.pooler(hidden, mask)  # (B, d)
    

def encode_text_inputs(texts, tokenizer):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

def encode_audio_inputs(audio, lengths, processor, sample_rate):
    inputs = processor(
        list(audio.cpu().numpy()),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    return inputs, lengths


class TurnPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_turns: int = 64):
        super().__init__()
        # learned is better than sinusoidal here -- turn sequences are short
        # and the model can learn dialogue-specific positional structure
        self.embedding = nn.Embedding(max_turns, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        return x + self.embedding(positions)  # broadcasts over B

@torch._dynamo.disable
class DualModalityEmbedder(nn.Module):

    def __init__(self, text_encoder_model_pretrained, audio_encoder_model_pretrained,
                 tokenizer, processor, SAMPLE_RATE):
        super().__init__()

        text_encoder  = ModalityEncoder(text_encoder_model_pretrained,  SelfAttentivePooling(768))
        audio_encoder = ModalityEncoder(audio_encoder_model_pretrained, SelfAttentivePooling(768))

        self.text_encoder  = text_encoder
        self.audio_encoder = audio_encoder
        self.tokenizer     = tokenizer
        self.processor     = processor
        self.SAMPLE_RATE   = SAMPLE_RATE

        # learned per-layer weights for WavLM SSL feature aggregation (StyleCap sec 3.2)
        # WavLM-base-plus has 1 CNN embedding layer + 12 transformer layers = 13 total
        num_wavlm_layers = 13
        self.wavlm_layer_weights = nn.Parameter(torch.zeros(num_wavlm_layers))

    def embed_text(self, texts):
        # texts is list[B][T] -- flatten to a single batch so the encoder sees B*T items
        B = len(texts)
        T = len(texts[0])
        flat_texts = [texts[b][t] for b in range(B) for t in range(T)]

        tokens = encode_text_inputs(flat_texts, self.tokenizer)

        # grab device from the model weights, not from input (there is no tensor input here)
        device = next(self.text_encoder.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}  # tokenizer always returns CPU tensors
    

        with torch.no_grad():
            warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")
            hidden = self.text_encoder.backbone(**tokens).last_hidden_state
        flat_emb = self.text_encoder(hidden, tokens["attention_mask"])  # (B*T, d)

        return flat_emb.view(B, T, -1)  # (B, T, d)

    def embed_audio(self, audio, lengths, text_only=None):
        # audio: (B, T, samples), lengths: (B, T)
        B, T, _ = audio.shape
        d_model  = 768

        device   = audio.device

        out = torch.zeros(B, T, d_model, dtype=audio.dtype, device=device)

        for t in range(T):
            # skip turns where every item in the batch has no audio
            if text_only is not None and text_only[:, t].all():
                continue

             # which items in this batch actually have audio for turn t
            has_audio = (~text_only[:, t].bool()) if text_only is not None else torch.ones(B, dtype=torch.bool, device=device)

            audio_t   = audio[:, t, :]    # (B, samples)
            lengths_t = lengths[:, t]     # (B,)

            inputs, lengths_t = encode_audio_inputs(audio_t, lengths_t, self.processor, self.SAMPLE_RATE)

            # processor outputs land on CPU same as tokenizer
            inputs = {k: v.to(device) for k, v in inputs.items()}  

            with torch.no_grad():
                warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")
                wavlm_out = self.audio_encoder.backbone(**inputs, output_hidden_states=True)

            # weighted sum over all WavLM layers (eq. from StyleCap)
            # hidden_states: tuple of (num_layers, B, T, 768)
            hidden_states = torch.stack(wavlm_out.hidden_states, dim=0)  # (13, B, T, 768)
            layer_weights = F.softmax(self.wavlm_layer_weights, dim=0)   # (13,)
            hidden = (layer_weights[:, None, None, None] * hidden_states).sum(dim=0)  # (B, T, 768)

            T_out    = hidden.shape[1]
            raw_mask = (torch.arange(audio_t.shape[1], device = device).unsqueeze(0)
                        < lengths_t.unsqueeze(1).to(device = device)).float()
            mask_ds  = F.interpolate(raw_mask.unsqueeze(1), size=T_out, mode="nearest").squeeze(1).bool()

            emb = self.audio_encoder(hidden, mask_ds)  # (B, d)
            # text-only items keep their zero-initialized embedding; writing NaN for them
            # (from the all-zero mask in SelfAttentivePooling) would corrupt the whole batch
            out[has_audio, t, :] = emb[has_audio]

        return out  # (B, T, d)

    @torch._dynamo.disable
    def forward(self, audio, lengths, texts, text_only=None):
        text_emb  = self.embed_text(texts)
        audio_emb = self.embed_audio(audio, lengths, text_only)
        return text_emb, audio_emb  # both (B, T, d)


class IntraSpeakerTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int
                 , dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)


    def forward(self, turn_embeddings: torch.Tensor, speaker_ids_list: List[List[str]]):
        B, seq_len, _ = turn_embeddings.shape
        device = turn_embeddings.device

        # Generate causal mask and ensure it has the same dtype as turn_embeddings
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device).to(turn_embeddings.dtype)

        # Create numerical speaker IDs for the current batch
        numerical_speaker_ids = []
        global_speaker_id_map = {}
        next_id = 0

        for b_idx in range(B):
            chain_speakers = speaker_ids_list[b_idx]
            chain_numerical_ids = []
            for speaker_str in chain_speakers:
                if speaker_str not in global_speaker_id_map:
                    global_speaker_id_map[speaker_str] = next_id
                    next_id += 1
                chain_numerical_ids.append(global_speaker_id_map[speaker_str])
            numerical_speaker_ids.append(chain_numerical_ids)

        # Convert to tensor (B, seq_len)
        speaker_ids_tensor = torch.tensor(numerical_speaker_ids, dtype=torch.long, device=device)

        # Create same speaker boolean mask (B, seq_len, seq_len)
        same_speaker_bool_mask = (speaker_ids_tensor.unsqueeze(2) == speaker_ids_tensor.unsqueeze(1))

        # Intra-speaker mask: allow attention within the same speaker, block attention to different speakers
        # Additive mask: 0 for allowed, -inf for blocked
        inf_val = torch.finfo(turn_embeddings.dtype).min
        # Explicitly cast 0. to the correct dtype for torch.where
        zero_tensor = torch.tensor(0., dtype=turn_embeddings.dtype, device=device)
        intra_speaker_additive_mask = torch.where(same_speaker_bool_mask, zero_tensor, inf_val)

        # Combine causal mask with intra-speaker mask
        # Causal mask is (S, S), intra_speaker_additive_mask is (B, S, S)
        # Unsqueezing causal_mask to (1, S, S) allows broadcasting
        final_mask = causal_mask.unsqueeze(0) + intra_speaker_additive_mask

        # Reshape the mask to (B * nhead, S, S)
        nhead_local = self.transformer_encoder.layers[0].self_attn.num_heads
        final_mask = final_mask.unsqueeze(1).expand(-1, nhead_local, -1, -1).reshape(B * nhead_local, seq_len, seq_len)

        output = self.transformer_encoder(turn_embeddings, mask=final_mask, is_causal=False)

        return output



class InterSpeakerTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int
                 , dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)


    def forward(self, turn_embeddings: torch.Tensor, speaker_ids_list: List[List[str]]):
        B, seq_len, _ = turn_embeddings.shape
        device = turn_embeddings.device

        # Generate causal mask and ensure it has the same dtype as turn_embeddings
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device).to(turn_embeddings.dtype)

        # Create numerical speaker IDs for the current batch
        numerical_speaker_ids = []
        global_speaker_id_map = {}
        next_id = 0

        for b_idx in range(B):
            chain_speakers = speaker_ids_list[b_idx]
            chain_numerical_ids = []
            for speaker_str in chain_speakers:
                if speaker_str not in global_speaker_id_map:
                    global_speaker_id_map[speaker_str] = next_id
                    next_id += 1
                chain_numerical_ids.append(global_speaker_id_map[speaker_str])
            numerical_speaker_ids.append(chain_numerical_ids)

        # Convert to tensor (B, seq_len)
        speaker_ids_tensor = torch.tensor(numerical_speaker_ids, dtype=torch.long, device=device)

        # Create same speaker boolean mask (B, seq_len, seq_len)
        same_speaker_bool_mask = (speaker_ids_tensor.unsqueeze(2) == speaker_ids_tensor.unsqueeze(1))

        # Inter-speaker mask: allow attention between different speakers, block attention to the same speaker
        inf_val = torch.finfo(turn_embeddings.dtype).min
        zero_tensor = torch.tensor(0., dtype=turn_embeddings.dtype, device=device)
        inter_speaker_additive_mask = torch.where(~same_speaker_bool_mask, zero_tensor, inf_val)

        # Always allow self-attention so turn 0 (and any turn with no prior cross-speaker turn)
        # never gets an all-masked row, which would produce NaN from softmax.
        eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)  # (1, S, S)
        inter_speaker_additive_mask = inter_speaker_additive_mask.masked_fill(eye, 0.0)

        # Combine causal mask with inter-speaker mask
        final_mask = causal_mask.unsqueeze(0) + inter_speaker_additive_mask

        # Reshape the mask to (B * nhead, S, S)
        nhead_local = self.transformer_encoder.layers[0].self_attn.num_heads
        final_mask = final_mask.unsqueeze(1).expand(-1, nhead_local, -1, -1).reshape(B * nhead_local, seq_len, seq_len)

        output = self.transformer_encoder(turn_embeddings, mask=final_mask, is_causal=False)

        return output
    
class SpeakerAwareTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Instantiate separate ContextAwareTransformer for intra-speaker masking
        self.intra_transformer = IntraSpeakerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # Instantiate separate ContextAwareTransformer for inter-speaker masking
        self.inter_transformer = InterSpeakerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, turn_embeddings: torch.Tensor, speaker_ids_list: List[List[str]]):
        intra_emb = self.intra_transformer(turn_embeddings, speaker_ids_list=speaker_ids_list)
        inter_emb = self.inter_transformer(turn_embeddings, speaker_ids_list=speaker_ids_list)
        # keep separate so the caller can concat all three streams (ctx, intra, inter)
        # as independent head groups before the shared FFN (eqs 9-11)
        return intra_emb, inter_emb
    

class ContextAwareTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # TransformerEncoderLayer is the basic building block for the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Set batch_first to True for (B, S, E) input
        )
        # TransformerEncoder stacks multiple EncoderLayers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)



    def forward(self, turn_embeddings: torch.Tensor):
        T = turn_embeddings.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=turn_embeddings.device)
        output = self.transformer_encoder(turn_embeddings, mask=causal_mask, is_causal=True)
        return output





class CrossModalFusionAttention(nn.Module):
    """
    CFA module from sec 2.4 of the paper. Each modality's Q attends to the
    other's K/V, and the resulting delta is layer-normed back into the original.

    Equations 12-15:
        delta_a->t = softmax(Q_t K_a^T / sqrt(d)) V_a
        delta_t->a = softmax(Q_a K_t^T / sqrt(d)) V_t
        Z_a = LN(Z_a + delta_t->a)
        Z_t = LN(Z_t + delta_a->t)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.nhead   = nhead
        self.d_head  = d_model // nhead

        # separate projections for each modality so each gets its own Q/K/V
        self.q_audio = nn.Linear(d_model, d_model)
        self.k_audio = nn.Linear(d_model, d_model)
        self.v_audio = nn.Linear(d_model, d_model)

        self.q_text  = nn.Linear(d_model, d_model)
        self.k_text  = nn.Linear(d_model, d_model)
        self.v_text  = nn.Linear(d_model, d_model)

        self.norm_audio = nn.LayerNorm(d_model)
        self.norm_text  = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B * nhead, T, d_head) for batched matmul
        B, T, _ = x.shape
        x = x.view(B, T, self.nhead, self.d_head)
        x = x.permute(0, 2, 1, 3)              # (B, nhead, T, d_head)
        return x.reshape(B * self.nhead, T, self.d_head)

    def _merge_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # inverse of _split_heads
        x = x.view(B, self.nhead, T, self.d_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, self.d_model)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # scaled dot-product; q/k/v are already split into heads
        scale  = self.d_head ** -0.5
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.bmm(weights, v)

    def forward(
        self,
        z_audio: torch.Tensor,   # (B, T, d_model) -- output of intra-modal audio encoder
        z_text:  torch.Tensor,   # (B, T, d_model) -- output of intra-modal text encoder
    ):
        B, T, _ = z_audio.shape

        # project each modality independently
        q_a = self._split_heads(self.q_audio(z_audio))
        k_a = self._split_heads(self.k_audio(z_audio))
        v_a = self._split_heads(self.v_audio(z_audio))

        q_t = self._split_heads(self.q_text(z_text))
        k_t = self._split_heads(self.k_text(z_text))
        v_t = self._split_heads(self.v_text(z_text))

        # text queries into audio keys/values -> what audio says about text context
        delta_a2t = self._merge_heads(self._attend(q_t, k_a, v_a), B, T)
        # audio queries into text keys/values -> what text says about audio context
        delta_t2a = self._merge_heads(self._attend(q_a, k_t, v_t), B, T)

        # residual add + norm (eqs 14-15)
        z_audio_fused = self.norm_audio(z_audio + delta_t2a)
        z_text_fused  = self.norm_text(z_text  + delta_a2t)

        return z_audio_fused, z_text_fused  # both (B, T, d_model)


class _IntraModalFFN(nn.Module):
    # eqs 9-11: the three d_sub streams are already concatenated to d_model
    # before arriving here, so no projection needed -- just residual + FFN + LN
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1   = nn.Linear(d_model, dim_feedforward)
        self.ff2   = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
 
    def forward(self, z_cat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # eq 9: residual with H* (original utterance embedding), then LN
        z = self.norm1(z_cat + h)
        # eqs 10-11: FFN + residual LN
        o = self.ff2(self.drop(F.relu(self.ff1(z))))
        return self.norm2(o + z)
    

class SCFA(nn.Module):
    """
    Speaker-aware Cross-modal Fusion Architecture
    
    Design credit to Zhao et al. 2023.

    Pipeline (matching fig 1 + sec 2):
        1. DualModalityEmbedder  -- utterance-level audio + text features
        2. ContextAwareTransformer (x2) -- global context for each modality
        3. SpeakerAwareTransformer (x2) -- intra/inter speaker attention per modality
        4. Intra-modal combination layer -- fuse context and speaker-aware streams -> 2 intra-modal vectors
        5. CrossModalFusionAttention -- cross-modal interaction -> 2 inter-modal vectors
        6. Concatenate all vectors from steps 4,5
    """

    def __init__(
        self,
        max_turns:      int,
        embedder:       DualModalityEmbedder,
        d_model:        int,
        num_ctx_layers: int,
        num_spk_layers: int,
        dim_feedforward: int,
        nhead:          int=1,
        dropout:        float = 0.1,
    ):
        super().__init__()
 
        self.embedder = embedder
 
        # each stream gets d_model//3 width so their concat lands back at d_model,
        # mirroring how MHA splits into heads then concatenates (sec 2.3)
        if d_model % 3 != 0:
            raise ValueError(f"d_model must be divisible by 3 (got {d_model})")
        d_sub  = d_model // 3
        n_sub  = max(1, nhead // 3)  # at least 1 head per sub-transformer

        self.turn_pos_enc = TurnPositionEncoding(d_model, max_turns)
 
        # context transformers -- one per modality (sec 2.3, eqs 5-6)
        self.ctx_audio = ContextAwareTransformer(d_sub, n_sub, num_ctx_layers, dim_feedforward, dropout)
        self.ctx_text  = ContextAwareTransformer(d_sub, n_sub, num_ctx_layers, dim_feedforward, dropout)
 
        # speaker-aware transformers -- one per modality (sec 2.3, eqs 7-11)
        self.spk_audio = SpeakerAwareTransformer(d_sub, n_sub, num_spk_layers, dim_feedforward, dropout)
        self.spk_text  = SpeakerAwareTransformer(d_sub, n_sub, num_spk_layers, dim_feedforward, dropout)
 
        # FFN + LN to fuse the three d_sub streams after concat (eqs 9-11)
        # concat of [ctx, intra, inter] = (B, T, 3*d_sub) = (B, T, d_model), no proj needed
        self.ffn_audio = _IntraModalFFN(d_model, dim_feedforward, dropout)
        self.ffn_text  = _IntraModalFFN(d_model, dim_feedforward, dropout)
 
        # cross-modal fusion (sec 2.4, eqs 12-15)
        self.cfa = CrossModalFusionAttention(d_model, nhead, dropout)

        # project d_model -> d_sub before each set of sub-transformers
        self.proj_audio = nn.Linear(d_model, d_sub)
        self.proj_text  = nn.Linear(d_model, d_sub)


    def _intra_modal_encode(
        self,
        emb:             torch.Tensor,   # (B, T, d_model) -- kept for FFN residual
        emb_sub:         torch.Tensor,   # (B, T, d_sub)   -- input to sub-transformers
        ctx_transformer: ContextAwareTransformer,
        spk_transformer: SpeakerAwareTransformer,
        ffn:             "_IntraModalFFN",
        speaker_ids:     List[List[str]],
    ) -> torch.Tensor:

        z_ctx            = ctx_transformer(emb_sub)
        z_intra, z_inter = spk_transformer(emb_sub, speaker_ids)

        z_cat = torch.cat([z_ctx, z_intra, z_inter], dim=-1)
        return ffn(z_cat, emb)  # residual uses full d_model emb

    @torch._dynamo.disable
    def forward(
        self,
        audio:       torch.Tensor,       # (B, T, samples)
        lengths:     torch.Tensor,       # (B, T)
        texts:       List[List[str]],    # [B][T]
        speaker_ids: List[List[str]],    # [B][T]
        text_only:   Optional[torch.Tensor] = None,  # (B, T) bool
    ):
        # 1 utterance encoding
        text_emb, audio_emb = self.embedder(audio, lengths, texts, text_only)
        # both (B, T, d_model)

        # 1.5 positional encoding
        # apply shared positional encoding before the modality-specific transformers
        # so both streams are grounded in the same turn positions going into CFA
        audio_emb = self.turn_pos_enc(audio_emb)
        text_emb  = self.turn_pos_enc(text_emb)

 
        # project to d_sub for sub-transformers
        audio_sub = self.proj_audio(audio_emb)
        text_sub  = self.proj_text(text_emb)

        z_audio = self._intra_modal_encode(
            audio_emb, audio_sub, self.ctx_audio, self.spk_audio, self.ffn_audio, speaker_ids
        )
        z_text = self._intra_modal_encode(
            text_emb, text_sub, self.ctx_text, self.spk_text, self.ffn_text, speaker_ids
        )

        # both (B, T, d_model)
 
        # 4 cross-modal fusion 
        z_audio_fused, z_text_fused = self.cfa(z_audio, z_text)
        # both (B, T, d_model)
 
        # Concatenate all 4 streams -> (B, T, 4 * d_model)
        return torch.cat([z_audio, z_text, z_audio_fused, z_text_fused], dim=-1)


class DialoguePooler(nn.Module):
    # collapses (B, T, d) -> (B, d) for prefix generation
    def __init__(self, d_model: int, mode: str = "attentive"):
        super().__init__()
        assert mode in ("attentive", "last"), f"unknown pooling mode: {mode}"
        self.mode = mode
        if mode == "attentive":
            self.attn_pool = SelfAttentivePooling(d_model)

    def forward(self, scfa_out: torch.Tensor) -> torch.Tensor:
        # scfa_out: (B, T, 4*d_model)
        if self.mode == "last":
            return scfa_out[:, -1, :]  # anchor turn has attended over all prior turns causally
        return self.attn_pool(scfa_out)  # (B, 4*d_model)