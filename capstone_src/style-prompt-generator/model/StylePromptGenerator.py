import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.DialogueEncoder import SCFA, DialoguePooler


# not really used, these are controlled via config parameters now
LLM_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # or whichever you choose
LLM_DIM  = 3072  # must match model's hidden_size


def load_tinyllama(device: str = "cpu", torch_dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(LLM_REPO)
    model = AutoModelForCausalLM.from_pretrained(LLM_REPO, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


class StylePromptHead(nn.Module):
    # Projects a pooled dialogue vector into K prefix vectors in TinyLlama
    # embedding space, following ClipCap / StyleCap.
    #
    # Expects a pre-pooled (B, 4*d_model) vector from DialoguePooler --
    # turn selection and pooling are handled upstream, not here.
    #
    # Pipeline:
    #   1. Compress + fuse the 4 SCFA streams into a single context vector z
    #   2. Transformer mapping network: K learnable constants attend to z,
    #      producing K output vectors in LLM embedding space

    def __init__(
        self,
        d_model:            int,
        num_prefix_tokens:  int,
        llm_dim:            int   = LLM_DIM,
        num_mapping_layers: int   = 8,
        nhead:              int   = 8,
        dropout:            float = 0.1,
    ):
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens

        self.stream_compressors = nn.ModuleList([
            nn.Linear(d_model, d_model // 2) for _ in range(4)
        ])
        

        self.stream_to_llm = nn.ModuleList([
            nn.Linear(d_model // 2, llm_dim) for _ in range(4)
        ])



        self.prefix_const = nn.Parameter(torch.randn(num_prefix_tokens, llm_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_dim,
            nhead=nhead,
            # dim_feedforward=llm_dim * 2,  # use default 2048, matching papers
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_mapping_layers, enable_nested_tensor=False)



    def forward(self, dialogue_vec: torch.Tensor) -> torch.Tensor:
        # dialogue_vec: (B, 4 * d_model) -- already pooled by DialoguePooler
        B, D = dialogue_vec.shape
        d_model = D // 4

        # split back into the 4 SCFA streams and compress each independently 
        # to create distinct context tokens
        streams    = dialogue_vec.split(d_model, dim=-1)
        compressed = [F.relu(comp(s)) for comp, s in zip(self.stream_compressors, streams)]
        ctx_tokens = torch.stack([proj(c) for proj, c in zip(self.stream_to_llm, compressed)], dim=1)  # (B, 4, llm_dim)

        # z sits at position 0; learnable constants learn to query it
        consts     = self.prefix_const.unsqueeze(0).expand(B, -1, -1)
        seq        = torch.cat([ctx_tokens, consts], dim=1)   # (B, 4+K, llm_dim)
        out        = self.transformer(seq)
        return out[:, 4:, :]# drop z position, return K prefix vectors



class StylePromptGenerator(nn.Module):
    # StylePromptHead + frozen TinyLlama.
    #
    # Prefix vectors are prepended to the (optionally present) system prompt
    # embeddings, then TinyLlama generates autoregressively. Only the
    # StylePromptHead is trained.

    def __init__(
        self,
        style_head:     StylePromptHead,
        tokenizer:      AutoTokenizer,
        llm:            AutoModelForCausalLM,
        max_prompt_tokens: int,
        system_prompt:  Optional[str] = None,
        max_new_tokens: int = 80,
    ):
        super().__init__()
        self.style_head     = style_head
        self.tokenizer      = tokenizer
        self.llm            = llm
        # Silence the do_sample=False vs temperature/top_p conflict in GenerationConfig
        self.llm.generation_config.temperature = None
        self.llm.generation_config.top_p = None
        self.system_prompt  = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens

        for p in self.llm.parameters():
            p.requires_grad = False

    def _embed_text(self, texts: List[str], device) -> tuple:
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_prompt_tokens
        )
        input_ids = tokens.input_ids.to(device)
        attn_mask = tokens.attention_mask.to(device)
        embeds    = self.llm.get_input_embeddings()(input_ids)  # (B, L, LLM_DIM)
        return embeds, attn_mask

    @torch.no_grad()
    def generate(self, dialogue_vec: torch.Tensor) -> List[str]:
        # dialogue_vec: (B, 4*d_model) -- pooled upstream by DialoguePooler
        B      = dialogue_vec.shape[0]
        device = dialogue_vec.device

        prefix_embeds = self.style_head(dialogue_vec)  # (B, K, LLM_DIM)
        K             = prefix_embeds.shape[1]
        prefix_mask   = torch.ones(B, K, dtype=torch.long, device=device)

        if self.system_prompt is not None:
            prompt_embeds, prompt_mask = self._embed_text([self.system_prompt] * B, device)
            inputs_embeds  = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        else:
            inputs_embeds  = prefix_embeds
            attention_mask = prefix_mask

        # must cast f32 embeds to bf16 dtype for llm
        inputs_embeds = inputs_embeds.to(self.llm.dtype)

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def forward(self, dialogue_vec: torch.Tensor) -> List[str]:
        return self.generate(dialogue_vec)


class SCFAWithStyleHead(nn.Module):
    # Full pipeline: SCFA -> DialoguePooler -> StylePromptGenerator -> style prompt strings.

    def __init__(
        self,
        scfa:            "SCFA",
        pooler:          "DialoguePooler",
        style_generator: StylePromptGenerator,
    ):
        super().__init__()
        self.scfa            = scfa
        self.pooler          = pooler
        self.style_generator = style_generator

    def forward(
        self,
        audio:       torch.Tensor,
        lengths:     torch.Tensor,
        texts,
        speaker_ids,
        text_only:   Optional[torch.Tensor] = None,
    ):
        # (B, T, 4*d_model)
        dialogue_ctx = self.scfa(audio, lengths, texts, speaker_ids, text_only)
        # (B, 4*d_model)
        dialogue_vec = self.pooler(dialogue_ctx)
        style_prompts = self.style_generator(dialogue_vec)

        return style_prompts, dialogue_vec


def build_style_generator(
    scfa:               "SCFA",
    pooler:             "DialoguePooler",
    d_model:            int           = 768,
    num_prefix_tokens:  int           = 10,
    num_mapping_layers: int           = 8,
    nhead:              int           = 8,
    max_new_tokens:     int           = 80,
    system_prompt:      Optional[str] = None,
    device:             str           = "cpu",
    torch_dtype                        = torch.float32,
) -> SCFAWithStyleHead:
    # Loads TinyLlama, wires everything up, returns a ready-to-use SCFAWithStyleHead.
    #
    # pooler should be a DialoguePooler instance configured with your chosen mode.
    #
    # system_prompt controls text conditioning after the prefix:
    #   None -> prefix-only generation (ClipCap / StyleCap style)
    #   str  -> prefix + text anchor
    #
    # Example:
    #   pooler = DialoguePooler(d_model=768 * 4, mode="attentive")
    #   model  = build_style_generator(scfa, pooler, device="cuda")
    #   style_prompts, vec = model(audio, lengths, texts, speaker_ids)
    tokenizer, llm = load_tinyllama(device=device, torch_dtype=torch_dtype)

    head = StylePromptHead(
        d_model=d_model,
        num_prefix_tokens=num_prefix_tokens,
        num_mapping_layers=num_mapping_layers,
        nhead=nhead,
    )

    generator = StylePromptGenerator(
        style_head=head,
        tokenizer=tokenizer,
        llm=llm,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )

    return SCFAWithStyleHead(scfa=scfa, pooler=pooler, style_generator=generator)