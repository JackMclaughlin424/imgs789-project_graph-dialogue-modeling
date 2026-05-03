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