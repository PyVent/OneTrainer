from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class HunyuanVideoEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_1_embedding", "llama"),
        ("text_encoder_2_embedding", "clip_l"),
    )
