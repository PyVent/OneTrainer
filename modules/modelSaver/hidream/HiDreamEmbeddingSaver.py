from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class HiDreamEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_1_embedding", "clip_l"),
        ("text_encoder_2_embedding", "clip_g"),
        ("text_encoder_3_embedding", "t5"),
        ("text_encoder_4_embedding", "llama"),
    )
