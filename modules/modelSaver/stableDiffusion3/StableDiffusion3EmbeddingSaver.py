from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class StableDiffusion3EmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_1_embedding", "clip_l"),
        ("text_encoder_2_embedding", "clip_g"),
        ("text_encoder_3_embedding", "t5"),
    )
