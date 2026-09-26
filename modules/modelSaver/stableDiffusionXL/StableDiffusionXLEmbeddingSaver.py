from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class StableDiffusionXLEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_1_embedding", "clip_l"),
        ("text_encoder_2_embedding", "clip_g"),
    )
