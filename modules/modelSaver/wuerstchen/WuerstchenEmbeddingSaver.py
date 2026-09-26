from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class WuerstchenEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("prior_text_encoder_embedding", "clip_g"),
    )
