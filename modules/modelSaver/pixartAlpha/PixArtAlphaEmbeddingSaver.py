from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class PixArtAlphaEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_embedding", "t5"),
    )
