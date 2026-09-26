from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class SanaEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_embedding", "gemma"),
    )
