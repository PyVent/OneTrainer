from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class ChromaEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_embedding", "t5"),
    )
