from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class FluxEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_1_embedding", "clip_l"),
        ("text_encoder_2_embedding", "t5"),
    )
