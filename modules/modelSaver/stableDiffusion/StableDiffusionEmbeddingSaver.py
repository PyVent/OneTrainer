from modules.modelSaver.mixin.EmbeddingSaverMixin import EmbeddingSaverMixin


class StableDiffusionEmbeddingSaver(EmbeddingSaverMixin):
    embedding_fields = (
        ("text_encoder_embedding", "emp_params"),
    )
