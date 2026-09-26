from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.anima.AnimaModelSaver import AnimaModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

AnimaFineTuneModelSaver = make_fine_tune_model_saver(
    [ModelType.ANIMA, ModelType.ANIMA_QWEN21_VAE],
    model_class=AnimaModel,
    model_saver_class=AnimaModelSaver,
    embedding_saver_class=None,
)
