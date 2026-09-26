from modules.util import path_util
from modules.util.enum.DataType import DataType
from modules.util.enum.PathIOType import PathIOType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.model_catalog import MODEL_TYPE_CHOICES


class BaseConvertModelUIView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, controller, ui_state, on_model_or_method_change):
        # model type
        self.components.label(frame, 0, 0, "Model Type",
                         tooltip="Type of the model")
        self.components.options_kv(
            frame, 0, 1, list(MODEL_TYPE_CHOICES), ui_state, "model_type", command=on_model_or_method_change,
        )

        # training method
        self.components.label(frame, 1, 0, "Training Method",
                         tooltip="The type of model to convert")
        # The view rebuilds the method selector when the selected model changes.

        # input name
        self.components.label(frame, 2, 0, "Input name",
                         tooltip="Filename, directory or hugging face repository of the base model")
        self.components.path_entry(
            frame, 2, 1, ui_state, "input_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # output data type
        self.components.label(frame, 3, 0, "Output Data Type",
                         tooltip="Precision to use when saving the output model")
        self.components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "output_dtype")

        # row 4 (output format) is built by build_dynamic_content -- it depends on model type / training
        # method, so the view rebuilds it via on_model_or_method_change whenever either one changes.

        # output model destination
        self.components.label(frame, 5, 0, "Model Output Destination",
                         tooltip="Filename or directory where the output model is saved")
        self.components.path_entry(
            frame, 5, 1, ui_state, "output_model_destination",
            mode="file",
            io_type=PathIOType.MODEL,
        )

        self.button = self.components.button(frame, 6, 1, "Convert", self.start_conversion)

    def build_dynamic_content(self, frame, controller, ui_state):
        row = 0

        # base model name -- LoRA/embedding conversion needs to load the base model to know its native
        # module names (used to reverse KOHYA/LEGACY un-flattening); a fine-tune conversion's "Input name"
        # already is the base model, so this field only applies to LoRA/embedding conversions.
        if controller.convert_model_args.training_method in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
            self.components.label(frame, row, 0, "Base Model Name",
                             tooltip="Filename, directory or Hugging Face repository of the base model this LoRA/embedding was trained on")
            self.components.path_entry(
                frame, row, 1, ui_state, "base_model_name",
                mode="file", path_modifier=path_util.json_path_modifier
            )
            row += 1

        # output format
        self.components.label(frame, row, 0, "Output Format",
                         tooltip="Format to use when saving the output model")
        self.components.options_kv(frame, row, 1, controller.get_output_formats(), ui_state, "output_model_format")
