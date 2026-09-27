from util.import_util import script_imports

script_imports()

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.util.args.ConvertModelArgs import ConvertModelArgs


def main():
    controller = ConvertModelUIController()
    controller.convert_model_args = ConvertModelArgs.parse_args()
    controller.perform_conversion()


if __name__ == '__main__':
    main()
