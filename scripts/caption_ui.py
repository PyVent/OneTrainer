from util.import_util import script_imports

script_imports()

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.PySide6CaptionUIView import PySide6CaptionUIView
from modules.util.args.CaptionUIArgs import CaptionUIArgs
from modules.util.ui.pyside6_util import create_application


def main():
    args = CaptionUIArgs.parse_args()
    _app = create_application()
    ui = CaptionUIController(args.dir, args.include_subdirectories).create_window(None, PySide6CaptionUIView)
    ui.exec()


if __name__ == '__main__':
    main()
