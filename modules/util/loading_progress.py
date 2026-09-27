"""Report dataset preparation to both the terminal and the training status line."""

from contextlib import contextmanager
from contextvars import ContextVar

from tqdm import tqdm

_status = ContextVar("loading_status", default=lambda _text: None)


@contextmanager
def loading_status(callback):
    token = _status.set(callback)
    try:
        yield
    finally:
        _status.reset(token)


def report_loading_status(message):
    _status.get()(message)


def loading_callback():
    return _status.get()


class LoadingProgress(tqdm):
    def __init__(self, *args, **kwargs):
        self._report_status = _status.get()
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        result = super().display(msg, pos)
        total = f"/{self.total}" if self.total is not None else ""
        percent = f" ({self.n / self.total:.0%})" if self.total else ""
        self._report_status(f"{self.desc}: {self.n}{total} {self.unit}{percent}")
        return result
