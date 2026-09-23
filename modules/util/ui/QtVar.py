from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject


class QtVar:
    """A value shared by a config field and its Qt controls."""

    def __init__(self, value: Any = ""):
        self._value = value
        self._subscribers: dict[int, Callable[[Any], None]] = {}
        self._next_id = 0

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value
        for callback in list(self._subscribers.values()):
            callback(value)

    def subscribe(self, callback: Callable[[Any], None], owner: QObject | None = None) -> int:
        """Notify on every set; an owned subscription ends with its Qt control."""
        self._next_id += 1
        subscription_id = self._next_id
        self._subscribers[subscription_id] = callback
        if owner is not None:
            owner.destroyed.connect(lambda: self.unsubscribe(subscription_id))
        return subscription_id

    def unsubscribe(self, subscription_id: int) -> None:
        self._subscribers.pop(subscription_id, None)
