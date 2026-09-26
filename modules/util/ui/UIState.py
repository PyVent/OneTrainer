from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.type_util import issubclass_safe
from modules.util.ui.QtVar import QtVar


class BaseUIState:
    """Editable Qt values backed by a config, object, or dictionary."""

    def __init__(self, obj: Any):
        self.obj = obj
        self.__var_types: dict[str, type] = {}
        self.__var_nullables: dict[str, bool] = {}
        self.__var_defaults: dict[str, Any] = {}
        self.__vars: dict[str, QtVar | BaseUIState] = {}

        for name, value, value_type, nullable, default in self._fields(obj):
            self.__var_types[name] = value_type
            self.__var_nullables[name] = nullable
            self.__var_defaults[name] = default
            if not (value_type in (str, bool, int, float)
                    or issubclass_safe(value_type, (Enum, BaseConfig))):
                continue
            if issubclass_safe(value_type, BaseConfig):
                self.__vars[name] = type(self)(value)
            else:
                var = QtVar(self._display_value(value, value_type))
                var.subscribe(
                    lambda current, field=name, kind=value_type, allow_null=nullable:
                    self._write_field(field, current, kind, allow_null)
                )
                self.__vars[name] = var

    @staticmethod
    def _fields(obj: Any):
        if isinstance(obj, BaseConfig):
            defaults = getattr(obj, "default_values", {})
            for name, value_type in obj.types.items():
                yield name, getattr(obj, name), value_type, obj.nullables.get(name, False), defaults.get(name)
        else:
            values = obj.items() if isinstance(obj, dict) else vars(obj).items()
            for name, value in values:
                if isinstance(value, (str, Enum, bool, int, float)):
                    yield name, value, type(value), False, None

    @staticmethod
    def _display_value(value: Any, value_type: type) -> Any:
        if value_type is bool:
            return bool(value)
        if value is None:
            return ""
        return value if value_type is str else str(value)

    @staticmethod
    def _parse_value(value: Any, value_type: type, nullable: bool) -> Any:
        if value_type is bool:
            return bool(value)
        if nullable and value in ("", "None"):
            return None
        if issubclass_safe(value_type, Enum):
            return value_type[value]
        if value_type in (int, float):
            try:
                return value_type(value)
            except (ValueError, TypeError):
                return None
        return value

    def _write_field(self, name: str, value: Any, value_type: type, nullable: bool) -> None:
        parsed = self._parse_value(value, value_type, nullable)
        if isinstance(self.obj, dict):
            self.obj[name] = parsed
        else:
            setattr(self.obj, name, parsed)

    def update(self, obj: Any) -> None:
        self.obj = obj
        for name, value, value_type, _, _ in self._fields(obj):
            var = self.__vars.get(name)
            if isinstance(var, BaseUIState):
                var.update(value)
            elif isinstance(var, QtVar):
                var.set(self._display_value(value, value_type))

    def get_var(self, name: str) -> QtVar | BaseUIState:
        state: BaseUIState = self
        parts = name.split(".")
        for part in parts[:-1]:
            nested = state.__vars[part]
            if not isinstance(nested, BaseUIState):
                raise KeyError(name)
            state = nested
        return state.__vars[parts[-1]]

    @dataclass(frozen=True)
    class VarMeta:
        type: type | None
        nullable: bool
        default: Any

    def get_field_metadata(self, name: str) -> BaseUIState.VarMeta:
        state: BaseUIState = self
        parts = name.split(".")
        for part in parts[:-1]:
            nested = state.__vars.get(part)
            if not isinstance(nested, BaseUIState):
                return BaseUIState.VarMeta(None, False, None)
            state = nested
        leaf = parts[-1]
        return BaseUIState.VarMeta(
            state.__var_types.get(leaf),
            state.__var_nullables.get(leaf, False),
            state.__var_defaults.get(leaf),
        )
