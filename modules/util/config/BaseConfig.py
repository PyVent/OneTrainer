from collections.abc import Callable
from enum import Enum
from typing import Any, get_args, get_origin

from modules.util.type_util import issubclass_safe


class ConfigValidationError(ValueError):
    def __init__(self, field: str, reason: str):
        super().__init__(f"Invalid configuration field '{field}': {reason}")


class BaseConfig:
    config_version: int
    config_migrations: dict[int, Callable[[dict], dict]]

    def __init__(
            self,
            data: list[tuple[str, Any, type, bool]],
            config_version: int | None = None,
            config_migrations: dict[int, Callable[[dict], dict]] | None = None
    ):
        self.config_version = config_version if config_version is not None else 0
        self.config_migrations = config_migrations if config_migrations is not None else {}

        self.types = {}
        self.nullables = {}
        self.default_values = {}
        for (name, value, var_type, nullable) in data:
            setattr(self, name, value)
            self.types[name] = var_type
            self.nullables[name] = nullable
            self.default_values[name] = value

    def to_dict(self) -> dict:
        data = {
            '__version': self.config_version,
        }

        for name in self.types:
            value = getattr(self, name)
            if issubclass_safe(self.types[name], BaseConfig):
                data[name] = value.to_dict()
            elif self.types[name] is list or get_origin(self.types[name]) is list:
                if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[0], BaseConfig):
                    data[name] = [le.to_dict() for le in value] if value is not None else None
                else:
                    data[name] = value
            elif self.types[name] is dict or get_origin(self.types[name]) is dict:
                if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[1], BaseConfig):
                    dict_data = {}
                    for dict_key, dict_value in value.items():
                        dict_data[dict_key] = dict_value.to_dict()
                    data[name] = dict_data
                else:
                    data[name] = value
            elif self.types[name] is str:
                data[name] = value
            elif issubclass_safe(self.types[name], Enum):
                data[name] = None if value is None else str(value)
            elif self.types[name] is bool or self.types[name] is int:
                data[name] = value
            elif self.types[name] is float:
                if value in [float('inf'), float('-inf')]:
                    data[name] = str(value)
                else:
                    data[name] = value

        return data

    def from_dict(
            self,
            data: dict,
            migrate: bool = True,
            *,
            strict: bool = False,
            _path: str = "",
    ) -> 'BaseConfig':
        """Apply present fields; strict mode rejects invalid values in user input."""
        if strict and not isinstance(data, dict):
            raise ConfigValidationError(_path or "<root>", "expected an object")

        if strict and '__version' in data:
            saved_version = data['__version']
            if type(saved_version) is not int or not 0 <= saved_version <= self.config_version:
                raise ConfigValidationError(_path or "<root>", "unsupported configuration version")

        if migrate:
            version = 0
            if '__version' in data:
                version = data['__version']

            while version in self.config_migrations:
                data = self.config_migrations[version](data)
                version += 1

            if strict and version < self.config_version:
                raise ConfigValidationError(_path or "<root>", f"no migration from version {version}")

        for name in self.types:
            if name not in data:
                continue

            field_path = f"{_path}.{name}" if _path else name
            try:
                if issubclass_safe(self.types[name], BaseConfig):
                    getattr(self, name).from_dict(data[name], migrate=migrate, strict=strict, _path=field_path)
                elif self.types[name] is list or get_origin(self.types[name]) is list:
                    if strict and data[name] is not None and not isinstance(data[name], list):
                        raise TypeError("expected a list")
                    if strict and data[name] is None and not self.nullables[name]:
                        raise TypeError("expected a list")
                    if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[0], BaseConfig):
                        list_type = get_args(self.types[name])[0]
                        if data[name] is not None:
                            old_value = \
                                getattr(self, name) if hasattr(self, name) and getattr(self, name) is not None else []
                            value = []
                            for i in range(len(data[name])):
                                if i < len(old_value) and i < len(data[name]):
                                    value.append(old_value[i].from_dict(
                                        data[name][i], migrate=migrate, strict=strict,
                                        _path=f"{field_path}[{i}]",
                                    ))
                                else:
                                    value.append(list_type.default_values().from_dict(
                                        data[name][i], migrate=migrate, strict=strict,
                                        _path=f"{field_path}[{i}]",
                                    ))
                        else:
                            value = None
                        setattr(self, name, value)
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] is dict or get_origin(self.types[name]) is dict:
                    if data[name] is None and self.nullables[name]:
                        setattr(self, name, None)
                        continue
                    if strict and not isinstance(data[name], dict):
                        raise TypeError("expected an object")
                    if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[1], BaseConfig):
                        dict_type = get_args(self.types[name])[1]
                        value = {}
                        for dict_key, dict_value in data[name].items():
                            value[dict_key] = dict_type.default_values().from_dict(
                                dict_value, migrate=migrate, strict=strict,
                                _path=f"{field_path}[{dict_key!r}]",
                            )
                        setattr(self, name, value)
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] is str:
                    if strict and data[name] is None and not self.nullables[name]:
                        raise TypeError("expected a string")
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else str(data[name]))
                    else:
                        setattr(self, name, str(data[name]))
                elif issubclass_safe(self.types[name], Enum):
                    if strict and data[name] is None and not self.nullables[name]:
                        raise TypeError("expected an enum value")
                    if strict and data[name] is not None and not isinstance(data[name], (str, self.types[name])):
                        raise TypeError("expected an enum value")
                    if isinstance(data[name], str):
                        if self.nullables[name]:
                            setattr(self, name, None if data[name] is None else self.types[name][data[name]])
                        else:
                            setattr(self, name, self.types[name][data[name]])
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] is bool:
                    if not strict:
                        setattr(self, name, data[name])
                    else:
                        value = data[name]
                        if value is None and self.nullables[name]:
                            pass
                        elif isinstance(value, str):
                            normalized = value.strip().lower()
                            if normalized in ("true", "1", "yes"):
                                value = True
                            elif normalized in ("false", "0", "no"):
                                value = False
                            else:
                                raise TypeError("expected a boolean")
                        elif type(value) is int and value in (0, 1):
                            value = bool(value)
                        elif type(value) is not bool:
                            raise TypeError("expected a boolean")
                        setattr(self, name, value)
                elif self.types[name] is int:
                    if strict and isinstance(data[name], bool):
                        raise TypeError("expected an integer")
                    if strict and isinstance(data[name], float) and not data[name].is_integer():
                        raise TypeError("expected an integer")
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else int(data[name]))
                    else:
                        setattr(self, name, int(data[name]))
                elif self.types[name] is float:
                    if strict and isinstance(data[name], bool):
                        raise TypeError("expected a number")
                    # check for strings to support dicts loaded from json
                    if data[name] in [float('inf'), float('-inf'), 'inf', '-inf']:
                        setattr(self, name, float(data[name]))
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else float(data[name]))
                    else:
                        setattr(self, name, float(data[name]))
            except ConfigValidationError:
                raise
            except Exception as exc:  # noqa: PERF203
                if strict:
                    raise ConfigValidationError(field_path, str(exc)) from exc
                print(f"Could not set {name} as {str(data[name])}")

        return self
