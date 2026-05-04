from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypedDict


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class _Unset:
    pass


UNSET: Final = _Unset()


@dataclass(frozen=True)
class PaintifyConfig:
    input_path: Path
    output_dir: Path
    difficulty: str = "easy"
    max_colors: int = 8
    max_size: int = 180
    min_region_size: int = 24
    smooth_radius: float = 1.0
    starter_palette: str | None = "basic"
    max_regions: int | None = None
    seed: int = 0

    def validated(self) -> "PaintifyConfig":
        if self.max_colors < 2:
            raise SettingsError("max_colors must be at least 2")
        if self.max_size < 8:
            raise SettingsError("max_size must be at least 8 pixels")
        if self.min_region_size < 1:
            raise SettingsError("min_region_size must be positive")
        if self.smooth_radius < 0:
            raise SettingsError("smooth_radius must be non-negative")
        if self.max_regions is not None and self.max_regions < 1:
            raise SettingsError("max_regions must be positive")
        return self


class PresetValues(TypedDict):
    max_colors: int
    max_size: int
    min_region_size: int
    smooth_radius: float
    starter_palette: str | None
    max_regions: int


PRESETS: dict[str, PresetValues] = {
    "easy": {
        "max_colors": 15,
        "max_size": 768,
        "min_region_size": 40,
        "smooth_radius": 0.6,
        "starter_palette": None,
        "max_regions": 300,
    },
    "medium": {
        "max_colors": 20,
        "max_size": 1024,
        "min_region_size": 20,
        "smooth_radius": 0.4,
        "starter_palette": None,
        "max_regions": 700,
    },
    "hard": {
        "max_colors": 30,
        "max_size": 1280,
        "min_region_size": 12,
        "smooth_radius": 0.25,
        "starter_palette": None,
        "max_regions": 1200,
    },
}


@dataclass(frozen=True)
class SettingsOverrides:
    max_colors: int | None = None
    max_size: int | None = None
    min_region_size: int | None = None
    smooth_radius: float | None = None
    starter_palette: str | None | _Unset = UNSET
    max_regions: int | None = None


@dataclass(frozen=True)
class PaintifyOptions:
    input_path: Path
    output_dir: Path
    difficulty: str = "easy"
    use_preset: bool = True
    overrides: SettingsOverrides = field(default_factory=SettingsOverrides)
    seed: int = 0


class SettingsResolver:
    @classmethod
    def apply_preset(
        cls,
        config: PaintifyConfig,
        overrides: dict[str, object] | None = None,
    ) -> PaintifyConfig:
        if config.difficulty not in PRESETS:
            names = ", ".join(sorted(PRESETS))
            raise SettingsError(f"unknown difficulty '{config.difficulty}', choose one of: {names}")
        preset = PRESETS[config.difficulty]
        values: dict[str, object] = {
            "max_colors": preset["max_colors"],
            "max_size": preset["max_size"],
            "min_region_size": preset["min_region_size"],
            "smooth_radius": preset["smooth_radius"],
            "starter_palette": preset["starter_palette"],
            "max_regions": preset["max_regions"],
        }
        values.update(overrides or {})
        return PaintifyConfig(
            input_path=config.input_path,
            output_dir=config.output_dir,
            difficulty=config.difficulty,
            max_colors=cls._as_int(values["max_colors"], "max_colors"),
            max_size=cls._as_int(values["max_size"], "max_size"),
            min_region_size=cls._as_int(values["min_region_size"], "min_region_size"),
            smooth_radius=cls._as_float(values["smooth_radius"], "smooth_radius"),
            starter_palette=cls._as_optional_str(values["starter_palette"], "starter_palette"),
            max_regions=cls._as_int(values["max_regions"], "max_regions"),
            seed=config.seed,
        ).validated()

    @staticmethod
    def preset_names() -> list[str]:
        return sorted(PRESETS)

    def resolve(self, options: PaintifyOptions) -> PaintifyConfig:
        values: dict[str, object] = {}
        self._apply_overrides(values, options)
        if options.use_preset:
            return self.apply_preset(
                PaintifyConfig(
                    input_path=options.input_path,
                    output_dir=options.output_dir,
                    difficulty=options.difficulty,
                    seed=options.seed,
                ),
                overrides=values,
            )
        return PaintifyConfig(
            input_path=options.input_path,
            output_dir=options.output_dir,
            difficulty="manual",
            max_colors=self._required_int(values, "max_colors"),
            max_size=self._required_int(values, "max_size"),
            min_region_size=self._required_int(values, "min_region_size"),
            smooth_radius=self._required_float(values, "smooth_radius"),
            starter_palette=self._required_optional_str(values, "starter_palette"),
            max_regions=self._required_int(values, "max_regions"),
            seed=options.seed,
        ).validated()

    def _apply_overrides(self, values: dict[str, object], options: PaintifyOptions) -> None:
        overrides = options.overrides
        if overrides.max_colors is not None:
            values["max_colors"] = overrides.max_colors
        if overrides.max_size is not None:
            values["max_size"] = overrides.max_size
        if overrides.min_region_size is not None:
            values["min_region_size"] = overrides.min_region_size
        if overrides.smooth_radius is not None:
            values["smooth_radius"] = overrides.smooth_radius
        if not isinstance(overrides.starter_palette, _Unset):
            values["starter_palette"] = overrides.starter_palette
        if overrides.max_regions is not None:
            values["max_regions"] = overrides.max_regions

    @staticmethod
    def _missing_error(name: str) -> SettingsError:
        return SettingsError(f"{name} is required when --no-preset is used")

    @classmethod
    def _as_int(cls, value: object, name: str) -> int:
        if not isinstance(value, int):
            raise SettingsError(f"{name} must be an integer")
        return value

    @classmethod
    def _required_int(cls, values: dict[str, object], name: str) -> int:
        if name not in values:
            raise cls._missing_error(name)
        return cls._as_int(values[name], name)

    @classmethod
    def _as_float(cls, value: object, name: str) -> float:
        if not isinstance(value, int | float):
            raise SettingsError(f"{name} must be a number")
        return float(value)

    @classmethod
    def _required_float(cls, values: dict[str, object], name: str) -> float:
        if name not in values:
            raise cls._missing_error(name)
        return cls._as_float(values[name], name)

    @classmethod
    def _as_optional_str(cls, value: object, name: str) -> str | None:
        if value is not None and not isinstance(value, str):
            raise SettingsError(f"{name} must be a string or None")
        return value

    @classmethod
    def _required_optional_str(cls, values: dict[str, object], name: str) -> str | None:
        if name not in values:
            raise cls._missing_error(name)
        return cls._as_optional_str(values[name], name)
