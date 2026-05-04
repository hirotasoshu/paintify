from __future__ import annotations

from pathlib import Path

import pytest

from paintify.config import PaintifyConfig, PaintifyOptions, SettingsOverrides, SettingsResolver


def test_presets_do_not_force_basic_starter_palette() -> None:
    resolver = SettingsResolver()

    for difficulty in ("easy", "medium", "hard"):
        resolved = resolver.resolve(
            PaintifyOptions(
                input_path=Path("in.png"), output_dir=Path("out"), difficulty=difficulty
            )
        )

        assert resolved.starter_palette is None


def test_presets_shift_detail_upward_for_better_default_results() -> None:
    resolver = SettingsResolver()
    easy = resolver.resolve(PaintifyOptions(Path("in.png"), Path("out"), difficulty="easy"))
    medium = resolver.resolve(PaintifyOptions(Path("in.png"), Path("out"), difficulty="medium"))
    hard = resolver.resolve(PaintifyOptions(Path("in.png"), Path("out"), difficulty="hard"))

    assert (easy.max_colors, medium.max_colors, hard.max_colors) == (15, 20, 30)
    assert (easy.max_size, medium.max_size, hard.max_size) == (768, 1024, 1280)
    assert (easy.min_region_size, medium.min_region_size, hard.min_region_size) == (40, 20, 12)
    assert (easy.max_regions, medium.max_regions, hard.max_regions) == (300, 700, 1200)
    assert easy.max_size < medium.max_size < hard.max_size
    assert easy.min_region_size > medium.min_region_size > hard.min_region_size
    assert easy.max_regions is not None
    assert medium.max_regions is not None
    assert hard.max_regions is not None
    assert easy.max_regions < medium.max_regions < hard.max_regions


def test_apply_preset_and_overrides() -> None:
    config = PaintifyConfig(Path("in.png"), Path("out"), difficulty="medium")
    applied = SettingsResolver.apply_preset(
        config,
        overrides={"max_colors": 7, "min_region_size": 12, "max_regions": 9},
    )

    assert applied.max_colors == 7
    assert applied.max_size == 1024
    assert applied.min_region_size == 12
    assert applied.max_regions == 9


def test_settings_resolver_applies_preset_with_explicit_overrides() -> None:
    resolved = SettingsResolver().resolve(
        PaintifyOptions(
            input_path=Path("in.png"),
            output_dir=Path("out"),
            difficulty="easy",
            overrides=SettingsOverrides(
                max_colors=4,
                max_size=64,
                min_region_size=8,
                smooth_radius=0.25,
                starter_palette=None,
                max_regions=12,
            ),
        )
    )

    assert resolved.max_colors == 4
    assert resolved.max_size == 64
    assert resolved.min_region_size == 8
    assert resolved.smooth_radius == 0.25
    assert resolved.starter_palette is None
    assert resolved.max_regions == 12


def test_no_preset_requires_explicit_manual_values() -> None:
    with pytest.raises(ValueError, match="max_colors is required when --no-preset is used"):
        SettingsResolver().resolve(
            PaintifyOptions(input_path=Path("in.png"), output_dir=Path("out"), use_preset=False)
        )


def test_no_preset_accepts_complete_manual_values() -> None:
    resolved = SettingsResolver().resolve(
        PaintifyOptions(
            input_path=Path("in.png"),
            output_dir=Path("out"),
            use_preset=False,
            overrides=SettingsOverrides(
                max_colors=5,
                max_size=80,
                min_region_size=3,
                smooth_radius=0,
                starter_palette=None,
                max_regions=20,
            ),
        )
    )

    assert resolved.difficulty == "manual"
    assert resolved.max_colors == 5
    assert resolved.max_size == 80
    assert resolved.min_region_size == 3
    assert resolved.smooth_radius == 0
    assert resolved.starter_palette is None
    assert resolved.max_regions == 20


def test_config_validation_rejects_too_few_colors() -> None:
    config = PaintifyConfig(Path("in.png"), Path("out"), max_colors=1)

    with pytest.raises(ValueError, match="max_colors"):
        config.validated()


def test_config_validation_rejects_invalid_max_regions() -> None:
    config = PaintifyConfig(Path("in.png"), Path("out"), max_regions=0)

    with pytest.raises(ValueError, match="max_regions"):
        config.validated()


def test_unknown_preset_is_rejected() -> None:
    config = PaintifyConfig(Path("in.png"), Path("out"), difficulty="expert")

    with pytest.raises(ValueError, match="unknown difficulty"):
        SettingsResolver.apply_preset(config)
