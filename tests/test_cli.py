from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from paintify import cli
from paintify.config import SettingsResolver

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _write_rgb_image(path: Path, rgb: np.ndarray) -> None:
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _plain_output(output: str) -> str:
    return ANSI_ESCAPE.sub("", output)


def test_cli_help_lists_core_options() -> None:
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    output = _plain_output(result.output)
    assert "Generate paint-by-numbers" in output
    assert "--difficulty" in output
    assert "--colors" in output
    assert "--smooth-radius" in output
    assert "--max-regions" in output
    assert "--palette-file" in output
    assert "--no-preset" in output
    assert "--starter-palette" not in output


def test_preset_names_are_stable() -> None:
    assert SettingsResolver.preset_names() == ["easy", "hard", "medium"]


def test_cli_exports_only_app() -> None:
    assert cli.__all__ == ["app"]
    assert not hasattr(cli, "preset_names")


def test_cli_no_preset_reports_missing_manual_values(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image = np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8)
    _write_rgb_image(image_path, image)

    result = CliRunner().invoke(cli.app, [str(image_path), "--no-preset"])

    assert result.exit_code != 0
    assert "max_colors is required when --no-preset is used" in _plain_output(result.output)


def test_cli_reports_clean_error_for_non_image_input(tmp_path: Path) -> None:
    image_path = tmp_path / "not-image.txt"
    image_path.write_text("definitely not an image", encoding="utf-8")

    result = CliRunner().invoke(cli.app, [str(image_path)])

    assert result.exit_code != 0
    assert "is not a readable image file" in result.output
    assert "UnidentifiedImageError" not in result.output


def test_cli_rejects_removed_starter_palette_option(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
    _write_rgb_image(image_path, image)

    result = CliRunner().invoke(cli.app, [str(image_path), "--starter-palette", "bogus"])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "Traceback" not in result.output


def test_cli_reports_clean_error_when_output_path_is_file(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "out-file"
    image = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
    _write_rgb_image(image_path, image)
    output_path.write_text("not a directory", encoding="utf-8")

    result = CliRunner().invoke(cli.app, [str(image_path), "--output-dir", str(output_path)])

    assert result.exit_code != 0
    assert "could not write output artifacts" in result.output
    assert "Traceback" not in result.output
