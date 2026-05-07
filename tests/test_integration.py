import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from paintify.cli import app
from paintify.cli.factory import create_paintify_generator
from paintify.config import PaintifyConfig


def _write_rgb_image(path: Path, rgb: np.ndarray) -> None:
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def test_cli_integration_smoke_generates_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "out"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8, :8] = (220, 20, 20)
    image[:8, 8:] = (20, 140, 200)
    image[8:, :] = (240, 230, 120)
    _write_rgb_image(image_path, image)

    result = CliRunner().invoke(
        app,
        [
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--difficulty",
            "easy",
            "--colors",
            "4",
            "--max-size",
            "32",
            "--min-region-size",
            "4",
            "--seed",
            "99",
            "--max-regions",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    for name in ("outline.svg", "preview.png", "palette.json", "manifest.json"):
        assert (output_dir / name).is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    palette = json.loads((output_dir / "palette.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 99
    assert manifest["settings"]["max_regions"] == 2
    assert len(manifest["regions"]) <= 2
    assert len(palette) <= 4
    assert manifest["regions"]


def test_cli_integration_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, :6] = (200, 50, 40)
    image[:, 6:] = (40, 80, 210)
    _write_rgb_image(image_path, image)

    runner = CliRunner()
    outputs = [tmp_path / "out1", tmp_path / "out2"]
    for output in outputs:
        result = runner.invoke(
            app,
            [
                str(image_path),
                "-o",
                str(output),
                "--colors",
                "3",
                "--max-size",
                "20",
                "--seed",
                "11",
            ],
        )
        assert result.exit_code == 0, result.output

    assert (outputs[0] / "manifest.json").read_text(encoding="utf-8") == (
        outputs[1] / "manifest.json"
    ).read_text(encoding="utf-8").replace(str(outputs[1]), str(outputs[0]))
    assert (outputs[0] / "palette.json").read_text(encoding="utf-8") == (
        outputs[1] / "palette.json"
    ).read_text(encoding="utf-8")


def test_hard_difficulty_preserves_palette_file_free_default(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "out"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :5] = (210, 40, 40)
    image[:, 5:] = (40, 80, 210)
    _write_rgb_image(image_path, image)

    result = CliRunner().invoke(
        app, [str(image_path), "-o", str(output_dir), "--difficulty", "hard"]
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["palette_file"] is None


def test_manual_mode_manifest_marks_manual_difficulty(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "out"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :5] = (210, 40, 40)
    image[:, 5:] = (40, 80, 210)
    _write_rgb_image(image_path, image)

    result = CliRunner().invoke(
        app,
        [
            str(image_path),
            "-o",
            str(output_dir),
            "--no-preset",
            "--colors",
            "4",
            "--max-size",
            "64",
            "--min-region-size",
            "3",
            "--smooth-radius",
            "0",
            "--max-regions",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["difficulty"] == "manual"


def test_palette_output_is_compacted_after_region_merge(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "out"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)
    image[4, 4] = (255, 0, 0)
    _write_rgb_image(image_path, image)

    config = PaintifyConfig(
        input_path=image_path,
        output_dir=output_dir,
        difficulty="easy",
        max_colors=2,
        max_size=8,
        min_region_size=2,
        smooth_radius=0,
        palette_file=None,
        max_regions=64,
        seed=0,
    )
    result = create_paintify_generator().generate(config)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    palette = json.loads((output_dir / "palette.json").read_text(encoding="utf-8"))
    assert len(result.palette) == 1
    assert len(palette) == 1
    assert {region["palette_index"] for region in manifest["regions"]} == {1}
    assert isinstance(palette[0]["name"], str)
    assert palette[0]["name"]
    assert isinstance(manifest["palette"][0]["name"], str)
    assert manifest["palette"][0]["name"]
