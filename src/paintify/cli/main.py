import json
from pathlib import Path

import typer
from rich.console import Console

from paintify.cli.factory import create_paintify_generator
from paintify.config import (
    PaintifyConfig,
    PaintifyOptions,
    SettingsError,
    SettingsOverrides,
    SettingsResolver,
)
from paintify.pipeline import GenerationResult
from paintify.processing.image import ImageInputError
from paintify.processing.palette import PaletteInputError
from paintify.rendering import ArtifactWriteError

app = typer.Typer(help="Generate paint-by-numbers artifacts from an image.")
console = Console()


@app.command()
def main(
    image: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("paintify-out"), "--output-dir", "-o"),
    difficulty: str = typer.Option(
        "easy", "--difficulty", "-d", help="Preset: easy, medium, hard."
    ),
    no_preset: bool = typer.Option(
        False, "--no-preset", help="Disable presets and require all settings."
    ),
    colors: int | None = typer.Option(
        None, "--colors", "-c", help="Maximum number of paint colors."
    ),
    palette_file: Path | None = typer.Option(
        None, "--palette-file", help="Palette JSON file to snap generated colors to."
    ),
    min_region_size: int | None = typer.Option(
        None, "--min-region-size", help="Minimum target region area in pixels."
    ),
    max_size: int | None = typer.Option(None, "--max-size", help="Largest working image side."),
    smooth_radius: float | None = typer.Option(
        None, "--smooth-radius", help="Gaussian smoothing radius."
    ),
    max_regions: int | None = typer.Option(
        None, "--max-regions", help="Maximum number of output regions."
    ),
    seed: int = typer.Option(0, "--seed", help="Deterministic quantization seed."),
    verbose: bool = typer.Option(False, "--verbose", help="Print diagnostic details."),
    debug_dir: Path | None = typer.Option(
        None, "--debug-dir", help="Write debug diagnostics to this directory."
    ),
) -> None:
    """Generate paint-by-numbers outline, preview, palette, and manifest artifacts."""
    try:
        config = SettingsResolver().resolve(
            PaintifyOptions(
                input_path=image,
                output_dir=output_dir,
                difficulty=difficulty,
                use_preset=not no_preset,
                overrides=SettingsOverrides(
                    max_colors=colors,
                    max_size=max_size,
                    min_region_size=min_region_size,
                    smooth_radius=smooth_radius,
                    palette_file=palette_file,
                    max_regions=max_regions,
                ),
                seed=seed,
            )
        )
    except SettingsError as error:
        raise typer.BadParameter(str(error)) from error

    try:
        result = create_paintify_generator().generate(config)
    except (ImageInputError, PaletteInputError, ArtifactWriteError) as error:
        raise typer.BadParameter(str(error)) from error
    _handle_diagnostics(verbose, debug_dir, config, result)
    console.print(
        f"[green]Wrote[/green] {result.output_dir} "
        f"with {len(result.palette)} colors and {len(result.regions)} regions."
    )


def _handle_diagnostics(
    verbose: bool,
    debug_dir: Path | None,
    config: PaintifyConfig,
    result: GenerationResult,
) -> None:
    if debug_dir is not None:
        try:
            _write_debug_stats(debug_dir, config, result)
        except OSError as error:
            raise typer.BadParameter("could not write debug diagnostics") from error
    if verbose:
        _print_diagnostics(config, result)


def _print_diagnostics(config: PaintifyConfig, result: GenerationResult) -> None:
    console.print("[cyan]Diagnostics[/cyan]")
    console.print(f"image_size: {result.document.image_size[0]}x{result.document.image_size[1]}")
    console.print(f"settings: colors={config.max_colors}, max_regions={config.max_regions}")
    console.print("artifacts: outline.svg, preview.png, palette.json, manifest.json")


def _write_debug_stats(
    debug_dir: Path,
    config: PaintifyConfig,
    result: GenerationResult,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(config.input_path),
        "output_dir": str(result.output_dir),
        "image_size": {
            "width": result.document.image_size[0],
            "height": result.document.image_size[1],
        },
        "settings": {
            "difficulty": config.difficulty,
            "max_colors": config.max_colors,
            "max_size": config.max_size,
            "min_region_size": config.min_region_size,
            "smooth_radius": config.smooth_radius,
            "palette_file": str(config.palette_file) if config.palette_file is not None else None,
            "max_regions": config.max_regions,
            "seed": config.seed,
        },
        "artifacts": ["outline.svg", "preview.png", "palette.json", "manifest.json"],
        "color_count": len(result.palette),
        "region_count": len(result.regions),
    }
    (debug_dir / "debug-stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
