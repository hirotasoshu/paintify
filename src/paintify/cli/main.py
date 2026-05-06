from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from paintify.cli.factory import create_paintify_generator
from paintify.config import PaintifyOptions, SettingsError, SettingsOverrides, SettingsResolver
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
    console.print(
        f"[green]Wrote[/green] {result.output_dir} "
        f"with {len(result.palette)} colors and {len(result.regions)} regions."
    )
