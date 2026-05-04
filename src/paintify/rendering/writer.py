from __future__ import annotations

from pathlib import Path

from paintify.models import OutputBundle


class ArtifactWriteError(RuntimeError):
    pass


class FilesystemArtifactWriter:
    def write(self, output_dir: Path, bundle: OutputBundle) -> None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for artifact in bundle.artifacts:
                path = output_dir / artifact.name
                if isinstance(artifact.payload, bytes):
                    path.write_bytes(artifact.payload)
                else:
                    path.write_text(artifact.payload, encoding="utf-8")
        except OSError as error:
            raise ArtifactWriteError(f"could not write output artifacts to {output_dir}") from error
