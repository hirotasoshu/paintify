from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paintify.pipeline import PaintByNumbersDocument


@dataclass(frozen=True)
class OutputArtifact:
    name: str
    payload: str | bytes


@dataclass(frozen=True)
class OutputBundle:
    document: PaintByNumbersDocument
    artifacts: list[OutputArtifact]
