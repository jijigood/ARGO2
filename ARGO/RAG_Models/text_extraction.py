"""Utility helpers for extracting raw O-RAN documents into plain text files."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractionStats:
    """Track text extraction results."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "total": self.processed + self.skipped + self.failed,
        }


class ExtractionError(RuntimeError):
    """Raised when a document cannot be converted to text."""


DEFAULT_EXTENSIONS = (".pdf", ".docx")


def _ensure_dependencies() -> None:
    try:  # noqa: F401
        import pypdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime safeguard
        raise ExtractionError(
            "pypdf is required for PDF extraction. Install via `pip install pypdf`."
        ) from exc

    try:  # noqa: F401
        import docx2txt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError(
            "docx2txt is required for DOCX extraction. Install via `pip install docx2txt`."
        ) from exc


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    texts = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - rare PDF quirks
            logger.warning("Failed to extract page %s from %s: %s", index, path.name, exc)
            text = ""
        texts.append(text)
    return "\n".join(texts).strip()


def _extract_docx(path: Path) -> str:
    import docx2txt

    text = docx2txt.process(str(path)) or ""
    return text.strip()


def _clean_text(text: str) -> str:
    # Collapse multiple blank lines while preserving paragraph boundaries.
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines = []
    blank_streak = 0
    for line in lines:
        if line:
            cleaned_lines.append(line)
            blank_streak = 0
        else:
            blank_streak += 1
            if blank_streak <= 1:
                cleaned_lines.append("")
    return "\n".join(cleaned_lines).strip()


def extract_documents_to_text(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    include_extensions: Optional[Iterable[str]] = None,
    exclude_extensions: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Convert supported documents into UTF-8 text files.

    Args:
        source_dir: Directory that contains the original files (PDF/DOCX).
        output_dir: Destination directory where plain text files will be written.
        force: Re-convert even if the output file already exists.
        include_extensions: Subset of extensions to include. Defaults to PDF & DOCX.
        exclude_extensions: Extensions that should be skipped explicitly.

    Returns:
        Dictionary summarising processed/skipped/failed counts.
    """

    _ensure_dependencies()

    source_path = Path(source_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    extensions = tuple(ext.lower() for ext in (include_extensions or DEFAULT_EXTENSIONS))
    excluded = set(ext.lower() for ext in (exclude_extensions or []))

    stats = ExtractionStats()
    for file_path in sorted(source_path.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in extensions or suffix in excluded:
            stats.skipped += 1
            continue

        relative = file_path.relative_to(source_path)
        destination = (output_path / relative).with_suffix(".txt")

        if destination.exists() and not force:
            stats.skipped += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            if suffix == ".pdf":
                raw_text = _extract_pdf(file_path)
            elif suffix == ".docx":
                raw_text = _extract_docx(file_path)
            else:
                stats.skipped += 1
                continue

            cleaned = _clean_text(raw_text)
            if not cleaned:
                logger.warning("No textual content extracted from %s", file_path.name)
                stats.failed += 1
                continue

            destination.write_text(cleaned, encoding="utf-8")
            stats.processed += 1
        except Exception as exc:  # pragma: no cover - runtime protection
            logger.error("Failed to extract %s: %s", file_path, exc)
            stats.failed += 1

    summary = stats.to_dict()
    logger.info(
        "Extraction summary from %s -> %s: %s", source_path, output_path, summary
    )
    return summary


__all__ = ["extract_documents_to_text", "ExtractionError"]
