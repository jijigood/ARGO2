"""Utilities for extracting raw text from O-RAN specification files."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import zipfile

from pypdf import PdfReader
from docx import Document  # type: ignore


logger = logging.getLogger(__name__)

SUPPORTED_SOURCE_EXTS = {".pdf", ".docx", ".yang", ".zip"}
TEXT_SUFFIX = ".txt"


def _ensure_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _derive_output_path(source_path: Path, output_dir: Path) -> Path:
    stem = source_path.name[: -len(source_path.suffix)] if source_path.suffix else source_path.name
    return output_dir / f"{stem}{TEXT_SUFFIX}"


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to extract page %s from %s: %s", idx, path.name, exc)
            page_text = ""
        if page_text.strip():
            pages.append(page_text)
    return "\n\n".join(pages)


def _extract_docx_text(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n".join(p for p in paragraphs if p)


def _extract_yang_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_zip_text(path: Path) -> str:
    texts: List[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            member_lower = member.lower()
            if member_lower.endswith(".yang") or member_lower.endswith(".txt"):
                with archive.open(member) as file_obj:
                    try:
                        data = file_obj.read().decode("utf-8", errors="ignore")
                    except Exception:
                        data = ""
                if data.strip():
                    texts.append(f"// File: {member}\n{data}")
    return "\n\n".join(texts)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".yang":
        return _extract_yang_text(path)
    if suffix == ".zip":
        return _extract_zip_text(path)
    raise ValueError(f"Unsupported file extension for extraction: {suffix}")


def convert_documents_to_text(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    force: bool = False,
    allow_missing: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Convert supported documents into plain-text files.

    Args:
        source_dir: Directory containing the original specifications.
        output_dir: Location where extracted text files will be stored.
        force: Regenerate text even if the target file already exists.
        allow_missing: Optional sequence of filename suffixes to ignore if extraction fails.
    Returns:
        List of paths to generated text files.
    """
    src_path = Path(source_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    out_path = Path(output_dir)
    _ensure_directory(out_path)

    converted: List[Path] = []
    skipped: List[str] = []
    for entry in sorted(src_path.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in SUPPORTED_SOURCE_EXTS:
            if suffix not in (allow_missing or []):
                logger.debug("Skipping unsupported file %s", entry.name)
            continue

        target_path = _derive_output_path(entry, out_path)
        if target_path.exists() and not force:
            converted.append(target_path)
            continue

        try:
            text = extract_text(entry)
        except Exception as exc:  # pragma: no cover - conversion fallback
            logger.error("Failed to extract %s: %s", entry.name, exc)
            skipped.append(entry.name)
            continue

        if not text.strip():
            logger.warning("No text extracted from %s; skipping", entry.name)
            skipped.append(entry.name)
            continue

        target_path.write_bytes(text.encode("utf-8", errors="ignore"))
        converted.append(target_path)

    if skipped:
        logger.info("Skipped %s files due to extraction issues", len(skipped))
    logger.info("Generated %s text files in %s", len(converted), out_path)
    return converted


def iter_text_files(directory: Path | str) -> Iterable[Path]:
    """Yield text files inside a directory sorted by name."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted(path for path in dir_path.iterdir() if path.is_file() and path.suffix.lower() == TEXT_SUFFIX)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract text from O-RAN specifications")
    parser.add_argument("source_dir", help="Directory containing PDF/DOCX specifications")
    parser.add_argument("output_dir", help="Directory to store extracted text files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted text")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    convert_documents_to_text(args.source_dir, args.output_dir, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
