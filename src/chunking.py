"""Ingestion and metadata-aware chunking for public spending documents."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm"}
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    section: str
    chunk_index: int
    metadata: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def clean_text(text: str, suffix: str) -> str:
    if suffix.lower() in {".html", ".htm"}:
        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
        text = TAG_PATTERN.sub(" ", text)
        text = html.unescape(text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(directory: str | Path) -> list[Document]:
    root = Path(directory)
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"), path.suffix)
        if text:
            documents.append(
                Document(
                    source=str(path),
                    title=path.stem,
                    text=text,
                    metadata={"file_type": path.suffix.lower()},
                )
            )
    return documents


def _fixed_split(text: str, max_chars: int, overlap: int) -> list[str]:
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("max_chars must be positive and overlap must be smaller than max_chars")
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return pieces


def chunk_document(document: Document, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    current_section = "Introduction"
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((current_section, body))

    for line in document.text.splitlines():
        match = HEADER_PATTERN.match(line)
        if match:
            flush()
            current_section = match.group(2).strip()
            lines.clear()
        else:
            lines.append(line)
    flush()

    chunks: list[Chunk] = []
    for section, body in sections:
        section_text = f"{section}\n{body}".strip()
        pieces = [section_text] if len(section_text) <= max_chars else _fixed_split(body, max_chars, overlap)
        for piece in pieces:
            chunks.append(
                Chunk(
                    text=piece,
                    source=document.source,
                    section=section,
                    chunk_index=len(chunks),
                    metadata=document.metadata | {"title": document.title},
                )
            )
    return chunks


def chunk_documents(documents: Iterable[Document], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars, overlap))
    return chunks

