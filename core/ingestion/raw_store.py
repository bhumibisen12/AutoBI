"""Immutable raw artifact persistence and manifest serialization."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path

from core.ingestion.exceptions import ImmutableRawError, IngestionError, UploadSizeLimitError
from core.schemas.ingestion import RawManifest

_HASH_CHUNK_BYTES = 1024 * 1024
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def compute_sha256(file_path: Path) -> str:
    """Compute the lowercase hex SHA-256 checksum of a file via streaming reads."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Strip directory components and unsafe characters from an uploaded filename."""
    basename = Path(filename).name
    cleaned = _SAFE_FILENAME_RE.sub("_", basename).strip("._")
    if not cleaned:
        raise IngestionError(f"Invalid filename after sanitization: {filename!r}")
    return cleaned


def make_read_only(file_path: Path) -> None:
    """Mark a raw artifact read-only to protect immutability."""
    os.chmod(file_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def save_immutable_raw_copy(
    source_path: Path,
    raw_batch_dir: Path,
    *,
    max_upload_bytes: int,
    target_filename: str | None = None,
) -> tuple[Path, int, str]:
    """Copy a source file into ``data/raw/<batch_id>/`` before any parsing occurs.

    Enforces ``max_upload_bytes``, prevents overwriting existing raw artifacts,
    verifies SHA-256 integrity after copy, and marks the copied file read-only.

    Returns:
        Tuple of ``(raw_copy_path, size_bytes, sha256_hex)``.
    """
    resolved_source = Path(source_path)
    if not resolved_source.is_file():
        raise IngestionError(f"Source file does not exist or is not a file: {resolved_source}")

    size_bytes = resolved_source.stat().st_size
    if size_bytes > max_upload_bytes:
        raise UploadSizeLimitError(
            f"File '{resolved_source.name}' is {size_bytes:,} bytes, exceeding the "
            f"configured limit of {max_upload_bytes:,} bytes."
        )

    safe_name = sanitize_filename(target_filename or resolved_source.name)
    raw_batch_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_batch_dir / safe_name

    if dest_path.exists():
        raise ImmutableRawError(
            f"Immutable raw artifact already exists and cannot be overwritten: {dest_path}"
        )

    source_sha256 = compute_sha256(resolved_source)
    shutil.copy2(resolved_source, dest_path)
    copied_sha256 = compute_sha256(dest_path)

    if copied_sha256 != source_sha256:
        dest_path.unlink(missing_ok=True)
        raise IngestionError(
            f"Checksum mismatch after copying '{resolved_source.name}' to raw layer."
        )

    make_read_only(dest_path)
    return dest_path, size_bytes, copied_sha256


def write_manifest(manifest: RawManifest, raw_batch_dir: Path) -> Path:
    """Serialize ``manifest.json`` into the raw batch directory."""
    raw_batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_batch_dir / "manifest.json"
    if manifest_path.exists():
        raise ImmutableRawError(f"Manifest already exists and is immutable: {manifest_path}")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    make_read_only(manifest_path)
    return manifest_path

