"""Upload validation: allowlist, magic bytes, and macro/executable/zip-bomb/encryption checks."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from core.config import Settings, get_settings
from core.schemas.security import AllowedFileKind, UploadValidationResult
from core.security.exceptions import (
    DisallowedFileTypeError,
    FileSizeSecurityError,
    MacroOrExecutableFileError,
    MagicByteMismatchError,
    PasswordProtectedFileError,
    RowLimitSecurityError,
    UploadSecurityError,
    ZipBombError,
)

ALLOWED_EXTENSIONS: dict[str, AllowedFileKind] = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".xls": "xls",
}

_MACRO_OR_EXECUTABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".xlsm",
        ".xlsb",
        ".xltm",
        ".xlam",
        ".docm",
        ".pptm",
        ".exe",
        ".dll",
        ".bat",
        ".cmd",
        ".ps1",
        ".sh",
        ".msi",
        ".vbs",
        ".js",
        ".jar",
        ".com",
        ".scr",
        ".so",
        ".dylib",
    }
)

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK\x03\x04"

_EXECUTABLE_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"MZ", "Windows PE/MZ executable"),
    (b"\x7fELF", "ELF executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit executable"),
    (b"\xce\xfa\xed\xfe", "Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit executable"),
    (b"\xca\xfe\xba\xbe", "Java class / Mach-O fat binary"),
    (b"#!", "shell script shebang"),
)

_BINARY_DISGUISED_AS_CSV_MAGICS: tuple[tuple[bytes, str], ...] = (
    (_ZIP_MAGIC, "ZIP/OOXML archive"),
    (_OLE2_MAGIC, "OLE2 binary compound document"),
    (b"%PDF", "PDF document"),
    (b"\x1f\x8b", "GZIP archive"),
    (b"Rar!", "RAR archive"),
    (b"7z\xbc\xaf\x27\x1c", "7z archive"),
    (b"\x89PNG", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
)

_OLE2_ENCRYPTION_MARKERS: tuple[bytes, ...] = (
    "EncryptedPackage".encode("utf-16le"),
    "EncryptionInfo".encode("utf-16le"),
    "StrongEncryptionSpace".encode("utf-16le"),
    b"EncryptedPackage",
    b"EncryptionInfo",
)

_OLE2_MACRO_MARKERS: tuple[bytes, ...] = (
    "_VBA_PROJECT_CUR".encode("utf-16le"),
    "VBA".encode("utf-16le") + b"\x00\x00",
    b"_VBA_PROJECT_CUR",
    b"vbaProject.bin",
)

_DEFAULT_MAX_ZIP_ENTRIES = 2_000
_DEFAULT_MAX_COMPRESSION_RATIO = 100.0
_RATIO_CHECK_MIN_UNCOMPRESSED_BYTES = 1 * 1024 * 1024


def _check_executable_magic(header: bytes, filename: str) -> None:
    """Reject any file carrying executable magic bytes regardless of extension."""
    for magic, label in _EXECUTABLE_MAGICS:
        if header.startswith(magic):
            raise MacroOrExecutableFileError(
                f"File '{filename}' was rejected because it has a {label} signature."
            )


def _has_zip_path_traversal(member_name: str) -> bool:
    """Return True if a ZIP entry path attempts directory traversal (zip-slip)."""
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        return True
    parts = PurePosixPath(normalized).parts
    return ".." in parts


def _validate_csv_content(file_path: Path, header: bytes) -> list[str]:
    """Validate that a CSV upload contains text rather than disguised binary content."""
    _check_executable_magic(header, file_path.name)
    for magic, label in _BINARY_DISGUISED_AS_CSV_MAGICS:
        if header.startswith(magic):
            raise MagicByteMismatchError(
                f"File '{file_path.name}' declares '.csv' but starts with a {label} signature."
            )

    has_utf16_bom = header.startswith((b"\xff\xfe", b"\xfe\xff"))
    if not has_utf16_bom and b"\x00" in header:
        raise MagicByteMismatchError(
            f"File '{file_path.name}' contains binary NUL bytes incompatible with CSV."
        )

    return ["extension_allowlist", "size_limit", "csv_magic_text_check"]


def _validate_xlsx_content(
    file_path: Path,
    header: bytes,
    *,
    max_uncompressed_bytes: int,
    max_zip_entries: int,
    max_compression_ratio: float,
) -> list[str]:
    """Validate an OOXML ``.xlsx`` upload against magic, encryption, macro, and zip-bomb checks."""
    _check_executable_magic(header, file_path.name)

    # Password-protected .xlsx files are wrapped in an OLE2 compound file container.
    if header.startswith(_OLE2_MAGIC):
        sample = file_path.read_bytes()[:262_144]
        if any(marker in sample for marker in _OLE2_ENCRYPTION_MARKERS):
            raise PasswordProtectedFileError(
                f"File '{file_path.name}' is password-protected/encrypted. "
                "Please remove password protection and re-upload."
            )
        raise MagicByteMismatchError(
            f"File '{file_path.name}' declares '.xlsx' but has an OLE2 binary signature."
        )

    if not header.startswith(_ZIP_MAGIC):
        raise MagicByteMismatchError(
            f"File '{file_path.name}' declares '.xlsx' but does not have a valid ZIP signature."
        )

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > max_zip_entries:
                raise ZipBombError(
                    f"File '{file_path.name}' contains {len(infos)} ZIP entries, "
                    f"exceeding the maximum allowed ({max_zip_entries})."
                )

            total_uncompressed = 0
            total_compressed = 0
            entry_names: set[str] = set()

            for info in infos:
                if _has_zip_path_traversal(info.filename):
                    raise ZipBombError(
                        f"File '{file_path.name}' contains unsafe ZIP member path "
                        f"{info.filename!r}."
                    )

                if info.flag_bits & 0x1:
                    raise PasswordProtectedFileError(
                        f"File '{file_path.name}' contains encrypted archive entries. "
                        "Please remove password protection and re-upload."
                    )

                lower_name = info.filename.lower()
                if (
                    "vbaproject" in lower_name
                    or lower_name.startswith("xl/macrosheets/")
                    or (lower_name.startswith("xl/") and lower_name.endswith(".bin"))
                ):
                    raise MacroOrExecutableFileError(
                        f"File '{file_path.name}' contains embedded macro payload "
                        f"('{info.filename}') and was rejected."
                    )

                total_uncompressed += info.file_size
                total_compressed += info.compress_size
                entry_names.add(info.filename)

                if total_uncompressed > max_uncompressed_bytes:
                    raise ZipBombError(
                        f"File '{file_path.name}' expands to {total_uncompressed:,} bytes, "
                        f"exceeding the safe limit of {max_uncompressed_bytes:,} bytes (zip bomb)."
                    )

            if total_uncompressed >= _RATIO_CHECK_MIN_UNCOMPRESSED_BYTES:
                ratio = total_uncompressed / max(total_compressed, 1)
                if ratio > max_compression_ratio:
                    raise ZipBombError(
                        f"File '{file_path.name}' has suspicious compression ratio "
                        f"({ratio:.1f}x > {max_compression_ratio:.1f}x), rejected as a zip bomb."
                    )

            if "[Content_Types].xml" not in entry_names:
                raise MagicByteMismatchError(
                    f"File '{file_path.name}' is a ZIP archive missing OOXML [Content_Types].xml."
                )

            content_types = zf.read("[Content_Types].xml")[:65_536].decode(
                "utf-8", errors="ignore"
            )
            lowered_ct = content_types.lower()
            if "macroenabled" in lowered_ct or "vbaproject" in lowered_ct:
                raise MacroOrExecutableFileError(
                    f"File '{file_path.name}' declares macro-enabled OOXML content "
                    "and was rejected."
                )
    except zipfile.BadZipFile as exc:
        raise MagicByteMismatchError(
            f"File '{file_path.name}' is corrupted or not a valid '.xlsx' ZIP archive."
        ) from exc

    return [
        "extension_allowlist",
        "size_limit",
        "xlsx_zip_ooxml_magic",
        "no_encryption",
        "no_macros",
        "zip_bomb_guard",
    ]


def _validate_xls_content(file_path: Path, header: bytes) -> list[str]:
    """Validate a legacy OLE2 ``.xls`` upload for magic bytes, encryption, and macros."""
    _check_executable_magic(header, file_path.name)

    if not header.startswith(_OLE2_MAGIC):
        raise MagicByteMismatchError(
            f"File '{file_path.name}' declares '.xls' but does not have an OLE2 signature."
        )

    sample = file_path.read_bytes()[:524_288]
    if any(marker in sample for marker in _OLE2_ENCRYPTION_MARKERS):
        raise PasswordProtectedFileError(
            f"File '{file_path.name}' is password-protected/encrypted. "
            "Please remove password protection and re-upload."
        )

    # Check BIFF8 FILEPASS record (0x002F) immediately near BOF (0x0809) if present.
    bof_idx = sample.find(b"\x09\x08\x10\x00")
    if bof_idx != -1:
        biff_window = sample[bof_idx : bof_idx + 256]
        if b"\x2f\x00" in biff_window:
            raise PasswordProtectedFileError(
                f"File '{file_path.name}' appears to be password-protected (BIFF FILEPASS). "
                "Please remove password protection and re-upload."
            )

    if any(marker in sample for marker in _OLE2_MACRO_MARKERS):
        raise MacroOrExecutableFileError(
            f"File '{file_path.name}' contains embedded VBA macros and was rejected."
        )

    return [
        "extension_allowlist",
        "size_limit",
        "xls_ole2_magic",
        "no_encryption",
        "no_macros",
    ]


def validate_upload(
    file_path: Path | str,
    *,
    settings: Settings | None = None,
    max_uncompressed_bytes: int | None = None,
    max_zip_entries: int = _DEFAULT_MAX_ZIP_ENTRIES,
    max_compression_ratio: float = _DEFAULT_MAX_COMPRESSION_RATIO,
) -> UploadValidationResult:
    """Validate an uploaded file before ingestion touches or copies it.

    Checks extension allowlist (``csv``, ``xlsx``, ``xls``), rejects macro-enabled
    and executable files, enforces configured upload size limits, verifies magic bytes,
    and defends against password-protected spreadsheets and ``.xlsx`` zip bombs.
    """
    resolved = Path(file_path)
    if not resolved.is_file():
        raise UploadSecurityError(f"Upload path does not exist or is not a file: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix in _MACRO_OR_EXECUTABLE_EXTENSIONS:
        raise MacroOrExecutableFileError(
            f"File '{resolved.name}' has disallowed macro/executable extension '{suffix}'."
        )
    if suffix not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DisallowedFileTypeError(
            f"File extension '{suffix or '<none>'}' is not permitted. "
            f"Allowed extensions: {allowed_list}."
        )

    cfg = settings or get_settings()
    size_bytes = resolved.stat().st_size
    if size_bytes > cfg.max_upload_bytes:
        raise FileSizeSecurityError(
            f"File '{resolved.name}' is {size_bytes:,} bytes, exceeding the maximum "
            f"upload limit of {cfg.max_upload_bytes:,} bytes ({cfg.max_upload_mb} MB)."
        )

    with resolved.open("rb") as handle:
        header = handle.read(8192)

    file_kind = ALLOWED_EXTENSIONS[suffix]
    if file_kind == "csv":
        checks = _validate_csv_content(resolved, header)
    elif file_kind == "xlsx":
        uncompressed_cap = (
            max_uncompressed_bytes
            if max_uncompressed_bytes is not None
            else max(cfg.max_upload_bytes * 10, 200 * 1024 * 1024)
        )
        checks = _validate_xlsx_content(
            resolved,
            header,
            max_uncompressed_bytes=uncompressed_cap,
            max_zip_entries=max_zip_entries,
            max_compression_ratio=max_compression_ratio,
        )
    else:
        checks = _validate_xls_content(resolved, header)

    return UploadValidationResult(
        filename=resolved.name,
        file_kind=file_kind,
        size_bytes=size_bytes,
        is_valid=True,
        checks_passed=checks,
        warnings=[],
    )


def validate_row_count(row_count: int, *, max_rows: int, context: str = "dataset") -> None:
    """Enforce the configured row limit with a clear security exception."""
    if row_count > max_rows:
        raise RowLimitSecurityError(
            f"Row count ({row_count:,}) for {context} exceeds the configured limit "
            f"of {max_rows:,} rows."
        )
