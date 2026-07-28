"""Managing the document corpus at runtime.

Ingestion happens once at startup because building the index takes seconds. That is right
for serving, but wrong for a tool whose whole premise is that you keep editing your resume:
a new PDF should not need a server restart. So uploads land in the documents directory and
trigger a rebuild behind a lock.

Writes are gated on ALLOW_DOCUMENT_WRITES. There is no authentication in front of these
endpoints, so a public deployment must set it to false — otherwise anyone who finds the URL
can add or delete files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = frozenset({".pdf", ".txt", ".md"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,120}$")


class DocumentError(ValueError):
    """A rejected upload or delete, with a message safe to show the user."""


@dataclass(frozen=True)
class StoredDocument:
    name: str
    size_bytes: int
    modified: str


def writes_enabled() -> bool:
    raw = os.getenv("ALLOW_DOCUMENT_WRITES")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def safe_filename(raw_name: str) -> str:
    """Reduce an uploaded name to a bare, validated filename.

    Path(...).name strips any directory component, which is what stops
    "../../.env" or an absolute path from escaping the documents directory.
    """
    name = Path(raw_name.replace("\\", "/")).name.strip()
    if not name:
        raise DocumentError("The file has no name.")
    if not _SAFE_NAME_RE.match(name):
        raise DocumentError(
            f"{name!r} is not an acceptable filename. Use letters, numbers, spaces, "
            "dots, underscores or hyphens."
        )
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise DocumentError(f"{name!r} is not a supported file type. Allowed: {allowed}.")
    return name


def resolve_inside(directory: Path, name: str) -> Path:
    """Resolve a validated name inside the directory, refusing anything that escapes it."""
    target = (directory / safe_filename(name)).resolve()
    root = directory.resolve()
    if root != target.parent:
        raise DocumentError("That path is outside the documents directory.")
    return target


def list_documents(directory: Path) -> list[StoredDocument]:
    if not directory.is_dir():
        return []
    stored: list[StoredDocument] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        info = path.stat()
        stored.append(
            StoredDocument(
                name=path.name,
                size_bytes=info.st_size,
                modified=_isoformat(info.st_mtime),
            )
        )
    return stored


def save_document(directory: Path, name: str, payload: bytes) -> StoredDocument:
    """Write one upload, overwriting a file of the same name.

    Overwriting is deliberate: replacing a resume with its newer version is the main thing
    this is for, and uploading "resume.pdf" again should update it rather than accumulate
    "resume (1).pdf" variants that then all get indexed.
    """
    if not payload:
        raise DocumentError(f"{name!r} is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise DocumentError(f"{name!r} is larger than the {limit_mb} MB limit.")

    target = resolve_inside(directory, name)
    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    info = target.stat()
    return StoredDocument(
        name=target.name,
        size_bytes=info.st_size,
        modified=_isoformat(info.st_mtime),
    )


def delete_document(directory: Path, name: str) -> None:
    target = resolve_inside(directory, name)
    if not target.is_file():
        raise DocumentError(f"{name!r} is not in the documents directory.")
    target.unlink()


def _isoformat(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")
