"""Atomic text-file updates shared by client configuration generators."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Replace file content while preserving symlinks, permissions and failed writes."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = target.stat().st_mode & 0o777 if target.exists() else None
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        if existing_mode is not None:
            temp_path.chmod(existing_mode)
        os.replace(temp_path, target)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise
