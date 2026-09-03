"""Importing pydicom after its codec plugins were installed in the same session.

The local training data is entirely uncompressed (`1.2.840.10008.1.2.1`), so
nothing here ever needs a codec. The hidden test set may not be, and a Kaggle
notebook therefore installs `python-gdcm` and `pylibjpeg` before inference —
into an interpreter that is already running.

Python caches its directory listings for `sys.path`. A package installed after
those listings were built is invisible until the caches are dropped, so pydicom
finds no decoder plugin and reports:

```text
AssertionError: no decoder for JPEG Lossless P14
```

That message names a missing codec, and during the B52 submission it was
believed. It was wrong: all four decoders were installed and `is_available` was
`True` for every one of them. What was stale was the import machinery.

`importlib.invalidate_caches()` is the fix, and it must happen before the first
pydicom import rather than after. This module owns that ordering so no caller
has to remember it, and does it once per process rather than once per series.

A notebook that installs codecs *after* pydicom has already been imported still
needs `reset()`. That is why it is public.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _invalidate_once() -> bool:
    """Drop the import caches, and remember that it was done.

    `lru_cache` rather than a module-level flag and a lock: it keeps the state
    inside the function, needs nothing built at import time, and a thread race
    at worst invalidates twice, which is idempotent.
    """
    importlib.invalidate_caches()
    return True


def reset() -> None:
    """Force the next `pydicom()` call to drop the import caches again.

    For a notebook that installs codecs after something has already read a
    DICOM. Rare, but the alternative is a run that reports a missing decoder it
    actually has.
    """
    _invalidate_once.cache_clear()


def pydicom() -> Any:
    """Import pydicom, dropping stale path caches on the first call."""
    _invalidate_once()
    import pydicom as module

    return module


def decoder_report() -> dict:
    """Which pixel-data handlers pydicom can actually use, and their versions.

    Printed by a submission before it starts, so "no decoder" is answered by a
    line in the log rather than by a guess. Returns names mapped to
    availability; a handler that fails to import at all is recorded as False
    rather than raising, because reporting is not the place to fall over.
    """
    module = pydicom()
    report: dict[str, Any] = {"pydicom": getattr(module, "__version__", "unknown")}
    for name in ("gdcm", "pylibjpeg", "libjpeg", "openjpeg", "pillow"):
        try:
            report[name] = bool(importlib.import_module(name))
        except Exception:
            report[name] = False

    handlers = getattr(module.config, "pixel_data_handlers", [])
    report["handlers"] = {
        getattr(handler, "HANDLER_NAME", repr(handler)): bool(
            getattr(handler, "is_available", lambda: False)()
        )
        for handler in handlers
    }
    return report
