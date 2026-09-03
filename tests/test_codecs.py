"""Importing pydicom after codecs were installed into a running interpreter.

This exists because of one wasted debugging session on a live submission. Four
decoders were installed and every one reported `is_available` as True, while
pydicom insisted it had no decoder for JPEG Lossless P14. Nothing was missing;
the import machinery's path caches were stale.

The ordering that fixes it — invalidate, then import — is easy to write the
wrong way round and impossible to notice locally, because the training data is
uncompressed and never needs a codec at all.
"""

from __future__ import annotations

import importlib

import pytest

from rsna_knee.imaging import codecs


@pytest.fixture(autouse=True)
def fresh():
    codecs.reset()
    yield
    codecs.reset()


def test_the_caches_are_dropped_before_the_first_import():
    """Invalidating after the import would be a no-op, and looks identical."""
    import inspect

    source = inspect.getsource(codecs.pydicom)
    assert source.index("_invalidate_once()") < source.index("import pydicom as module")


def test_the_first_call_drops_the_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: calls.append(1))

    codecs.pydicom()
    assert calls == [1]


def test_the_caches_are_dropped_once_not_per_call(monkeypatch):
    """It is called per series otherwise, and a full run reads tens of thousands."""
    calls = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: calls.append(1))

    codecs.pydicom()
    codecs.pydicom()
    codecs.pydicom()

    assert len(calls) == 1


def test_reset_makes_the_next_call_drop_them_again(monkeypatch):
    """For a notebook that installs codecs after something already read a DICOM."""
    calls = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: calls.append(1))

    codecs.pydicom()
    codecs.reset()
    codecs.pydicom()

    assert len(calls) == 2


def test_it_returns_the_real_module():
    assert codecs.pydicom().__name__ == "pydicom"


def test_the_report_names_pydicom_and_the_handlers():
    report = codecs.decoder_report()

    assert "pydicom" in report
    assert isinstance(report["handlers"], dict)
    assert all(isinstance(value, bool) for value in report["handlers"].values())


def test_a_codec_that_will_not_import_is_recorded_not_raised(monkeypatch):
    """Reporting is not the place to fall over."""
    real = importlib.import_module

    def refuse(name, *args, **kwargs):
        if name == "gdcm":
            raise ImportError("boom")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", refuse)
    assert codecs.decoder_report()["gdcm"] is False


def test_the_readers_route_through_the_helper():
    """A direct `import pydicom` anywhere else reintroduces the bug silently."""
    from rsna_knee.imaging import dicom_io, dicom_metadata

    for module in (dicom_io, dicom_metadata):
        source = __import__("inspect").getsource(module)
        assert "from .codecs import pydicom as _pydicom" in source
        assert "\n    import pydicom\n" not in source
