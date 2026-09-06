# tests/test_error_handler.py -- PHASE 14: Friendly error handling

from core.error_handler import (
    is_known_app_error,
    log_and_convert,
    to_friendly_message,
)


class _FakeUpscaleError(Exception):
    """Stands in for the app's own hand-written error classes
    (UpscaleError, FaceRestoreError, VideoStudioError, ...)."""


def test_known_technical_exceptions_get_friendly_message():
    assert "memory" in to_friendly_message(MemoryError()).lower()
    assert "found" in to_friendly_message(FileNotFoundError("x")).lower()
    assert "permission" in to_friendly_message(PermissionError("x")).lower()


def test_app_error_message_passed_through_unchanged():
    exc = _FakeUpscaleError("The image is too large for this model.")
    assert to_friendly_message(exc) == "The image is too large for this model."


def test_app_error_is_known_app_error():
    exc = _FakeUpscaleError("some message")
    assert is_known_app_error(exc) is True


def test_technical_exception_is_not_known_app_error():
    assert is_known_app_error(KeyError("x")) is False
    assert is_known_app_error(PermissionError("x")) is False


def test_unlisted_custom_exception_message_passed_through():
    """Any exception type not in the technical-exceptions map is treated
    as an app-raised error with an already human-readable message
    (matches is_known_app_error's documented design) -- it is passed
    through, not replaced with a generic fallback."""
    class WeirdError(Exception):
        pass

    msg = to_friendly_message(WeirdError("some internal detail"))
    assert msg == "some internal detail"


def test_unlisted_exception_with_empty_message_uses_generic_fallback():
    class WeirdError(Exception):
        pass

    msg = to_friendly_message(WeirdError())
    assert "unexpected" in msg.lower()


def test_os_error_uses_strerror_when_available():
    exc = OSError(28, "No space left on device")
    msg = to_friendly_message(exc)
    assert "disk" in msg.lower() or "file" in msg.lower()


def test_never_raises_on_exotic_exception():
    class Explode(Exception):
        def __str__(self):
            raise RuntimeError("str() itself is broken")

    # to_friendly_message must not itself raise even if str(exc) blows up.
    try:
        to_friendly_message(Explode())
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"to_friendly_message raised unexpectedly: {e}")


def test_log_and_convert_returns_bridge_compatible_shape():
    try:
        raise ValueError("bad value")
    except ValueError as exc:
        result = log_and_convert(exc, context="test_context")
    assert set(result.keys()) == {"ok", "error"}
    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"]


def test_log_and_convert_never_raises_even_with_bad_context():
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = log_and_convert(exc, context=None)
    assert result["ok"] is False