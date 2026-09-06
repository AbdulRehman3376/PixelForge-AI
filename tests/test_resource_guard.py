# tests/test_resource_guard.py -- PHASE 14: timeout/cancellation/resize/cleanup

import time

import pytest
from PIL import Image

from core.resource_guard import (
    CancelToken,
    ProcessingTimeout,
    cleanup_memory,
    resize_if_too_large,
    run_with_timeout,
)


# ----- CancelToken -----

def test_cancel_token_starts_uncancelled():
    token = CancelToken()
    assert token.is_cancelled() is False


def test_cancel_token_cancel_and_reset():
    token = CancelToken()
    token.cancel()
    assert token.is_cancelled() is True
    token.reset()
    assert token.is_cancelled() is False


def test_cancel_token_raise_if_cancelled():
    token = CancelToken()
    token.raise_if_cancelled()  # no-op, not cancelled
    token.cancel()
    with pytest.raises(Exception):
        token.raise_if_cancelled()


def test_cancel_token_raise_if_cancelled_custom_exception_type():
    class MyCancelled(Exception):
        pass

    token = CancelToken()
    token.cancel()
    with pytest.raises(MyCancelled):
        token.raise_if_cancelled(MyCancelled)


def test_cancel_token_exposes_raw_event_for_legacy_apis():
    token = CancelToken()
    assert token.event.is_set() is False
    token.cancel()
    assert token.event.is_set() is True


# ----- run_with_timeout -----

def test_run_with_timeout_returns_result_when_fast_enough():
    result = run_with_timeout(lambda a, b: a + b, 2.0, 2, 3)
    assert result == 5


def test_run_with_timeout_raises_processing_timeout_when_too_slow():
    with pytest.raises(ProcessingTimeout):
        run_with_timeout(lambda: time.sleep(1.0), 0.1)


def test_run_with_timeout_propagates_the_original_exception():
    def boom():
        raise ValueError("specific failure")

    with pytest.raises(ValueError, match="specific failure"):
        run_with_timeout(boom, 2.0)


def test_run_with_timeout_passes_kwargs():
    def fn(a, multiplier=1):
        return a * multiplier

    assert run_with_timeout(fn, 2.0, 5, multiplier=3) == 15


# ----- resize_if_too_large -----

def test_resize_if_too_large_downscales_oversized_image():
    img = Image.new("RGB", (8000, 2000))
    resized = resize_if_too_large(img, max_dimension=4000)
    assert max(resized.size) == 4000
    # Aspect ratio preserved.
    assert resized.size[0] / resized.size[1] == pytest.approx(8000 / 2000, rel=0.01)


def test_resize_if_too_large_leaves_small_image_unchanged():
    img = Image.new("RGB", (200, 100))
    result = resize_if_too_large(img, max_dimension=4000)
    assert result.size == (200, 100)
    assert result is img  # no unnecessary copy


def test_resize_if_too_large_handles_portrait_orientation():
    img = Image.new("RGB", (2000, 9000))
    resized = resize_if_too_large(img, max_dimension=4000)
    assert max(resized.size) == 4000


# ----- cleanup_memory -----

def test_cleanup_memory_does_not_raise():
    cleanup_memory("test_context")
    cleanup_memory()  # no context arg