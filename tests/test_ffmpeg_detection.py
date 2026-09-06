# tests/test_ffmpeg_detection.py -- PHASE 14 unit test coverage: FFmpeg detection

from core.video_studio import ffmpeg_status


def test_ffmpeg_status_returns_expected_shape():
    status = ffmpeg_status()
    assert set(status.keys()) == {"available", "version", "path"}
    assert isinstance(status["available"], bool)


def test_ffmpeg_status_when_unavailable_has_no_path(monkeypatch):
    import core.video_studio as vs
    monkeypatch.setattr(vs.shutil, "which", lambda name: None)
    status = ffmpeg_status()
    assert status["available"] is False
    assert status["path"] is None
    assert status["version"] == ""


def test_ffmpeg_status_never_raises_if_binary_is_broken(monkeypatch):
    import core.video_studio as vs

    def _boom(*args, **kwargs):
        raise OSError("simulated broken ffmpeg binary")

    monkeypatch.setattr(vs.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(vs.subprocess, "run", _boom)
    status = ffmpeg_status()
    # Must degrade gracefully (still reports available+path, version unknown)
    # rather than raising and crashing the Settings page.
    assert status["available"] is True
    assert status["version"] == "unknown"