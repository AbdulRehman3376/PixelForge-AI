# tests/test_health_check.py -- PHASE 14: Startup health check

from core.health_check import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARNING,
    run_startup_health_check,
)


def test_run_startup_health_check_returns_expected_shape():
    report = run_startup_health_check()
    assert set(report.keys()) == {"overall", "checks", "summary"}
    assert report["overall"] in (STATUS_OK, STATUS_WARNING, STATUS_ERROR)
    assert isinstance(report["checks"], list) and report["checks"]
    assert isinstance(report["summary"], str) and report["summary"]


def test_every_check_has_a_name_and_status():
    report = run_startup_health_check()
    names = {c["name"] for c in report["checks"]}
    # These are the checks this module is documented to run; if one is
    # ever removed/renamed the aggregator's summary math would silently
    # miscount, so pin the expected set here.
    assert names == {
        "ffmpeg", "upscale_model", "face_restore_model", "background_remover",
        "database", "config", "output_folder", "ram",
    }
    for check in report["checks"]:
        assert check["status"] in (STATUS_OK, STATUS_WARNING, STATUS_ERROR)


def test_a_broken_check_does_not_stop_the_others(monkeypatch):
    """A check that raises must be isolated -- every other check still runs
    and the report is still well-formed (graceful degradation)."""
    import core.health_check as hc

    def _boom():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(hc, "_check_ffmpeg", _boom)
    report = run_startup_health_check()

    names = {c["name"] for c in report["checks"]}
    assert "ffmpeg" in names  # still present, just marked as a warning
    ffmpeg_result = next(c for c in report["checks"] if c["name"] == "ffmpeg")
    assert ffmpeg_result["status"] == STATUS_WARNING
    # Every other check still ran and reported something.
    assert len(report["checks"]) == 8


def test_overall_is_ok_when_every_check_passes(monkeypatch):
    import core.health_check as hc

    def _all_ok():
        return {"status": STATUS_OK, "detail": "fine"}

    for name in ("_check_ffmpeg", "_check_upscale_model", "_check_face_restore_model",
                 "_check_bg_remover", "_check_database", "_check_config",
                 "_check_output_folder", "_check_ram"):
        monkeypatch.setattr(hc, name, _all_ok)

    report = run_startup_health_check()
    assert report["overall"] == STATUS_OK
    assert "8/8 OK" in report["summary"]