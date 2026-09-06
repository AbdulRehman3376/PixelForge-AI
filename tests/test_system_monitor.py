# tests/test_system_monitor.py -- PHASE 14: CPU/RAM/disk monitoring

from core.system_monitor import (
    check_disk_space,
    check_disk_writable,
    get_cpu_percent,
    get_ram_status,
    get_system_snapshot,
    psutil_available,
)


def test_get_cpu_percent_returns_number_or_unknown():
    value = get_cpu_percent(interval=0.01)
    assert value == -1.0 or 0.0 <= value <= 100.0


def test_get_ram_status_shape():
    status = get_ram_status()
    assert set(status.keys()) == {
        "available", "total_gb", "used_gb", "available_gb", "percent_used",
    }
    if status["available"]:
        assert status["total_gb"] >= status["used_gb"] >= 0


def test_check_disk_space_on_valid_existing_path(tmp_path):
    result = check_disk_space(str(tmp_path), required_mb=1)
    assert result["ok"] is True
    assert result["free_mb"] > 0


def test_check_disk_space_unreasonable_requirement_fails(tmp_path):
    result = check_disk_space(str(tmp_path), required_mb=10 ** 12)  # 1 exabyte
    assert result["ok"] is False


def test_check_disk_space_walks_up_to_existing_parent(tmp_path):
    missing = tmp_path / "does" / "not" / "exist" / "yet"
    result = check_disk_space(str(missing), required_mb=1)
    # Should have walked up to tmp_path (or an ancestor) rather than raising.
    assert result["ok"] is True


def test_check_disk_writable_on_writable_folder(tmp_path):
    result = check_disk_writable(str(tmp_path))
    assert result["ok"] is True
    assert result["error"] is None


def test_check_disk_writable_creates_missing_folder(tmp_path):
    target = tmp_path / "new_output_folder"
    assert not target.exists()
    result = check_disk_writable(str(target))
    assert result["ok"] is True
    assert target.exists()


def test_get_system_snapshot_shape(tmp_path):
    snapshot = get_system_snapshot(str(tmp_path))
    assert set(snapshot.keys()) == {"cpu_percent", "ram", "disk", "psutil_available"}
    assert snapshot["psutil_available"] == psutil_available()