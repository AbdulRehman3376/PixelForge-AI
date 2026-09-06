# tests/test_batch_processor.py -- PHASE 14 unit test coverage: batch queue

from core.batch_processor import BatchController


def test_new_controller_starts_with_empty_queue():
    controller = BatchController()
    assert controller.items == []
    assert controller._running is False


def test_add_files_skips_nonexistent_paths():
    controller = BatchController()
    result = controller.add_files(["/this/path/does/not/exist.jpg"])
    assert result["added"] == 0
    assert result["skipped"] == 1


def test_add_files_adds_valid_image(sample_image_path):
    controller = BatchController()
    result = controller.add_files([sample_image_path])
    assert result["added"] == 1
    assert len(controller.items) == 1


def test_add_files_skips_duplicate_path(sample_image_path):
    controller = BatchController()
    controller.add_files([sample_image_path])
    result = controller.add_files([sample_image_path])
    assert result["added"] == 0
    assert result["skipped"] == 1
    assert len(controller.items) == 1


def test_remove_item(sample_image_path):
    controller = BatchController()
    controller.add_files([sample_image_path])
    item_id = controller.items[0].id
    removed = controller.remove_item(item_id)
    assert removed is True
    assert controller.items == []


def test_remove_item_unknown_id_returns_false():
    controller = BatchController()
    assert controller.remove_item("not_a_real_id") is False


def test_set_included_toggles_flag(sample_image_path):
    controller = BatchController()
    controller.add_files([sample_image_path])
    item_id = controller.items[0].id
    ok = controller.set_included(item_id, False)
    assert ok is True
    assert controller.items[0].included is False


def test_clear_empties_the_queue(sample_image_path):
    controller = BatchController()
    controller.add_files([sample_image_path])
    controller.clear()
    assert controller.items == []
    assert controller.log == []


def test_clear_raises_if_batch_is_running(sample_image_path):
    import pytest
    controller = BatchController()
    controller.add_files([sample_image_path])
    controller._running = True
    with pytest.raises(RuntimeError):
        controller.clear()


def test_add_folder_missing_folder_returns_error():
    controller = BatchController()
    result = controller.add_folder("/this/folder/does/not/exist")
    assert result["added"] == 0
    assert "error" in result