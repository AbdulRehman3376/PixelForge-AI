# tests/test_database.py -- PHASE 14 unit test coverage: database (SQLite)


def test_create_and_get_project(isolated_database):
    project = isolated_database.create_project("Test Project")
    assert project["name"] == "Test Project"
    assert project["status"] == "active"

    fetched = isolated_database.get_project(project["id"])
    assert fetched["id"] == project["id"]


def test_create_project_with_empty_name_gets_default(isolated_database):
    project = isolated_database.create_project("")
    assert project["name"] == "Untitled Project"


def test_update_project_name(isolated_database):
    project = isolated_database.create_project("Original")
    updated = isolated_database.update_project(project["id"], name="Renamed")
    assert updated["name"] == "Renamed"


def test_update_project_rejects_invalid_status(isolated_database):
    import pytest
    project = isolated_database.create_project("X")
    with pytest.raises(ValueError):
        isolated_database.update_project(project["id"], status="not_a_real_status")


def test_update_project_ignores_unknown_fields(isolated_database):
    project = isolated_database.create_project("X")
    result = isolated_database.update_project(project["id"], totally_made_up_field="ignored")
    assert result["id"] == project["id"]


def test_trash_and_restore_project(isolated_database):
    project = isolated_database.create_project("Trashable")
    trashed = isolated_database.trash_project(project["id"])
    assert trashed["status"] == "trashed"
    restored = isolated_database.restore_project(project["id"])
    assert restored["status"] == "active"


def test_delete_project_hard_delete(isolated_database):
    project = isolated_database.create_project("ToDelete")
    ok = isolated_database.delete_project(project["id"])
    assert ok is True
    assert isolated_database.get_project(project["id"]) is None


def test_list_projects_returns_created_project(isolated_database):
    isolated_database.create_project("Listed Project")
    projects = isolated_database.list_projects()
    names = [p["name"] for p in projects]
    assert "Listed Project" in names


def test_recent_projects_respects_limit(isolated_database):
    for i in range(5):
        isolated_database.create_project(f"Project {i}")
    recent = isolated_database.recent_projects(limit=2)
    assert len(recent) <= 2


def test_add_and_list_media(isolated_database):
    project = isolated_database.create_project("Media Project")
    media = isolated_database.add_media(project["id"], input_path="/fake/path.jpg", media_type="image")
    listed = isolated_database.list_media(project["id"])
    assert any(m["id"] == media["id"] for m in listed)


def test_db_path_points_to_isolated_temp_file(isolated_database, tmp_path):
    assert str(tmp_path) in isolated_database.db_path()


def test_close_is_safe_to_call_when_no_connection_open(isolated_database):
    isolated_database.close()
    isolated_database.close()  # calling twice must not raise