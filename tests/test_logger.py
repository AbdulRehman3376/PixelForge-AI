# tests/test_logger.py -- PHASE 14: Logging system

import logging

from core import logger as logger_module
from core.logger import get_logger, setup_logging, log_path


def test_setup_logging_creates_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(logger_module, "_LOG_PATH", tmp_path / "logs" / "pixelforge.log")
    monkeypatch.setattr(logger_module, "_configured", False)
    # Remove any handlers a previous test attached to the shared logger.
    root = logging.getLogger("pixelforge")
    root.handlers.clear()

    setup_logging()

    assert (tmp_path / "logs" / "pixelforge.log").exists()


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "_LOG_DIR", tmp_path / "logs2")
    monkeypatch.setattr(logger_module, "_LOG_PATH", tmp_path / "logs2" / "pixelforge.log")
    monkeypatch.setattr(logger_module, "_configured", False)
    root = logging.getLogger("pixelforge")
    root.handlers.clear()

    setup_logging()
    handler_count_after_first = len(root.handlers)
    setup_logging()  # second call must not attach duplicate handlers
    assert len(root.handlers) == handler_count_after_first


def test_get_logger_returns_namespaced_child():
    log = get_logger("core.some_module")
    assert log.name == "pixelforge.core.some_module"


def test_get_logger_without_name_returns_root():
    log = get_logger()
    assert log.name == "pixelforge"


def test_logging_writes_messages_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "_LOG_DIR", tmp_path / "logs3")
    monkeypatch.setattr(logger_module, "_LOG_PATH", tmp_path / "logs3" / "pixelforge.log")
    monkeypatch.setattr(logger_module, "_configured", False)
    root = logging.getLogger("pixelforge")
    root.handlers.clear()

    setup_logging()
    log = get_logger("tests.test_logger")
    log.info("hello from test_logging_writes_messages_to_file")

    content = (tmp_path / "logs3" / "pixelforge.log").read_text(encoding="utf-8")
    assert "hello from test_logging_writes_messages_to_file" in content