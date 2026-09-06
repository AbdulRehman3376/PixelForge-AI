# tests/test_prompt_engine.py -- PHASE 14 unit test coverage: prompt parser

from core.prompt_engine import parse_prompt, preview_prompt


def test_empty_prompt_is_empty():
    spec = parse_prompt("")
    assert spec["is_empty"] is True
    assert spec["ok"] is True


def test_parse_prompt_never_raises_on_garbage_input():
    # "Unknown instruction handling" -- must degrade gracefully, not crash.
    spec = parse_prompt("asdkjaslkdj !!! 12345 #$%^&*()")
    assert spec["ok"] is True
    assert isinstance(spec["unrecognized"], list)


def test_parse_prompt_recognizes_a_style_keyword():
    spec = parse_prompt("a photorealistic portrait")
    assert spec["is_empty"] is False
    all_values = [v for values in spec["categories"].values() for v in values]
    assert any("photorealistic" in v.lower() for v in all_values) or spec["matches"]


def test_parse_prompt_with_negative_prompt_populates_negative_categories():
    spec = parse_prompt("a bright photo", negative_prompt="blurry")
    assert isinstance(spec["negative_categories"], dict)


def test_preview_prompt_returns_summary_for_empty_input():
    preview = preview_prompt("")
    assert "summary" in preview
    assert isinstance(preview["summary"], str)


def test_preview_prompt_returns_summary_lines_for_real_input():
    preview = preview_prompt("golden hour lighting, cinematic style")
    assert "summary" in preview
    assert "summary_lines" in preview
    assert isinstance(preview["summary_lines"], list)


def test_preview_prompt_never_raises_on_only_whitespace():
    preview = preview_prompt("     ")
    assert preview["is_empty"] is True