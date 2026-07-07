import pytest

from core.json_utils import extract_json_block, parse_llm_json
from core.models import ErrorCode, FallbackBlueprint


def test_extract_json_block_from_markdown():
    text = 'Some text\n```json\n{"a": 1}\n```\nmore text'
    assert extract_json_block(text) == '{"a": 1}'


def test_extract_json_block_without_markdown():
    text = 'prefix {"a": 1} suffix'
    assert extract_json_block(text) == '{"a": 1}'


def test_parse_strict_json():
    result = parse_llm_json('{"goals": ["x"]}')
    assert result == {"goals": ["x"]}


def test_parse_json_with_markdown():
    result = parse_llm_json('```json\n{"goals": ["x"]}\n```')
    assert result == {"goals": ["x"]}


def test_parse_repaired_single_quotes():
    result = parse_llm_json("{'goals': ['x'],}")
    assert result == {"goals": ["x"]}


def test_parse_falls_back():
    result = parse_llm_json("not json at all")
    assert isinstance(result, FallbackBlueprint)
    assert result.reason_code == ErrorCode.JSON_PARSE_FAIL


def test_parse_empty_input():
    result = parse_llm_json("")
    assert isinstance(result, FallbackBlueprint)
