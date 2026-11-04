from __future__ import annotations

from pathlib import Path

import pytest

from evalora.config import DEFAULT_CONFIG_YAML, ConfigError, load_config


def test_starter_config_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "evalora.yaml"
    path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    config = load_config(path)
    assert config.generation.seed == 42
    assert config.gate.ci_confidence == 0.95
    assert config.pipeline.response_mapping.answer == "answer"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_overlap_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        DEFAULT_CONFIG_YAML.replace("chunk_overlap: 100", "chunk_overlap: 900"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_config(path)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("docs: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_missing_question_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        DEFAULT_CONFIG_YAML.replace("    adversarial: 15\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="adversarial"):
        load_config(path)


def test_error_includes_field_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        DEFAULT_CONFIG_YAML.replace("  concurrency: 4", "  concurrency: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"pipeline\.concurrency"):
        load_config(path)


def test_groq_provider_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        DEFAULT_CONFIG_YAML.replace("provider: gemini", "provider: groq"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="provider"):
        load_config(path)
