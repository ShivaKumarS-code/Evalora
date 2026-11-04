from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evalora.cli import app
from evalora.config import DEFAULT_CONFIG_YAML, load_config

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == "evalora 0.1.0"


def test_no_cache_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--no-cache", "version"])
    assert result.exit_code == 0
    assert result.output.strip() == "evalora 0.1.0"


def test_init_writes_valid_config(tmp_path: Path) -> None:
    target = tmp_path / "evalora.yaml"
    result = runner.invoke(app, ["init", "--path", str(target)])
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == DEFAULT_CONFIG_YAML
    config = load_config(target)
    assert config.pipeline.concurrency == 4


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "evalora.yaml"
    target.write_text("sentinel", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(target)])
    assert result.exit_code == 1
    assert target.read_text(encoding="utf-8") == "sentinel"


def test_init_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "evalora.yaml"
    target.write_text("sentinel", encoding="utf-8")
    result = runner.invoke(app, ["init", "--path", str(target), "--force"])
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == DEFAULT_CONFIG_YAML
