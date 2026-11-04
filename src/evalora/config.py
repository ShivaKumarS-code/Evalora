"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_validator, model_validator

from evalora.schemas import QuestionType


class ConfigError(Exception):
    """Raised when a config file is missing, unreadable, or invalid."""


class DocsConfig(BaseModel):
    paths: list[Path] = Field(min_length=1)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def _default_cases_per_type() -> dict[QuestionType, int]:
    return {
        QuestionType.SINGLE_HOP: 30,
        QuestionType.MULTI_HOP: 15,
        QuestionType.UNANSWERABLE: 15,
        QuestionType.ADVERSARIAL: 15,
    }


class GenerationConfig(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-2.0-flash"
    cases_per_type: dict[QuestionType, int] = Field(default_factory=_default_cases_per_type)
    seed: int = 42

    @field_validator("cases_per_type")
    @classmethod
    def _all_types_present(cls, value: dict[QuestionType, int]) -> dict[QuestionType, int]:
        missing = [
            question_type.value for question_type in QuestionType if question_type not in value
        ]
        if missing:
            raise ValueError(f"cases_per_type is missing: {', '.join(missing)}")
        if any(count < 0 for count in value.values()):
            raise ValueError("cases_per_type counts must be zero or greater")
        return value


class ResponseMapping(BaseModel):
    answer: str = "answer"
    contexts: str = "contexts"


class PipelineConfig(BaseModel):
    endpoint: AnyHttpUrl
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = Field(default=30.0, gt=0)
    concurrency: int = Field(default=4, ge=1)
    match_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    response_mapping: ResponseMapping = Field(default_factory=ResponseMapping)


class JudgeConfig(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0


class ThresholdConfig(BaseModel):
    max_drop: float = Field(ge=0.0, le=1.0)
    min_absolute: float | None = Field(default=None, ge=0.0, le=1.0)


def _default_thresholds() -> dict[str, ThresholdConfig]:
    return {
        "faithfulness": ThresholdConfig(max_drop=0.05),
        "context_recall": ThresholdConfig(max_drop=0.05),
        "correctness": ThresholdConfig(max_drop=0.05),
    }


class GateConfig(BaseModel):
    baseline: str = "main"
    thresholds: dict[str, ThresholdConfig] = Field(default_factory=_default_thresholds)
    ci_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class EvaloraConfig(BaseModel):
    docs: DocsConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    pipeline: PipelineConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    gate: GateConfig = Field(default_factory=GateConfig)


DEFAULT_CONFIG_YAML = """\
docs:
  paths: ["./docs"]
  chunk_size: 800
  chunk_overlap: 100

generation:
  provider: gemini
  model: gemini-2.0-flash
  cases_per_type:
    single_hop: 30
    multi_hop: 15
    unanswerable: 15
    adversarial: 15
  seed: 42

pipeline:
  endpoint: http://localhost:8000/query
  headers: {}
  timeout_s: 30
  concurrency: 4
  match_threshold: 0.8
  response_mapping:
    answer: answer
    contexts: contexts

judge:
  provider: gemini
  model: gemini-2.0-flash
  temperature: 0

gate:
  baseline: main
  thresholds:
    faithfulness: {max_drop: 0.05}
    context_recall: {max_drop: 0.05}
    correctness: {max_drop: 0.05}
  ci_confidence: 0.95
"""


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid config in {path}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> EvaloraConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Could not read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {config_path} must contain a YAML mapping at the top level")
    try:
        return EvaloraConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(config_path, exc)) from exc


def redacted_config_snapshot(
    config: EvaloraConfig, *, redact_headers: bool = True
) -> dict[str, Any]:
    snapshot = config.model_dump(mode="json")
    if redact_headers:
        pipeline = snapshot.get("pipeline")
        if isinstance(pipeline, dict):
            headers = pipeline.get("headers")
            if isinstance(headers, dict):
                pipeline["headers"] = {key: "***" for key in headers}
    return snapshot
