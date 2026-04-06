from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / ".env"


@dataclass(frozen=True)
class JudgeModelConfig:
    judge_key: str
    ollama_url: str
    model_name: str
    model_path: str
    request_timeout_seconds: int
    temperature: float
    output_reference: str


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_file() -> None:
    if not DOTENV_PATH.exists():
        return

    for raw_line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        os.environ.setdefault(key, value)


@lru_cache(maxsize=1)
def load_project_dotenv() -> None:
    _load_dotenv_file()


def _env(key: str, default: str) -> str:
    load_project_dotenv()
    return os.environ.get(key, default)


def get_judge_model_config(
    judge_key: str,
    *,
    default_model_name: str,
    default_output_reference: str,
) -> JudgeModelConfig:
    prefix = judge_key.upper()
    return JudgeModelConfig(
        judge_key=judge_key,
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434/api/generate"),
        model_name=_env(f"{prefix}_MODEL_NAME", _env("DEFAULT_MODEL_NAME", default_model_name)),
        model_path=_env(f"{prefix}_MODEL_PATH", ""),
        request_timeout_seconds=int(_env("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180")),
        temperature=float(_env("OLLAMA_TEMPERATURE", "0")),
        output_reference=_env(f"{prefix}_OUTPUT_REFERENCE", default_output_reference),
    )
