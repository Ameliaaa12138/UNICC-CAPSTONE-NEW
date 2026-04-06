from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from app.api import app
except ModuleNotFoundError:
    from api import app

try:
    from app.orchestrator import run_pipeline
except ModuleNotFoundError:
    from orchestrator import run_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("_") else f"{prefix}_"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UNICC AI Safety Lab pipeline locally against a JSON submission."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "sample_submission.json"),
        help="Path to the input submission JSON file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for generated artifact filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    prefix = _normalize_prefix(args.output_prefix)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        input_data = json.load(file)

    results = run_pipeline(input_data)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    submission_output_path = OUTPUTS_DIR / f"{prefix}submission.json"
    judge_output_paths = [
        OUTPUTS_DIR / f"{prefix}judge1_output.json",
        OUTPUTS_DIR / f"{prefix}judge2_output.json",
        OUTPUTS_DIR / f"{prefix}judge3_output.json",
    ]
    critique_round_path = OUTPUTS_DIR / f"{prefix}critique_round.json"
    synthesis_output_path = OUTPUTS_DIR / f"{prefix}synthesis_output.json"
    full_result_path = OUTPUTS_DIR / f"{prefix}full_result.json"
    log_path = LOGS_DIR / f"{prefix}pipeline_log.json"

    _write_json(submission_output_path, input_data)
    for path, judge_output in zip(judge_output_paths, results["judge_outputs"], strict=True):
        _write_json(path, judge_output)
    _write_json(critique_round_path, results["critique_round"])
    _write_json(synthesis_output_path, results["synthesis_output"])
    _write_json(full_result_path, results)

    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "submission_id": input_data["submission_id"],
        "status": "completed",
        "source_input": str(input_path),
        "generated_files": [
            submission_output_path.name,
            *(path.name for path in judge_output_paths),
            critique_round_path.name,
            synthesis_output_path.name,
            full_result_path.name,
        ],
    }
    _write_json(log_path, log_entry)

    print(json.dumps(results, indent=2))
    print("\nArtifacts written to:")
    print(f"- {submission_output_path}")
    for path in judge_output_paths:
        print(f"- {path}")
    print(f"- {critique_round_path}")
    print(f"- {synthesis_output_path}")
    print(f"- {full_result_path}")
    print(f"- {log_path}")


if __name__ == "__main__":
    main()
