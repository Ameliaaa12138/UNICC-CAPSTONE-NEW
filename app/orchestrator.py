from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

try:
    from app.judge1 import run_judge_1
    from app.judge2 import run_judge_2
    from app.judge3 import run_judge_3
    from app.synthesis import run_synthesis
except ModuleNotFoundError:
    from judge1 import run_judge_1
    from judge2 import run_judge_2
    from judge3 import run_judge_3
    from synthesis import run_synthesis


class SubmittedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    file_type: str = ""
    file_path: str = ""
    description: str = ""


class SubmissionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str
    submitted_by: str
    submission_timestamp: str = ""
    agent_name: str
    agent_description: str
    use_case: str
    deployment_context: str
    selected_frameworks: list[str] = Field(default_factory=list)
    risk_focus: list[str] = Field(default_factory=list)
    submitted_evidence: list[SubmittedEvidence] = Field(default_factory=list)
    notes: str = ""


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    reference: str
    description: str


class PolicyAlignmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    status: str
    note: str


class DetectedRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_name: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    description: str
    evidence_reference: str
    mitigation: str


class ExpertJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str
    module_name: str
    module_version: str
    assessment_timestamp: str
    perspective_type: str
    overall_risk_score: int = Field(ge=0, le=100)
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_findings: list[str]
    reasoning_summary: str
    evidence: list[EvidenceItem]
    policy_alignment: list[PolicyAlignmentItem]
    detected_risks: list[DetectedRisk]
    recommended_action: str
    raw_output_reference: str
    error_flag: bool
    error_message: str


class CritiqueRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participating_modules: list[str]
    agreement_points: list[str]
    disagreement_points: list[str]
    arbitration_notes: list[str]
    reconciled_risk_score: int = Field(ge=0, le=100)
    reconciled_risk_tier: Literal["Low", "Medium", "High", "Critical"]
    recommended_action: str


TIER_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}


def _keywords_for_findings(findings: list[str]) -> set[str]:
    joined = " ".join(findings).lower()
    topics = set()
    keyword_map = {
        "bias": ("bias", "fair", "discrimin"),
        "privacy": ("privacy", "pii", "data", "sensitive"),
        "transparency": ("transparency", "document", "explain", "disclosure"),
        "misuse": ("jailbreak", "misuse", "harm", "attack"),
        "compliance": ("compliance", "legal", "regulat", "governance", "policy"),
        "security": ("prompt injection", "evasion", "backdoor", "poison", "security", "breach"),
        "operations": ("oversight", "monitor", "audit", "owner", "deployment", "operational"),
    }
    for topic, patterns in keyword_map.items():
        if any(pattern in joined for pattern in patterns):
            topics.add(topic)
    return topics


def _risk_tier_from_score(score: int) -> Literal["Low", "Medium", "High", "Critical"]:
    if score >= 78:
        return "Critical"
    if score >= 58:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _highest_tier(tiers: list[str]) -> Literal["Low", "Medium", "High", "Critical"]:
    highest = "Low"
    for tier in tiers:
        if TIER_ORDER[tier] > TIER_ORDER[highest]:
            highest = tier
    return highest  # type: ignore[return-value]


def _majority_tier(results: list[ExpertJudgeOutput]) -> str | None:
    counts = Counter(result.risk_tier for result in results)
    tier, count = counts.most_common(1)[0]
    if count >= 2:
        return tier
    return None


def _confidence_weight(result: ExpertJudgeOutput) -> float:
    weight = max(result.confidence, 0.35)
    if result.error_flag:
        weight *= 0.5
    return weight


def _reconciled_score(results: list[ExpertJudgeOutput]) -> int:
    scores = [result.overall_risk_score for result in results]
    weighted_average = sum(result.overall_risk_score * _confidence_weight(result) for result in results) / sum(
        _confidence_weight(result) for result in results
    )
    median_score = float(median(scores))
    max_score = float(max(scores))
    score_spread = max(scores) - min(scores)

    if score_spread >= 20:
        return int(round(weighted_average * 0.45 + median_score * 0.25 + max_score * 0.30))
    if len({result.risk_tier for result in results}) > 1:
        return int(round(weighted_average * 0.65 + median_score * 0.35))
    return int(round(weighted_average))


def _critique_judges(judge_outputs: list[dict[str, Any]]) -> CritiqueRound:
    validated_results = TypeAdapter(list[ExpertJudgeOutput]).validate_python(judge_outputs)
    agreement_points: list[str] = []
    disagreement_points: list[str] = []
    arbitration_notes: list[str] = []

    participating_modules = [result.module_name for result in validated_results]
    tiers = [result.risk_tier for result in validated_results]
    tier_counts = Counter(tiers)
    score_spread = max(result.overall_risk_score for result in validated_results) - min(
        result.overall_risk_score for result in validated_results
    )

    majority_tier = _majority_tier(validated_results)
    if len(tier_counts) == 1:
        agreement_points.append(f"All three judges assign a {tiers[0]} overall risk tier.")
    elif majority_tier:
        agreement_points.append(
            f"Two of three judges converge on a {majority_tier} overall risk tier."
        )
        disagreement_points.append(
            "At least one judge assigns a different severity tier, so the council is not in full consensus."
        )
    else:
        disagreement_points.append(
            "All three judges split across different risk tiers, indicating a materially contested assessment."
        )

    topic_counter: Counter[str] = Counter()
    for result in validated_results:
        topic_counter.update(_keywords_for_findings(result.key_findings))

    shared_topics = sorted(topic for topic, count in topic_counter.items() if count >= 2)
    if shared_topics:
        agreement_points.append(
            "At least two judges flag overlapping topics: " + ", ".join(shared_topics) + "."
        )
    else:
        disagreement_points.append(
            "The judges emphasize different risk clusters across technical, governance, and operational perspectives."
        )

    if score_spread >= 20:
        disagreement_points.append(
            f"The council differs materially on severity by {score_spread} risk-score points."
        )

    if any(result.error_flag for result in validated_results):
        disagreement_points.append(
            "One or more judge modules returned fallback outputs because their primary evaluation path failed."
        )

    arbitration_notes.append(
        "The critique round uses a confidence-weighted reconciliation across all participating judges rather than a fixed Judge 1 vs Judge 2 comparison."
    )
    if score_spread >= 20:
        arbitration_notes.append(
            "Because score spread is material, the reconciled score is biased upward toward the highest-risk judgment to avoid understating unresolved concerns."
        )
    else:
        arbitration_notes.append(
            "Because the score spread is limited, the reconciled score stays close to the confidence-weighted council average."
        )
    if any(result.error_flag for result in validated_results):
        arbitration_notes.append(
            "Fallback or low-confidence outputs were downweighted, but their concerns were still retained in the disagreement log."
        )

    reconciled_risk_score = _reconciled_score(validated_results)
    reconciled_risk_tier = _risk_tier_from_score(reconciled_risk_score)

    if majority_tier and TIER_ORDER[majority_tier] > TIER_ORDER[reconciled_risk_tier]:
        reconciled_risk_tier = majority_tier  # type: ignore[assignment]

    highest_tier = _highest_tier(tiers)
    if highest_tier == "Critical" and TIER_ORDER[reconciled_risk_tier] < TIER_ORDER["High"]:
        reconciled_risk_tier = "High"

    if any(result.error_flag for result in validated_results) or reconciled_risk_tier == "Critical":
        recommended_action = "Escalate to human review after reconciling the combined technical, governance, and operational concerns."
    elif reconciled_risk_tier == "High":
        recommended_action = "Retest after addressing the combined technical, governance, and operational concerns."
    elif disagreement_points:
        recommended_action = "Proceed only with documented conditions and explicit review of the remaining cross-judge disagreements."
    else:
        recommended_action = "Use the shared findings as the primary remediation backlog before the next review."

    return CritiqueRound(
        participating_modules=participating_modules,
        agreement_points=agreement_points or ["All judges found at least some review-worthy concerns."],
        disagreement_points=disagreement_points,
        arbitration_notes=arbitration_notes,
        reconciled_risk_score=reconciled_risk_score,
        reconciled_risk_tier=reconciled_risk_tier,
        recommended_action=recommended_action,
    )


def run_pipeline(input_data: dict[str, Any]) -> dict[str, Any]:
    validated_input = SubmissionInput.model_validate(input_data)
    normalized_input = validated_input.model_dump()

    raw_outputs = [
        run_judge_1(normalized_input),
        run_judge_2(normalized_input),
        run_judge_3(normalized_input),
    ]
    validated_outputs = TypeAdapter(list[ExpertJudgeOutput]).validate_python(raw_outputs)

    critique_round = _critique_judges([result.model_dump() for result in validated_outputs])
    synthesis_output = run_synthesis(
        [result.model_dump() for result in validated_outputs],
        critique_round.model_dump(),
    )

    return {
        "judge_outputs": [result.model_dump() for result in validated_outputs],
        "critique_round": critique_round.model_dump(),
        "synthesis_output": synthesis_output,
    }
