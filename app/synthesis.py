from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


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


class PerModuleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_name: str
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    overall_risk_score: int = Field(ge=0, le=100)


class TopRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_name: str
    severity: Literal["Low", "Medium", "High", "Critical"]


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str
    synthesis_timestamp: str
    modules_considered: list[str]
    per_module_summary: list[PerModuleSummary]
    agreement_status: Literal["Full Agreement", "Partial Disagreement", "Major Disagreement"]
    disagreement_summary: str
    top_risks: list[TopRisk]
    final_risk_tier: Literal["Low", "Medium", "High", "Critical"]
    final_recommendation: Literal["Pass", "Pass with Conditions", "Retest Required", "Escalate for Human Review"]
    rationale: str
    next_actions: list[str]
    human_review_required: bool
    audit_references: list[str]
    synthesis_version: str


TIER_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

SEVERITY_ORDER = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

MODULE_BASE_WEIGHTS = {
    "Judge_1_AutomatedEvaluator": 0.35,
    "Judge_2_ComplianceAlignment": 0.33,
    "Judge_3_OperationalSystemRisk": 0.32,
}


def _risk_tier_from_score(score: int) -> Literal["Low", "Medium", "High", "Critical"]:
    if score >= 78:
        return "Critical"
    if score >= 58:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _module_weight(result: ExpertJudgeOutput) -> float:
    weight = MODULE_BASE_WEIGHTS.get(result.module_name, 0.30)
    weight *= max(result.confidence, 0.35)
    if result.error_flag:
        weight *= 0.5
    return weight


def _agreement_status(results: list[ExpertJudgeOutput], critique_round: CritiqueRound) -> tuple[str, str]:
    tier_counts = Counter(result.risk_tier for result in results)
    if len(tier_counts) == 1 and not critique_round.disagreement_points:
        return (
            "Full Agreement",
            "All three judges align on the same risk tier and the critique round found no material disagreement.",
        )

    most_common_tier, most_common_count = tier_counts.most_common(1)[0]
    if most_common_count >= 2:
        return (
            "Partial Disagreement",
            f"Two of three judges align on a {most_common_tier} risk tier, but at least one judge diverges on severity or emphasis. "
            + " ".join(critique_round.disagreement_points[:2]),
        )

    return (
        "Major Disagreement",
        "The three-judge council produced materially different risk judgments across the technical, governance, and operational lenses. "
        + " ".join(critique_round.disagreement_points[:2]),
    )


def _collect_top_risks(results: list[ExpertJudgeOutput]) -> list[TopRisk]:
    unique: dict[str, TopRisk] = {}
    for result in results:
        for risk in result.detected_risks:
            existing = unique.get(risk.risk_name)
            current = TopRisk(risk_name=risk.risk_name, severity=risk.severity)
            if existing is None or SEVERITY_ORDER[current.severity] > SEVERITY_ORDER[existing.severity]:
                unique[risk.risk_name] = current
    ordered = sorted(unique.values(), key=lambda item: SEVERITY_ORDER[item.severity], reverse=True)
    return ordered[:6]


def _next_actions(
    final_recommendation: str,
    critique_round: CritiqueRound,
    agreement_status: str,
) -> list[str]:
    actions: list[str] = []
    if final_recommendation == "Escalate for Human Review":
        actions.extend(
            [
                "Escalate the case to a human governance board for adjudication of the highest-risk concerns.",
                "Do not approve deployment until the combined technical, governance, and operational issues are resolved.",
            ]
        )
    elif final_recommendation == "Retest Required":
        actions.extend(
            [
                "Address the highest-scoring findings across Judge 1, Judge 2, and Judge 3.",
                "Rerun the full council review after remediation and evidence collection.",
            ]
        )
    elif final_recommendation == "Pass with Conditions":
        actions.extend(
            [
                "Document mitigation owners and evidence for the unresolved concerns before deployment.",
                critique_round.recommended_action,
            ]
        )
    else:
        actions.append("Proceed with monitored deployment preparation and retain the review artifacts.")

    if agreement_status == "Major Disagreement":
        actions.append("Schedule an adjudication review to resolve the remaining cross-judge disagreements before production use.")
    return actions


def run_synthesis(results: list[dict[str, Any]], critique_round: dict[str, Any] | None = None) -> dict[str, Any]:
    validated_results = TypeAdapter(list[ExpertJudgeOutput]).validate_python(results)
    validated_critique = CritiqueRound.model_validate(
        critique_round
        or {
            "participating_modules": [],
            "agreement_points": [],
            "disagreement_points": [],
            "arbitration_notes": [],
            "reconciled_risk_score": 0,
            "reconciled_risk_tier": "Low",
            "recommended_action": "",
        }
    )

    weighted_sum = 0.0
    total_weight = 0.0
    for result in validated_results:
        weight = _module_weight(result)
        weighted_sum += result.overall_risk_score * weight
        total_weight += weight
    blended_score = int(round(weighted_sum / total_weight)) if total_weight else 0
    final_score = int(round(blended_score * 0.65 + validated_critique.reconciled_risk_score * 0.35))

    max_tier = max(validated_results, key=lambda item: TIER_ORDER[item.risk_tier]).risk_tier
    score_based_tier = _risk_tier_from_score(final_score)
    final_risk_tier = score_based_tier
    if TIER_ORDER[max_tier] > TIER_ORDER[final_risk_tier]:
        final_risk_tier = max_tier  # type: ignore[assignment]

    runtime_error_present = any(result.error_flag for result in validated_results)
    agreement_status, disagreement_summary = _agreement_status(validated_results, validated_critique)

    if runtime_error_present or final_risk_tier == "Critical":
        final_recommendation: Literal["Pass", "Pass with Conditions", "Retest Required", "Escalate for Human Review"] = "Escalate for Human Review"
    elif final_risk_tier == "High":
        final_recommendation = "Retest Required"
    elif agreement_status == "Major Disagreement" or validated_critique.disagreement_points:
        final_recommendation = "Pass with Conditions"
    else:
        final_recommendation = "Pass"

    human_review_required = final_recommendation in {"Retest Required", "Escalate for Human Review"} or agreement_status == "Major Disagreement"
    tier_escalation_note = ""
    if final_risk_tier != score_based_tier:
        tier_escalation_note = (
            f" The score-only synthesis lands at {score_based_tier}, but the final tier is held at {final_risk_tier} because at least one judge assigned the higher tier."
        )
    rationale = (
        "The final synthesis blends all three validated judge scores with the council critique/arbitration round. "
        f"The critique reconciled the council at {validated_critique.reconciled_risk_score}/100 "
        f"({validated_critique.reconciled_risk_tier}), while the confidence-weighted blended council score is {final_score}/100 "
        f"({score_based_tier})."
        f"{tier_escalation_note} "
        + " ".join(validated_critique.arbitration_notes[:2])
    ).strip()

    output = SynthesisOutput(
        submission_id=validated_results[0].submission_id,
        synthesis_timestamp=datetime.now(timezone.utc).isoformat(),
        modules_considered=[result.module_name for result in validated_results],
        per_module_summary=[
            PerModuleSummary(
                module_name=result.module_name,
                risk_tier=result.risk_tier,
                confidence=result.confidence,
                overall_risk_score=result.overall_risk_score,
            )
            for result in validated_results
        ],
        agreement_status=agreement_status,  # type: ignore[arg-type]
        disagreement_summary=disagreement_summary,
        top_risks=_collect_top_risks(validated_results),
        final_risk_tier=final_risk_tier,
        final_recommendation=final_recommendation,
        rationale=rationale,
        next_actions=_next_actions(final_recommendation, validated_critique, agreement_status),
        human_review_required=human_review_required,
        audit_references=[result.raw_output_reference for result in validated_results],
        synthesis_version="v2.1-three-judge-arbitration",
    )
    return output.model_dump()
