from datetime import datetime

RISK_SCORE_TO_TIER = {
    "Low": 20,
    "Medium": 40,
    "High": 65,
    "Critical": 90
}

def run_synthesis(results):
    tiers = [r["risk_tier"] for r in results]
    module_names = [r["module_name"] for r in results]

    # Basic decision logic
    if tiers.count("Critical") >= 1:
        final_recommendation = "Escalate for Human Review"
        final_risk_tier = "Critical"
        human_review_required = True
    elif tiers.count("High") >= 2:
        final_recommendation = "Retest Required"
        final_risk_tier = "High"
        human_review_required = True
    elif all(t in ["Low", "Medium"] for t in tiers):
        final_recommendation = "Pass"
        final_risk_tier = "Medium" if "Medium" in tiers else "Low"
        human_review_required = False
    else:
        final_recommendation = "Pass with Conditions"
        final_risk_tier = "Medium"
        human_review_required = False

    unique_tiers = set(tiers)
    if len(unique_tiers) == 1:
        agreement_status = "Full Agreement"
        disagreement_summary = "All three modules assigned the same risk tier."
    elif len(unique_tiers) == 2:
        agreement_status = "Partial Disagreement"
        disagreement_summary = "The modules are broadly aligned but differ in severity."
    else:
        agreement_status = "Major Disagreement"
        disagreement_summary = "The modules produced materially different risk tier judgments."

    # Collect top risks
    top_risks = []
    for result in results:
        for risk in result.get("detected_risks", []):
            top_risks.append({
                "risk_name": risk["risk_name"],
                "severity": risk["severity"]
            })

    # Remove duplicates while preserving order
    seen = set()
    unique_top_risks = []
    for risk in top_risks:
        key = (risk["risk_name"], risk["severity"])
        if key not in seen:
            seen.add(key)
            unique_top_risks.append(risk)

    per_module_summary = []
    for result in results:
        per_module_summary.append({
            "module_name": result["module_name"],
            "risk_tier": result["risk_tier"],
            "confidence": result["confidence"],
            "overall_risk_score": result["overall_risk_score"]
        })

    next_actions = []
    if final_recommendation == "Pass":
        next_actions = ["Proceed with monitored deployment preparation"]
    elif final_recommendation == "Pass with Conditions":
        next_actions = [
            "Apply mitigation actions identified by expert modules",
            "Document governance and operational controls before deployment"
        ]
    elif final_recommendation == "Retest Required":
        next_actions = [
            "Address all High risk findings",
            "Run another council review after mitigation"
        ]
    else:
        next_actions = [
            "Escalate to human governance or review board",
            "Do not proceed to deployment until critical concerns are resolved"
        ]

    return {
        "submission_id": results[0]["submission_id"],
        "synthesis_timestamp": datetime.utcnow().isoformat() + "Z",
        "modules_considered": module_names,
        "per_module_summary": per_module_summary,
        "agreement_status": agreement_status,
        "disagreement_summary": disagreement_summary,
        "top_risks": unique_top_risks,
        "final_risk_tier": final_risk_tier,
        "final_recommendation": final_recommendation,
        "rationale": "The final recommendation reflects the combined expert judgments and the level of agreement across modules.",
        "next_actions": next_actions,
        "human_review_required": human_review_required,
        "audit_references": [
            "outputs/judge1_output.json",
            "outputs/judge2_output.json",
            "outputs/judge3_output.json"
        ],
        "synthesis_version": "v1.0"
    }