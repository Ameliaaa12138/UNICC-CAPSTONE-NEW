import streamlit as st
import requests
from typing import Any, Dict, List

BACKEND_URL = "http://127.0.0.1:8000/submit"


st.set_page_config(
    page_title="AI Safety Agent Demo",
    layout="wide"
)

st.title("AI Safety Agent Demo")
st.markdown(
    "This interface serves as the entry point to a multi-judge AI safety evaluation pipeline."
)


def safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def safe_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return str(value)


def build_payload(
    submission_id: str,
    submitted_by: str,
    agent_name: str,
    agent_description: str,
    use_case: str,
    deployment_context: str,
    selected_frameworks: str,
    risk_focus: str,
    notes: str,
) -> Dict[str, str]:
    return {
        "submission_id": submission_id,
        "submitted_by": submitted_by,
        "agent_name": agent_name,
        "agent_description": agent_description,
        "use_case": use_case,
        "deployment_context": deployment_context,
        "selected_frameworks": selected_frameworks,
        "risk_focus": risk_focus,
        "notes": notes,
    }


def call_backend(payload: Dict[str, str]) -> Dict[str, Any]:
    response = requests.post(BACKEND_URL, data=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ===== Input Section =====
st.header("Submission Information")

col1, col2 = st.columns(2)

with col1:
    submission_id = st.text_input("Submission ID", value="demo_001")
    submitted_by = st.text_input("Submitted By", value="Qing Wang")
    agent_name = st.text_input("Agent Name", value="AI Safety Agent Demo")

with col2:
    use_case = st.text_input(
        "Use Case",
        value="Evaluating the safety and trustworthiness of AI agents before deployment."
    )
    deployment_context = st.text_input(
        "Deployment Context",
        value="NYU AI Sandbox demo environment"
    )

agent_description = st.text_area(
    "Agent Description / Input for Evaluation",
    value="A multi-judge AI safety evaluation system that uses multiple expert judges and a synthesis step to assess the safety risk of an AI agent.",
    height=180
)

with st.expander("Optional Fields"):
    selected_frameworks = st.text_input(
        "Selected Frameworks",
        value=""
    )
    risk_focus = st.text_input(
        "Risk Focus",
        value=""
    )
    notes = st.text_area(
        "Notes",
        value="Demo submission for frontend-backend integration testing.",
        height=100
    )


# ===== Button =====
if st.button("Evaluate Safety", use_container_width=True):
    payload = build_payload(
        submission_id=submission_id,
        submitted_by=submitted_by,
        agent_name=agent_name,
        agent_description=agent_description,
        use_case=use_case,
        deployment_context=deployment_context,
        selected_frameworks=selected_frameworks,
        risk_focus=risk_focus,
        notes=notes,
    )

    try:
        with st.spinner("Running multi-judge safety evaluation..."):
            result = call_backend(payload)

        st.success("Evaluation completed successfully.")

        # ===== Submission Echo =====
        st.header("Submitted Input")
        st.markdown(f"- **Submission ID:** `{submission_id}`")
        st.markdown(f"- **Submitted By:** `{submitted_by}`")
        st.markdown(f"- **Agent Name:** `{agent_name}`")
        st.markdown(f"- **Use Case:** `{use_case}`")
        st.markdown(f"- **Deployment Context:** `{deployment_context}`")
        st.markdown(f"- **Agent Description:** {agent_description}")

        results = result.get("results", {})
        synthesis_output = results.get("synthesis_output", {})
        critique_round = results.get("critique_round", {})
        judge_outputs = safe_list(results.get("judge_outputs", []))

        # ===== Final Decision =====
        st.header("Final Decision")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Final Risk Tier",
                safe_text(synthesis_output.get("final_risk_tier"))
            )
        with metric_col2:
            st.metric(
                "Final Recommendation",
                safe_text(synthesis_output.get("final_recommendation"))
            )
        with metric_col3:
            st.metric(
                "Human Review Required",
                safe_text(synthesis_output.get("human_review_required"))
            )

        st.subheader("Rationale")
        st.write(safe_text(synthesis_output.get("rationale")))

        st.subheader("Next Actions")
        next_actions = safe_list(synthesis_output.get("next_actions"))
        if next_actions:
            for action in next_actions:
                st.write(f"- {action}")
        else:
            st.write("No next actions returned.")

        # ===== Judge Outputs =====
        st.header("Judge Outputs")

        if judge_outputs:
            for idx, judge in enumerate(judge_outputs, start=1):
                module_name = safe_text(judge.get("module_name"), f"Judge {idx}")
                with st.expander(f"Judge {idx}: {module_name}", expanded=False):
                    jcol1, jcol2, jcol3 = st.columns(3)

                    with jcol1:
                        st.metric("Risk Tier", safe_text(judge.get("risk_tier")))
                    with jcol2:
                        st.metric("Risk Score", safe_text(judge.get("overall_risk_score")))
                    with jcol3:
                        st.metric("Confidence", safe_text(judge.get("confidence")))

                    st.write(f"**Perspective Type:** {safe_text(judge.get('perspective_type'))}")
                    st.write(f"**Reasoning Summary:** {safe_text(judge.get('reasoning_summary'))}")
                    st.write(f"**Recommended Action:** {safe_text(judge.get('recommended_action'))}")

                    key_findings = safe_list(judge.get("key_findings"))
                    if key_findings:
                        st.write("**Key Findings:**")
                        for finding in key_findings:
                            st.write(f"- {finding}")

                    error_flag = judge.get("error_flag", False)
                    st.write(f"**Error Flag:** {error_flag}")

                    if error_flag:
                        st.warning(safe_text(judge.get("error_message"), "Unknown error."))

        else:
            st.info("No judge outputs returned.")

        # ===== Critique Round =====
        st.header("Critique Round")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(
                "Reconciled Risk Score",
                safe_text(critique_round.get("reconciled_risk_score"))
            )
        with cc2:
            st.metric(
                "Reconciled Risk Tier",
                safe_text(critique_round.get("reconciled_risk_tier"))
            )

        participating_modules = safe_list(critique_round.get("participating_modules"))
        agreement_points = safe_list(critique_round.get("agreement_points"))
        disagreement_points = safe_list(critique_round.get("disagreement_points"))
        arbitration_notes = safe_list(critique_round.get("arbitration_notes"))

        st.subheader("Participating Modules")
        if participating_modules:
            for module in participating_modules:
                st.write(f"- {module}")
        else:
            st.write("No participating modules returned.")

        st.subheader("Agreement Points")
        if agreement_points:
            for point in agreement_points:
                st.write(f"- {point}")
        else:
            st.write("No agreement points returned.")

        st.subheader("Disagreement Points")
        if disagreement_points:
            for point in disagreement_points:
                st.write(f"- {point}")
        else:
            st.write("No disagreement points returned.")

        st.subheader("Arbitration Notes")
        if arbitration_notes:
            for note in arbitration_notes:
                st.write(f"- {note}")
        else:
            st.write("No arbitration notes returned.")

        st.subheader("Critique Recommendation")
        st.write(safe_text(critique_round.get("recommended_action")))

        # ===== Raw JSON =====
        st.header("Raw JSON Output")
        st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the backend. Please make sure the backend is running at "
            "http://127.0.0.1:8000"
        )
    except requests.exceptions.Timeout:
        st.error("The backend request timed out.")
    except requests.exceptions.HTTPError as e:
        st.error("The backend returned an HTTP error.")
        st.exception(e)
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.exception(e)