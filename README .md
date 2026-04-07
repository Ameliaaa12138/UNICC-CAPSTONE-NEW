# UNICC AI Safety Lab (Spring 2026)

A multi agent AI safety evaluation system that uses three expert judges and a synthesis layer to assess AI agents before deployment.

---

## Overview

This project implements a council based AI Safety Lab for evaluating AI agents in a controlled environment.

This system is implemented as a fully integrated end-to-end pipeline.

The Streamlit frontend is connected to the FastAPI backend via HTTP requests.
User inputs are sent to the backend through the `/submit` endpoint, where the multi-judge evaluation pipeline is executed.
The frontend dynamically renders the real evaluation results returned from the backend.

The system:

1. Collects structured input about an AI agent through the frontend interface
2. Evaluates it using three independent expert judges
3. Reconciles outputs through a critique round
4. Produces a final safety recommendation for human review

The goal is to provide transparent, auditable, and multi perspective safety evaluation before deployment.

---

## System Architecture

Input  
→ Judge 1 (Technical Safety)  
→ Judge 2 (Governance and Policy)  
→ Judge 3 (Operational Risk)  
→ Critique Round  
→ Synthesis Layer  
→ Final Decision  

---

## Expert Judges

### Judge 1: Technical Safety Judge
Focus: prompt injection, jailbreak susceptibility, adversarial robustness, and unsafe behavior.

### Judge 2: Governance and Policy Alignment Judge
Focus: governance, policy alignment, compliance related concerns, and control documentation.

### Judge 3: Operational and System Risk Judge
Focus: workflow reliability, deployment context, operational vulnerabilities, and system level concerns.

---

## Current Platform Status

- The project supports a full pipeline from submission to final decision
- File upload and API based submission are supported
- Outputs and logs are automatically generated
- The system is designed as a prototype for AI agent safety evaluation before deployment

---

## Repository Structure

```text
UNICC CAPSTONE/
├── app/
│   ├── judge1.py
│   ├── judge2.py
│   ├── judge3.py
│   ├── main.py
│   ├── orchestrator.py
│   └── synthesis.py
├── demo.py
├── requirements.txt
├── README.md
├── outputs/
├── logs/
└── artifacts/
```

---

## Installation

```text
pip install -r requirements.txt
```

---

## Local SLM Setup

This project uses a local Small Language Model through Ollama.

### 1. Install Ollama

Install Ollama on your local machine.

### 2. Pull the model

```text
ollama pull llama3.2:3b
```

### 3. Start Ollama

```text
ollama serve
```

### 4. Optional local API test

```text
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Return a short JSON object with status ok",
  "stream": false
}'
```

---

## Running the Project

### 1. Start Backend

```text
cd ~/Desktop/"UNICC CAPSTONE"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend docs:

```text
http://127.0.0.1:8000/docs
```

### 2. Start Frontend

```text
cd ~/Desktop/"UNICC CAPSTONE"
streamlit run demo.py
```

Frontend:

```text
http://localhost:8501
```

---

## How the System Works

- User submits agent information through the UI or API
- The Streamlit frontend sends data to the backend through `POST /submit` and renders the returned results dynamically
- Backend runs three judges
- A critique round reconciles judge outputs
- A synthesis layer produces the final decision
- The result is returned for dashboard display and saved to output files

---

## API Usage

### Main Endpoint

`POST /submit`

### Processing Flow

Input  
→ API  
→ Multi Judge Evaluation  
→ Critique Round  
→ Final Decision  

---

## Input Example

### Deployment Context
NYU AI Sandbox demo environment

### Agent Description
A multi judge AI safety evaluation system that uses multiple expert judges and a synthesis step to assess the safety risk of an AI agent.

---

## Output Structure

### Judge Output Fields

- `risk_score`
- `risk_level`
- `confidence`
- `reasoning`
- `recommendation`

### Critique Round Fields

- `agreement_points`
- `disagreement_points`
- `arbitration_notes`
- `reconciled_risk_score`
- `reconciled_risk_tier`

### Final Synthesis Fields

- `final_recommendation`
- `final_risk_tier`
- `rationale`
- `next_actions`

---

## Output Files

After the pipeline runs, the system generates:

- `outputs/judge1_output.json`
- `outputs/judge2_output.json`
- `outputs/judge3_output.json`
- `outputs/synthesis_output.json`
- `outputs/full_result.json`

Log file:

- `logs/pipeline_log.json`

---

## API Key and Environment Handling

This project is designed to run without requiring a hardcoded API key.

### Key Principles

- No real API key should be hardcoded in the source code
- Environment variables should be used for any external API based configuration
- The system should fail gracefully with a clear error message if a required key or model is missing
- No part of the pipeline should depend on one personal key only

### Recommended Pattern

Use environment variables such as:

```text
OPENAI_API_KEY=your_key_here
```

or

```text
GEMINI_API_KEY=your_key_here
```

or for local model configuration:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

---

## Important Note About Judge 3

Some project configurations describe Judge 3 as using Gemini API.

If Judge 3 is configured to use Gemini:

```text
GEMINI_API_KEY=your_google_api_key_here
```

In that case:

- Judge 3 must read the key from an environment variable
- Judge 3 must not hardcode any key
- If the key is missing, the system should return a clear error or fallback response
- The rest of the pipeline should not silently break

---

## Graceful Failure Handling

If required resources are missing, such as:

- Ollama is not running
- The local model is not installed
- An external API key is missing

The system should return a clear message, for example:

```json
{
  "status": "error",
  "message": "Required model or API key not available"
}
```

This prevents silent failure and makes the evaluation environment easier to debug.

---

## Optional Mock Mode

For testing or CI environments, a mock mode can be used.

```text
MOCK_MODE=1 
```
Note: The current demo uses live backend integration and does not rely on mock mode.

In mock mode:

- No live LLM calls are made
- The system returns simulated outputs for testing

---

## Frontend Integration Notes

For dashboard integration, the frontend should focus on these response fields.

### Final Decision

- `results.synthesis_output.final_recommendation`
- `results.synthesis_output.final_risk_tier`
- `results.synthesis_output.rationale`
- `results.synthesis_output.next_actions`

### Critique Round

- `results.critique_round.agreement_points`
- `results.critique_round.disagreement_points`
- `results.critique_round.arbitration_notes`
- `results.critique_round.reconciled_risk_score`
- `results.critique_round.reconciled_risk_tier`

If the current backend does not expose a field named `final_verdict`, the frontend should use `results.synthesis_output` as the final result source.

---

## Integration Rules

Future development should primarily happen in:

- `app/judge1.py`
- `app/judge2.py`
- `app/judge3.py`

The following files should remain stable unless necessary:

- `app/main.py`
- `app/orchestrator.py`
- `app/synthesis.py`

All future expert modules should preserve the common input and output schema.

---

## Demo Summary

Input → Frontend → API → Multi Judge Evaluation → Critique Round → Synthesis → Frontend Display → Final Decision

---

## Notes

- This is a prototype system for AI safety evaluation
- The current version is intended for development, testing, and future portability
- It is designed to support architecture definition, expert module integration, upload and submission flow prototyping, and future deployment in a sandbox environment

---

## Key Idea

The interface acts as the entry point to a multi agent AI safety evaluation pipeline, enabling users to interact with the system and view real-time evaluation results generated by the backend.
