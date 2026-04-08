"""
inference.py — Baseline Agent for Email Triage Environment
===========================================================
Runs the agent across all 3 tasks (easy → medium → hard).
Uses OpenAI client pointed at API_BASE_URL / MODEL_NAME / HF_TOKEN.

Log format is STRICTLY:
  [START] {"task_id": ..., "episode_id": ...}
  [STEP]  {"step": ..., "email_id": ..., "action": {...}, "reward": ..., "done": ...}
  [END]   {"task_id": ..., "total_reward": ..., "steps": ...}

Any deviation causes incorrect evaluation scoring — do NOT modify the log format.
"""

import os
import json
import sys
import time
import requests
from openai import OpenAI

# ── Environment config ────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# ── OpenAI client pointed at HF Inference API ────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

# ── Valid enum values (mirrors models.py) — used to sanitise LLM output ──────
VALID_PRIORITIES  = {"urgent", "normal", "low"}
VALID_CATEGORIES  = {
    "billing", "technical_support", "sales", "hr",
    "product_feedback", "legal_compliance", "other",
}
VALID_ROUTES      = {
    "billing_support", "tier1_support", "tier2_support", "engineering_escalation",
    "sales_team", "sales_engineering", "hr_recruitment", "product_team",
    "customer_success", "customer_retention", "legal_team", "compliance_team",
    "data_privacy_officer", "partnerships_team",
}


def _sanitise(decision: dict) -> dict:
    """
    Coerce any invalid or unexpected LLM output values to safe defaults
    before the dict is sent to the FastAPI /step endpoint.
    Prevents Pydantic validation errors from crashing the episode.
    """
    decision["priority"] = (
        decision.get("priority") if decision.get("priority") in VALID_PRIORITIES else "normal"
    )
    decision["category"] = (
        decision.get("category") if decision.get("category") in VALID_CATEGORIES else "other"
    )
    decision["route"] = (
        decision.get("route") if decision.get("route") in VALID_ROUTES else "tier1_support"
    )
    # Optional fields: nullify anything that isn't a known enum value
    # (catches "none", "None", "N/A", empty string, etc.)
    sc = decision.get("secondary_category")
    decision["secondary_category"] = sc if sc in VALID_CATEGORIES else None

    sr = decision.get("secondary_route")
    decision["secondary_route"] = sr if sr in VALID_ROUTES else None

    return decision

SYSTEM_PROMPT = """You are an expert email triage assistant. 
Your job is to read an email and decide:
1. priority   — must be exactly one of: urgent, normal, low
2. category   — must be exactly one of: billing, technical_support, sales, hr, product_feedback, legal_compliance, other
3. route      — must be exactly one of: billing_support, tier1_support, tier2_support, engineering_escalation, sales_team, sales_engineering, hr_recruitment, product_team, customer_success, customer_retention, legal_team, compliance_team, data_privacy_officer, partnerships_team
4. secondary_category (optional) — same values as category, use only if the email has a clear second intent
5. secondary_route (optional)    — same values as route, use only if there is a clear secondary routing need

Rules:
- urgent: time-sensitive, financial risk, legal threat, production outage, or strong churn signal
- normal: needs resolution but no immediate deadline
- low: informational, no action needed soon
- Always route legal/GDPR/fraud emails to legal_team or compliance_team, NOT to billing or support
- If an email contains multiple intents, identify the PRIMARY one and set secondary fields

Respond ONLY with a valid JSON object — no explanation, no markdown, no extra text.
Example:
{
  "priority": "urgent",
  "category": "billing",
  "route": "billing_support",
  "secondary_category": null,
  "secondary_route": null,
  "reasoning": "Duplicate charge + explicit urgency = billing urgent"
}"""


def call_llm(email_subject: str, email_body: str, sender: str) -> dict:
    """Call the LLM and parse its triage decision."""
    user_message = f"""Triage this email:

FROM: {sender}
SUBJECT: {email_subject}

BODY:
{email_body}

Respond with a JSON object only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=300,
            temperature=0.1,   # low temp for consistent structured output
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return _sanitise(json.loads(raw))

    except json.JSONDecodeError:
        # Fallback: safe default action
        return _sanitise({
            "priority":           "normal",
            "category":           "other",
            "route":              "tier1_support",
            "secondary_category": None,
            "secondary_route":    None,
            "reasoning":          "JSON parse failed — using fallback",
        })
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
        return _sanitise({
            "priority":           "normal",
            "category":           "other",
            "route":              "tier1_support",
            "secondary_category": None,
            "secondary_route":    None,
            "reasoning":          f"LLM error: {str(e)}",
        })


def run_task(task_id: str) -> dict:
    """Run one full episode for a given task. Returns summary stats."""

    # ── /reset ────────────────────────────────────────────────────────────────
    reset_resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    # Grab episode_id from state (reset doesn't return it directly)
    state_resp = requests.get(f"{ENV_URL}/state")
    state_resp.raise_for_status()
    episode_id = state_resp.json().get("episode_id", "unknown")

    # ── [START] log ───────────────────────────────────────────────────────────
    print(json.dumps({
        "log_type":  "START",
        "task_id":   task_id,
        "episode_id": episode_id,
    }), flush=True)

    total_reward = 0.0
    step_num     = 0
    done         = False

    while not done:
        step_num += 1

        # ── Call LLM to decide triage ─────────────────────────────────────────
        llm_decision = call_llm(
            email_subject=obs["subject"],
            email_body=obs["body"],
            sender=obs["sender"],
        )

        # Build action payload — filter None values
        action_payload = {
            "priority": llm_decision.get("priority", "normal"),
            "category": llm_decision.get("category", "other"),
            "route":    llm_decision.get("route",    "tier1_support"),
        }
        if llm_decision.get("secondary_category"):
            action_payload["secondary_category"] = llm_decision["secondary_category"]
        if llm_decision.get("secondary_route"):
            action_payload["secondary_route"] = llm_decision["secondary_route"]
        if llm_decision.get("reasoning"):
            action_payload["reasoning"] = llm_decision["reasoning"]

        # ── /step ─────────────────────────────────────────────────────────────
        step_resp = requests.post(
            f"{ENV_URL}/step",
            json=action_payload,
            headers={"Content-Type": "application/json"},
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = result["reward"]
        done   = result["done"]
        total_reward += reward

        # Update obs for next step
        obs = result["observation"]

        # ── [STEP] log ────────────────────────────────────────────────────────
        print(json.dumps({
            "log_type":  "STEP",
            "task_id":   task_id,
            "episode_id": episode_id,
            "step":      step_num,
            "email_id":  result["info"].get("graded_email_id", "unknown"),
            "action":    action_payload,
            "reward":    reward,
            "done":      done,
            "cumulative_reward": result["info"].get("cumulative_reward", total_reward),
        }), flush=True)

        # Small delay to avoid hammering the API
        time.sleep(0.5)

    # ── [END] log ─────────────────────────────────────────────────────────────
    print(json.dumps({
        "log_type":     "END",
        "task_id":      task_id,
        "episode_id":   episode_id,
        "total_reward": round(total_reward, 4),
        "steps":        step_num,
        "avg_reward":   round(total_reward / max(step_num, 1), 4),
    }), flush=True)

    return {
        "task_id":      task_id,
        "total_reward": round(total_reward, 4),
        "steps":        step_num,
    }


def main():
    print(f"[INFO] Starting Email Triage inference", file=sys.stderr)
    print(f"[INFO] ENV_URL={ENV_URL}  MODEL={MODEL_NAME}", file=sys.stderr)

    # Wait for server to be ready
    for attempt in range(12):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"[INFO] Server is ready.", file=sys.stderr)
                break
        except Exception:
            pass
        print(f"[INFO] Waiting for server... attempt {attempt + 1}/12", file=sys.stderr)
        time.sleep(5)
    else:
        print("[ERROR] Server did not become ready in time.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for task_id in TASKS:
        print(f"\n[INFO] Running task: {task_id}", file=sys.stderr)
        try:
            result = run_task(task_id)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            # Log a zero-score END so evaluator doesn't crash
            print(json.dumps({
                "log_type":     "END",
                "task_id":      task_id,
                "episode_id":   "error",
                "total_reward": 0.0,
                "steps":        0,
                "avg_reward":   0.0,
            }), flush=True)

    # Final summary to stderr (not evaluated)
    print("\n[INFO] === Final Summary ===", file=sys.stderr)
    for r in all_results:
        print(
            f"[INFO] {r['task_id']}: total_reward={r['total_reward']}  steps={r['steps']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
