"""
main.py — Email Triage OpenEnv Server
Implements the full OpenEnv HTTP spec:
  POST /reset          → start a new episode, receive first email
  POST /step           → submit a triage action, receive reward + next email
  GET  /state          → inspect current environment state
  GET  /health         → liveness probe (must return 200)
  GET  /tasks          → list all available tasks with metadata
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models import (
    TriageAction,
    EmailObservation,
    TriageState,
    StepResult,
)
from graders.triage_grader import grade


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Triage Environment",
    description=(
        "An OpenEnv-compatible RL environment where an AI agent learns to "
        "prioritize, categorize, and route real-world emails."
    ),
    version="1.0.0",
)

# ── Load email dataset once at startup ────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "emails.json"

with open(DATA_PATH) as f:
    ALL_EMAILS: list[dict] = json.load(f)

EMAILS_BY_DIFFICULTY: dict[str, list[dict]] = {
    "easy":   [e for e in ALL_EMAILS if e["difficulty"] == "easy"],
    "medium": [e for e in ALL_EMAILS if e["difficulty"] == "medium"],
    "hard":   [e for e in ALL_EMAILS if e["difficulty"] == "hard"],
}

# Task definitions: each task locks to a difficulty level
TASKS: dict[str, dict[str, Any]] = {
    "easy_triage": {
        "difficulty":   "easy",
        "description":  "Triage emails with clear, explicit signals. Unambiguous priority, category, and routing.",
        "max_steps":    5,
    },
    "medium_triage": {
        "difficulty":   "medium",
        "description":  "Triage emails with implicit urgency and overlapping categories. Requires reading between the lines.",
        "max_steps":    5,
    },
    "hard_triage": {
        "difficulty":   "hard",
        "description":  "Triage complex multi-intent emails. Identify primary and secondary intent, legal/compliance edge cases, and churn signals.",
        "max_steps":    5,
    },
}

# ── In-memory session store (one active session) ──────────────────────────────
# For a production multi-session env, use a dict keyed by episode_id.

_session: dict[str, Any] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_email(difficulty: str, seen_ids: list[str]) -> dict | None:
    """Return an unseen email for the given difficulty, or None if exhausted."""
    pool = [e for e in EMAILS_BY_DIFFICULTY[difficulty] if e["id"] not in seen_ids]
    return random.choice(pool) if pool else None


def _email_to_observation(email: dict, task_id: str) -> EmailObservation:
    return EmailObservation(
        email_id=email["id"],
        subject=email["subject"],
        body=email["body"],
        sender=email["sender"],
        timestamp=email["timestamp"],
        difficulty=email["difficulty"],
        task_id=task_id,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — must always return 200."""
    return {"status": "ok", "environment": "email-triage-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks and their metadata."""
    return {
        "tasks": [
            {
                "task_id":    task_id,
                "difficulty": meta["difficulty"],
                "description": meta["description"],
                "max_steps":  meta["max_steps"],
                "email_count": len(EMAILS_BY_DIFFICULTY[meta["difficulty"]]),
            }
            for task_id, meta in TASKS.items()
        ]
    }


@app.post("/reset")
def reset(task_id: str = "easy_triage") -> dict:
    """
    Start a new episode.
    Query param: task_id (easy_triage | medium_triage | hard_triage)
    Returns the first EmailObservation for this episode.
    """
    global _session

    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}",
        )

    task_meta  = TASKS[task_id]
    difficulty = task_meta["difficulty"]
    episode_id = str(uuid.uuid4())

    first_email = _pick_email(difficulty, seen_ids=[])
    if first_email is None:
        raise HTTPException(status_code=500, detail="No emails available for this task.")

    _session = {
        "episode_id":   episode_id,
        "task_id":      task_id,
        "difficulty":   difficulty,
        "max_steps":    task_meta["max_steps"],
        "step_count":   0,
        "seen_ids":     [first_email["id"]],
        "current_email": first_email,
        "last_action":  None,
        "last_reward":  None,
        "done":         False,
        "cumulative_reward": 0.0,
    }

    obs = _email_to_observation(first_email, task_id)
    return obs.model_dump()


@app.post("/step")
def step(action: TriageAction) -> dict:
    """
    Submit a triage decision for the current email.
    Returns reward, done flag, grading breakdown, and the next email (if any).
    """
    global _session

    if not _session:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )

    if _session["done"]:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call /reset to start a new one.",
        )

    current_email = _session["current_email"]
    ground_truth  = current_email["ground_truth"]

    # ── Grade the action ──────────────────────────────────────────────────────
    grading = grade(action, ground_truth)
    reward  = grading["reward"]

    _session["step_count"]        += 1
    _session["last_action"]        = action
    _session["last_reward"]        = reward
    _session["cumulative_reward"] += reward

    # ── Check if episode should end ───────────────────────────────────────────
    next_email = _pick_email(_session["difficulty"], _session["seen_ids"])
    done       = (
        next_email is None
        or _session["step_count"] >= _session["max_steps"]
    )
    _session["done"] = done

    # ── Prepare next observation ──────────────────────────────────────────────
    if not done and next_email:
        _session["seen_ids"].append(next_email["id"])
        _session["current_email"] = next_email
        next_obs = _email_to_observation(next_email, _session["task_id"])
    else:
        # Return the same email observation on final step (episode over)
        next_obs = _email_to_observation(current_email, _session["task_id"])

    result = StepResult(
        observation=next_obs,
        reward=reward,
        done=done,
        info={
            "grading":            grading["breakdown"],
            "feedback":           grading["feedback"],
            "step":               _session["step_count"],
            "cumulative_reward":  round(_session["cumulative_reward"], 4),
            "episode_id":         _session["episode_id"],
            "graded_email_id":    current_email["id"],
        },
    )
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    """Return the full current environment state."""
    if not _session:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )

    s = _session
    env_state = TriageState(
        episode_id=s["episode_id"],
        task_id=s["task_id"],
        difficulty=s["difficulty"],
        current_email_id=s["current_email"]["id"] if s.get("current_email") else None,
        last_action=s["last_action"],
        last_reward=s["last_reward"],
        done=s["done"],
        step_count=s["step_count"],
    )
    return {
        **env_state.model_dump(),
        "cumulative_reward": round(s["cumulative_reward"], 4),
        "max_steps":         s["max_steps"],
        "emails_seen":       len(s["seen_ids"]),
    }
