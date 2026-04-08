"""
graders/triage_grader.py
Computes reward (0.0–1.0) by comparing the agent's TriageAction
against the ground-truth label stored in emails.json.

Reward breakdown:
  priority  → 0.35
  category  → 0.35
  route     → 0.30

For medium/hard emails, `acceptable_routes` lists additional valid routes.
For hard emails with secondary intents, bonus credit is awarded.

Secondary intent penalties:
  ignored (ground truth exists, agent gave None) → heavier penalty
  wrong   (agent guessed, but incorrectly)       → lighter penalty
"""

from __future__ import annotations
from typing import Any

from models import TriageAction


PRIORITY_WEIGHT = 0.35
CATEGORY_WEIGHT = 0.35
ROUTE_WEIGHT    = 0.30
SECONDARY_BONUS = 0.10   # extra reward for catching secondary intent on hard tasks

# ── Penalty tuning knobs ─────────────────────────────────────────────────────
SECONDARY_IGNORED_PENALTY  = 0.05   # agent didn't even try when GT exists
SECONDARY_WRONG_PENALTY    = 0.02   # agent tried but got it wrong


def grade(action: TriageAction, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a dict with:
        reward      – float 0.0–1.0
        breakdown   – per-field scores
        feedback    – human-readable explanation
    """
    breakdown: dict[str, float] = {}
    feedback:  list[str]        = []

    # ── Priority ─────────────────────────────────────────────────────────────
    correct_priority = ground_truth["priority"]
    if action.priority.value == correct_priority:
        breakdown["priority"] = PRIORITY_WEIGHT
        feedback.append(f"✅ Priority '{action.priority.value}' is correct. (+{PRIORITY_WEIGHT})")
    else:
        breakdown["priority"] = 0.0
        feedback.append(
            f"❌ Priority '{action.priority.value}' is wrong "
            f"(expected '{correct_priority}'). (+0.0)"
        )

    # ── Category ─────────────────────────────────────────────────────────────
    correct_category = ground_truth["category"]
    if action.category.value == correct_category:
        breakdown["category"] = CATEGORY_WEIGHT
        feedback.append(f"✅ Category '{action.category.value}' is correct. (+{CATEGORY_WEIGHT})")
    else:
        breakdown["category"] = 0.0
        feedback.append(
            f"❌ Category '{action.category.value}' is wrong "
            f"(expected '{correct_category}'). (+0.0)"
        )

    # ── Route ─────────────────────────────────────────────────────────────────
    correct_route       = ground_truth["route"]
    acceptable_routes   = ground_truth.get("acceptable_routes", [correct_route])
    if correct_route not in acceptable_routes:
        acceptable_routes = [correct_route] + acceptable_routes

    if action.route.value in acceptable_routes:
        breakdown["route"] = ROUTE_WEIGHT
        feedback.append(f"✅ Route '{action.route.value}' is correct. (+{ROUTE_WEIGHT})")
    else:
        breakdown["route"] = 0.0
        feedback.append(
            f"❌ Route '{action.route.value}' is wrong "
            f"(acceptable: {acceptable_routes}). (+0.0)"
        )

    # ── Secondary intent (bonus + penalties, hard tasks only) ─────────────────
    secondary_score    = 0.0
    secondary_feedback = []

    gt_has_secondary_cat   = bool(ground_truth.get("secondary_category"))
    gt_has_secondary_route = bool(ground_truth.get("secondary_route"))

    # — secondary category —
    if gt_has_secondary_cat:
        if action.secondary_category is None:
            # Agent ignored it entirely → heavier penalty
            secondary_score -= SECONDARY_IGNORED_PENALTY
            secondary_feedback.append(
                f"⚠️  Secondary category ignored (expected "
                f"'{ground_truth['secondary_category']}'). "
                f"(-{SECONDARY_IGNORED_PENALTY})"
            )
        elif action.secondary_category.value == ground_truth["secondary_category"]:
            # Correct
            secondary_score += SECONDARY_BONUS / 2
            secondary_feedback.append(
                f"✅ Secondary category '{action.secondary_category.value}' correct. "
                f"(+{SECONDARY_BONUS / 2})"
            )
        else:
            # Wrong guess → lighter penalty
            secondary_score -= SECONDARY_WRONG_PENALTY
            secondary_feedback.append(
                f"❌ Secondary category '{action.secondary_category.value}' wrong "
                f"(expected '{ground_truth['secondary_category']}'). "
                f"(-{SECONDARY_WRONG_PENALTY})"
            )

    # — secondary route —
    if gt_has_secondary_route:
        acceptable_secondary = ground_truth.get(
            "acceptable_secondary_routes",
            [ground_truth["secondary_route"]],
        )

        if action.secondary_route is None:
            # Agent ignored it entirely → heavier penalty
            secondary_score -= SECONDARY_IGNORED_PENALTY
            secondary_feedback.append(
                f"⚠️  Secondary route ignored (expected one of "
                f"{acceptable_secondary}). (-{SECONDARY_IGNORED_PENALTY})"
            )
        elif action.secondary_route.value in acceptable_secondary:
            # Correct
            secondary_score += SECONDARY_BONUS / 2
            secondary_feedback.append(
                f"✅ Secondary route '{action.secondary_route.value}' correct. "
                f"(+{SECONDARY_BONUS / 2})"
            )
        else:
            # Wrong guess → lighter penalty
            secondary_score -= SECONDARY_WRONG_PENALTY
            secondary_feedback.append(
                f"❌ Secondary route '{action.secondary_route.value}' wrong "
                f"(acceptable: {acceptable_secondary}). "
                f"(-{SECONDARY_WRONG_PENALTY})"
            )

    breakdown["secondary"] = secondary_score
    feedback.extend(secondary_feedback)

    # ── Total reward (strictly between 0 and 1, never 0.0 or 1.0) ────────────
    # The evaluator requires reward ∈ (0, 1) — both endpoints are rejected.
    # We clamp to [0.001, 0.999] to satisfy this constraint while preserving
    # the full signal range.
    raw_reward = sum(breakdown.values())
    reward     = min(max(round(raw_reward, 4), 0.001), 0.999)

    feedback.append(f"\n📊 Total reward: {reward:.4f} / 1.0")

    return {
        "reward":    reward,
        "breakdown": breakdown,
        "feedback":  "\n".join(feedback),
    }
