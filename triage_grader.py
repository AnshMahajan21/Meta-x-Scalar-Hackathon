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
"""

from __future__ import annotations
from typing import Any

from models import TriageAction


PRIORITY_WEIGHT = 0.35
CATEGORY_WEIGHT = 0.35
ROUTE_WEIGHT    = 0.30
SECONDARY_BONUS = 0.10   # extra reward for catching secondary intent on hard tasks


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
    # always include the primary correct route in the acceptable list
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

    # ── Secondary intent bonus (hard tasks only) ──────────────────────────────
    secondary_bonus = 0.0
    secondary_feedback = []

    if ground_truth.get("secondary_category") and action.secondary_category:
        if action.secondary_category.value == ground_truth["secondary_category"]:
            secondary_bonus += SECONDARY_BONUS / 2
            secondary_feedback.append(
                f"✅ Secondary category '{action.secondary_category.value}' correct. "
                f"(+{SECONDARY_BONUS / 2})"
            )

    if ground_truth.get("secondary_route") and action.secondary_route:
        acceptable_secondary = ground_truth.get(
            "acceptable_secondary_routes",
            [ground_truth["secondary_route"]]
        )
        if action.secondary_route.value in acceptable_secondary:
            secondary_bonus += SECONDARY_BONUS / 2
            secondary_feedback.append(
                f"✅ Secondary route '{action.secondary_route.value}' correct. "
                f"(+{SECONDARY_BONUS / 2})"
            )

    breakdown["secondary_bonus"] = secondary_bonus
    feedback.extend(secondary_feedback)

    # ── Total reward (capped at 1.0) ──────────────────────────────────────────
    raw_reward = sum(breakdown.values())
    reward     = min(round(raw_reward, 4), 1.0)

    feedback.append(f"\n📊 Total reward: {reward:.4f} / 1.0")

    return {
        "reward":    reward,
        "breakdown": breakdown,
        "feedback":  "\n".join(feedback),
    }
