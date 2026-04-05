from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    urgent = "urgent"
    normal = "normal"
    low    = "low"

class Category(str, Enum):
    billing             = "billing"
    technical_support   = "technical_support"
    sales               = "sales"
    hr                  = "hr"
    product_feedback    = "product_feedback"
    legal_compliance    = "legal_compliance"
    other               = "other"

class Route(str, Enum):
    billing_support       = "billing_support"
    tier1_support         = "tier1_support"
    tier2_support         = "tier2_support"
    engineering_escalation = "engineering_escalation"
    sales_team            = "sales_team"
    sales_engineering     = "sales_engineering"
    hr_recruitment        = "hr_recruitment"
    product_team          = "product_team"
    customer_success      = "customer_success"
    customer_retention    = "customer_retention"
    legal_team            = "legal_team"
    compliance_team       = "compliance_team"
    data_privacy_officer  = "data_privacy_officer"
    partnerships_team     = "partnerships_team"


# ── Action (what the agent submits) ─────────────────────────────────────────

class TriageAction(BaseModel):
    """The agent's triage decision for the current email."""
    priority:           Priority          = Field(..., description="Urgency level: urgent | normal | low")
    category:           Category          = Field(..., description="Primary category of the email")
    route:              Route             = Field(..., description="Department/team to route the email to")
    secondary_category: Optional[Category] = Field(None, description="Secondary category for multi-intent emails")
    secondary_route:    Optional[Route]    = Field(None, description="Secondary route for multi-intent emails")
    reasoning:          Optional[str]      = Field(None, description="Agent's reasoning (used for partial credit on hard tasks)")


# ── Observation (what the agent receives) ────────────────────────────────────

class EmailObservation(BaseModel):
    """The email presented to the agent."""
    email_id:   str = Field(..., description="Unique email identifier")
    subject:    str = Field(..., description="Email subject line")
    body:       str = Field(..., description="Full email body")
    sender:     str = Field(..., description="Sender email address")
    timestamp:  str = Field(..., description="ISO 8601 timestamp")
    difficulty: str = Field(..., description="Task difficulty: easy | medium | hard")
    task_id:    str = Field(..., description="Current task name")
    instruction: str = Field(
        default=(
            "Triage this email. Decide its priority (urgent/normal/low), "
            "category, and which team/department should handle it. "
            "For complex emails, also identify a secondary_category and secondary_route."
        ),
        description="Agent instruction"
    )


# ── State (environment internal state) ───────────────────────────────────────

class TriageState(BaseModel):
    """Full environment state."""
    episode_id:       str               = Field(..., description="Current episode UUID")
    task_id:          str               = Field(..., description="Active task name")
    difficulty:       str               = Field(..., description="Current difficulty level")
    current_email_id: Optional[str]     = Field(None, description="ID of the email being triaged")
    last_action:      Optional[TriageAction] = Field(None, description="Agent's last submitted action")
    last_reward:      Optional[float]   = Field(None, description="Reward from the last step")
    done:             bool              = Field(False, description="Whether the episode is complete")
    step_count:       int               = Field(0, description="Number of steps taken")


# ── StepResult (returned by /step) ───────────────────────────────────────────

class StepResult(BaseModel):
    """Full response from a /step call."""
    observation: EmailObservation
    reward:      float = Field(..., description="Reward for this step (0.0 – 1.0)")
    done:        bool  = Field(..., description="True when the episode has ended")
    info:        dict  = Field(default_factory=dict, description="Grading breakdown and debug info")
