# 📧 Email Triage Environment

> An OpenEnv-compatible reinforcement learning environment where an AI agent learns to **prioritize**, **categorize**, and **route** real-world emails — just like a human support operations team would.

Built for the **Meta × Scaler OpenEnv Hackathon** by **Code Catalyst**.

---

## 🌍 Overview

In any organization, email triage is a critical but time-consuming task. Emails arrive with varying urgency, from billing disputes to legal threats to sales inquiries — and routing them to the wrong team costs time and money.

This environment trains an AI agent to make these decisions automatically and accurately, with partial-credit rewards that guide the agent toward better reasoning over time.

The agent interacts with the environment through the standard OpenEnv API (`reset()` / `step()` / `state()`), receives an email as an observation, submits a triage decision as an action, and gets a reward signal based on how accurately it triaged the email.

---

## 🗂️ Project Structure

```
email-triage-env/
├── main.py                  # FastAPI server (OpenEnv HTTP spec)
├── models.py                # Typed Action / Observation / State (Pydantic)
├── inference.py             # Baseline agent script
├── openenv.yaml             # OpenEnv spec file
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   └── emails.json          # 15 labeled emails with ground truth
└── graders/
    └── triage_grader.py     # Reward computation logic
```

---

## 🎯 Tasks

The environment exposes **3 tasks** of increasing difficulty, each using a different set of emails:

| Task ID | Difficulty | Description |
|---|---|---|
| `easy_triage` | 🟢 Easy | Emails with explicit urgency signals. Clear, unambiguous triage decisions. |
| `medium_triage` | 🟡 Medium | Emails with implicit urgency and overlapping categories. Requires reading between the lines. |
| `hard_triage` | 🔴 Hard | Multi-intent emails including legal/compliance edge cases, churn signals, and fraud reports. |

Each task runs for **up to 5 steps** (5 different emails). The agent must triage each email correctly to maximise cumulative reward.

---

## 👁️ Observation Space

Each observation is an email presented to the agent:

```json
{
  "email_id":    "easy_001",
  "subject":     "URGENT: Cannot access my account - payment due today",
  "body":        "Hi, I absolutely cannot log into my account...",
  "sender":      "john.doe@gmail.com",
  "timestamp":   "2024-03-15T09:00:00Z",
  "difficulty":  "easy",
  "task_id":     "easy_triage",
  "instruction": "Triage this email. Decide its priority, category, and which team should handle it."
}
```

---

## ⚡ Action Space

The agent submits a JSON triage decision:

```json
{
  "priority":           "urgent",
  "category":           "billing",
  "route":              "billing_support",
  "secondary_category": "technical_support",
  "secondary_route":    "tier2_support",
  "reasoning":          "Duplicate charge + explicit urgency = billing urgent"
}
```

### Valid values

**`priority`** — `urgent` | `normal` | `low`

**`category`** — `billing` | `technical_support` | `sales` | `hr` | `product_feedback` | `legal_compliance` | `other`

**`route`** — `billing_support` | `tier1_support` | `tier2_support` | `engineering_escalation` | `sales_team` | `sales_engineering` | `hr_recruitment` | `product_team` | `customer_success` | `customer_retention` | `legal_team` | `compliance_team` | `data_privacy_officer` | `partnerships_team`

`secondary_category`, `secondary_route`, and `reasoning` are **optional** — used for multi-intent emails (hard tasks).

---

## 🏆 Reward Function

Reward is computed per field with **partial credit**:

| Field | Weight | Notes |
|---|---|---|
| `priority` | **0.35** | Exact match required |
| `category` | **0.35** | Exact match required |
| `route` | **0.30** | Primary OR any acceptable alternate route |
| Secondary intent bonus | **+0.10** | Hard tasks only — catching secondary category + route |

**Total reward range: `0.0 – 1.0`** (capped at 1.0 including bonus)

Partial credit means the agent always receives a learning signal even when it doesn't get everything right. For example, correct priority + wrong category + wrong route = reward of `0.35`.

Medium and hard emails have `acceptable_routes` — multiple valid routing destinations — so the agent is not unfairly penalised for reasonable alternate decisions.

---

## 🔌 API Reference

All endpoints follow the OpenEnv HTTP spec.

### `GET /health`
Liveness probe. Always returns `200`.
```json
{ "status": "ok", "environment": "email-triage-env", "version": "1.0.0" }
```

### `GET /tasks`
List all available tasks.
```json
{
  "tasks": [
    { "task_id": "easy_triage", "difficulty": "easy", "max_steps": 5, "email_count": 5 },
    ...
  ]
}
```

### `POST /reset?task_id=easy_triage`
Start a new episode. Returns the first email observation.
```bash
curl -X POST "http://localhost:7860/reset?task_id=easy_triage"
```

### `POST /step`
Submit a triage action. Returns reward, done flag, and next email.
```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"priority": "urgent", "category": "billing", "route": "billing_support"}'
```

**Response:**
```json
{
  "observation": { ... },
  "reward": 1.0,
  "done": false,
  "info": {
    "grading": { "priority": 0.35, "category": 0.35, "route": 0.30, "secondary_bonus": 0.0 },
    "feedback": "✅ Priority correct. ✅ Category correct. ✅ Route correct.",
    "step": 1,
    "cumulative_reward": 1.0,
    "graded_email_id": "easy_001"
  }
}
```

### `GET /state`
Inspect current environment state.
```json
{
  "episode_id": "3f2a1b...",
  "task_id": "easy_triage",
  "difficulty": "easy",
  "step_count": 1,
  "last_reward": 1.0,
  "cumulative_reward": 1.0,
  "done": false
}
```

---

## 🚀 Setup & Running Locally

### Prerequisites
- Python 3.10+
- Docker (for containerised testing)
- A Hugging Face account + API token

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/email-triage-env.git
cd email-triage-env
pip install -r requirements.txt
```

### 2. Start the environment server

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Visit `http://localhost:7860/docs` for the interactive Swagger UI.

### 3. Run the baseline inference agent

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

### 4. Build and test with Docker

```bash
docker build -t email-triage-env .

docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token_here \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  email-triage-env
```

---

## 🤖 Baseline Agent

`inference.py` implements a zero-shot LLM agent using the **OpenAI client** pointed at the Hugging Face Inference API.

The agent:
1. Receives an email observation from `/reset`
2. Sends the email to an LLM with a structured system prompt
3. Parses the LLM's JSON triage decision
4. Submits it to `/step` and receives a reward
5. Repeats until the episode ends (`done: true`)

Logs follow the **strict `[START]` / `[STEP]` / `[END]` format** required by the hackathon evaluator.

### Sample output

```
{"log_type": "START", "task_id": "easy_triage", "episode_id": "3f2a1b..."}
{"log_type": "STEP", "task_id": "easy_triage", "step": 1, "email_id": "easy_001", "action": {"priority": "urgent", "category": "billing", "route": "billing_support"}, "reward": 1.0, "done": false, ...}
{"log_type": "END", "task_id": "easy_triage", "total_reward": 4.65, "steps": 5, "avg_reward": 0.93}
```

### Required environment variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (e.g. HF Inference API) |
| `MODEL_NAME` | Model identifier (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `HF_TOKEN` | Your Hugging Face API token |
| `ENV_URL` | URL of the running environment server |

---

## 📊 Evaluation Criteria

Submissions are evaluated on:

| Criterion | Details |
|---|---|
| **Runtime correctness** | Server starts, `/health` returns 200, `/reset` responds |
| **Interface compliance** | Valid `openenv.yaml`, typed models, all 3 endpoints work |
| **Task design** | 3 tasks with clear difficulty progression, realistic scenarios |
| **Grading logic** | Reward in `0.0–1.0`, partial credit, meaningful signal |
| **Baseline reproduces** | `inference.py` runs without error and produces scored output |

---

## 👥 Team

**Code Catalyst** — Meta × Scaler OpenEnv Hackathon 2026

- Ansh Mahajan *(Team Lead)*
- Deepak Singh Garakoti
- Devansh Tanwar

---
