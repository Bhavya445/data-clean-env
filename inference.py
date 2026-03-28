"""
Inference Script — DataCleanEnv
================================
Runs a baseline LLM agent against all 3 tasks and reports scores.

Required environment variables:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face / API key
"""

import os
import json
import re
from openai import OpenAI
from client import DataCleanEnv
from models import DataCleanAction
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """
You are an expert data quality analyst. You will be given a CSV dataset and a task description.
Your job is to identify data quality issues and propose fixes.

You MUST respond in valid JSON with exactly this structure:
{
  "issues": ["issue 1 description", "issue 2 description", ...],
  "fixes":  ["fix for issue 1",     "fix for issue 2",     ...]
}

Rules:
- issues and fixes must have the same length
- Be specific: name the exact column and the exact problem
- Do not invent issues that don't exist in the data
- Respond with JSON only — no extra text, no markdown fences
"""

# ---------------------------------------------------------------------------
# Parse LLM response into DataCleanAction
# ---------------------------------------------------------------------------

def parse_response(text: str) -> DataCleanAction:
    """Extract JSON from model output and build a DataCleanAction."""
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        data = json.loads(clean)
        issues = data.get("issues", [])
        fixes  = data.get("fixes", [])
        # Ensure equal length
        length = min(len(issues), len(fixes))
        return DataCleanAction(
            issues=issues[:length],
            fixes=fixes[:length],
            raw_response=text,
        )
    except (json.JSONDecodeError, KeyError):
        # Fallback: return empty action if parsing fails
        return DataCleanAction(issues=[], fixes=[], raw_response=text)


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(env_client, task_name: str) -> float:
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}")

    with env_client as env:
        result = env.reset()
        obs = result.observation

        print(f"Task description:\n  {obs.task_description}\n")
        print(f"Dataset preview (first 5 lines):")
        for line in obs.dataset_csv.strip().split("\n")[:6]:
            print(f"  {line}")
        print("  ...")

        # Build prompt
        user_prompt = (
            f"Task: {obs.task_description}\n\n"
            f"Dataset (CSV):\n{obs.dataset_csv}"
        )

        # Call LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        raw_text = response.choices[0].message.content
        print(f"\nModel response (truncated):\n  {raw_text[:300]}...")

        # Parse into action
        action = parse_response(raw_text)
        print(f"\nParsed issues ({len(action.issues)}):")
        for i, issue in enumerate(action.issues[:5]):
            print(f"  {i+1}. {issue}")

        # Step environment
        step_result = env.step(action)
        score = step_result.reward

        print(f"\nFeedback: {step_result.observation.step_feedback}")
        print(f"Score: {score:.3f}")

        return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("DataCleanEnv — Baseline Inference Script")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")

    tasks = ["null_hunter", "type_fixer", "full_audit"]
    scores = {}

    for task in tasks:
        env_client = DataCleanEnv(base_url=ENV_BASE_URL).sync()
        score = run_task(env_client, task)
        scores[task] = score

    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print(f"{'='*60}")
    for task, score in scores.items():
        print(f"  {task:20s}: {score:.3f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':20s}: {avg:.3f}")
    print(f"{'='*60}")

    return scores


if __name__ == "__main__":
    main()
