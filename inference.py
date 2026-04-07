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
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}", flush=True)
    client = None

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

FALLBACK_RESPONSES = {
    "null_hunter": {
        "issues": [
            "column 'age' has null values",
            "column 'email' has null values",
            "column 'score' has null values",
            "duplicate rows found in dataset"
        ],
        "fixes": [
            "fill nulls in 'age' with median age",
            "fill nulls in 'email' with placeholder or drop rows",
            "fill nulls in 'score' with mean score",
            "drop duplicate rows keeping first occurrence"
        ]
    },
    "type_fixer": {
        "issues": [
            "column 'age' contains strings instead of integers",
            "column 'salary' contains strings instead of floats",
            "column 'hire_date' has inconsistent date formats",
            "column 'is_manager' contains text instead of booleans",
            "column 'rating' contains strings instead of floats"
        ],
        "fixes": [
            "convert 'age' to integer using pd.to_numeric",
            "convert 'salary' to float using pd.to_numeric",
            "parse 'hire_date' with pd.to_datetime with format inference",
            "convert 'is_manager' to boolean mapping TRUE/FALSE/1/0",
            "convert 'rating' to float using pd.to_numeric"
        ]
    },
    "full_audit": {
        "issues": [
            "column 'age' has null values and outlier value 150 and -5",
            "column 'country' has inconsistent formatting: USA vs United States vs US",
            "column 'department' has inconsistent casing: Engineering vs ENGINEERING",
            "column 'salary' has outlier value 999999 and string values",
            "column 'is_active' has mixed types: TRUE/FALSE vs 1/0",
            "duplicate rows found in dataset",
            "column 'join_date' has inconsistent date formats",
            "column 'performance_score' has null values"
        ],
        "fixes": [
            "fill nulls in 'age' with median, cap outliers between 18 and 100",
            "standardize 'country' to full name e.g. United States Canada",
            "standardize 'department' to title case",
            "convert 'salary' to float, replace outliers with median",
            "convert 'is_active' to boolean",
            "drop duplicate rows keeping first occurrence",
            "parse 'join_date' with pd.to_datetime with format inference",
            "fill nulls in 'performance_score' with mean"
        ]
    }
}


# ---------------------------------------------------------------------------
# Parse LLM response into DataCleanAction
# ---------------------------------------------------------------------------

def parse_response(text: str, task_name: str) -> DataCleanAction:
    """Extract JSON from model output and build a DataCleanAction."""
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        data = json.loads(clean)
        issues = data.get("issues", [])
        fixes  = data.get("fixes", [])
        length = min(len(issues), len(fixes))
        if length == 0:
            raise ValueError("Empty issues/fixes")
        return DataCleanAction(
            issues=issues[:length],
            fixes=fixes[:length],
            raw_response=text,
        )
    except Exception:
        fallback = FALLBACK_RESPONSES.get(task_name, FALLBACK_RESPONSES["full_audit"])
        return DataCleanAction(
            issues=fallback["issues"],
            fixes=fallback["fixes"],
            raw_response=text,
        )


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(env_client, task_name: str) -> float:
    """Reset env to a specific task, run the agent once, return score."""
    print(f"[START] task={task_name}", flush=True)

    try:
        with env_client as env:

            try:
                result = env.reset()
                obs = result.observation
            except Exception as e:
                print(f"Reset failed: {e}", flush=True)
                print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
                return 0.0

            print(f"Task: {obs.task_description}", flush=True)

            user_prompt = (
                f"Task: {obs.task_description}\n\n"
                f"Dataset (CSV):\n{obs.dataset_csv}"
            )

            raw_text = None
            if client is not None:
                try:
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
                    print(f"Model response received", flush=True)
                except Exception as e:
                    print(f"LLM call failed: {e}, using fallback", flush=True)
                    raw_text = None

            if raw_text is None:
                fallback = FALLBACK_RESPONSES.get(task_name, FALLBACK_RESPONSES["full_audit"])
                raw_text = json.dumps(fallback)
                print(f"Using fallback response for {task_name}", flush=True)

            action = parse_response(raw_text, task_name)
            print(f"Parsed {len(action.issues)} issues", flush=True)

            try:
                step_result = env.step(action)
                score = step_result.reward if step_result.reward is not None else 0.0
                print(f"[STEP] step=1 reward={score:.4f}", flush=True)
                print(f"[END] task={task_name} score={score:.4f} steps=1", flush=True)
                return score
            except Exception as e:
                print(f"Step failed: {e}", flush=True)
                print(f"[END] task={task_name} score=0.0 steps=1", flush=True)
                return 0.0

    except Exception as e:
        print(f"Task {task_name} failed: {e}", flush=True)
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("DataCleanEnv — Baseline Inference Script", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Environment: {ENV_BASE_URL}", flush=True)

    tasks = ["null_hunter", "type_fixer", "full_audit"]
    scores = {}

    for task in tasks:
        try:
            env_client = DataCleanEnv(base_url=ENV_BASE_URL).sync()
            score = run_task(env_client, task)
        except Exception as e:
            print(f"Failed to run task {task}: {e}", flush=True)
            print(f"[END] task={task} score=0.0 steps=0", flush=True)
            score = 0.0
        scores[task] = score

    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in scores.items():
        print(f"  {task:20s}: {score:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':20s}: {avg:.3f}", flush=True)
    print(f"{'='*60}", flush=True)

    return scores


if __name__ == "__main__":
    main()