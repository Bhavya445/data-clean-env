---
title: Data Clean Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# 🧹 DataCleanEnv

An OpenEnv environment where an AI agent learns to clean dirty datasets.

## Overview

Real-world data is messy. DataCleanEnv challenges an agent to inspect tabular datasets
and identify data quality issues — missing values, wrong types, outliers, and inconsistencies —
then propose concrete fixes. This mirrors what every data analyst and data engineer does daily.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `null_hunter` | Easy | Find null values and duplicate rows in a 20-row dataset |
| `type_fixer` | Medium | Identify columns with wrong data types in a 15-row dataset |
| `full_audit` | Hard | Full data quality audit on a 20-row dataset with 5 issue types |

## Action & Observation Space

### Action (`DataCleanAction`)
```python
{
  "issues": ["column 'age' has 4 null values", "row 20 is a duplicate of row 1"],
  "fixes":  ["fill nulls in 'age' with median age", "drop duplicate row 20"]
}
```

### Observation (`DataCleanObservation`)
```python
{
  "dataset_csv": "id,name,age,...\n1,Alice,30,...\n...",
  "task_name": "null_hunter",
  "task_description": "Identify null values and duplicates...",
  "step_feedback": "Nulls: 75% | Duplicates: yes | Fix quality: 80%",
  "score_so_far": 0.72
}
```

## Reward Function

```
score = 0.40 * issue_detection
      + 0.30 * fix_correctness
      + 0.20 * precision (no hallucinations)
      + 0.10 * format compliance
```

Scores are always in range [0.0, 1.0] with partial credit throughout.

## Setup

```bash
# Install
pip install -e .

# Run server locally
uv run server
# or
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run with Docker
docker build -t data-clean-env -f server/Dockerfile .
docker run -p 7860:7860 data-clean-env
```

## Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

## Baseline Scores

| null_hunter | 0.867 |
| type_fixer  | 0.867 |
| full_audit  | 0.867 |
| **Average** | **0.867** |

*(Scores obtained with `Qwen/Qwen2.5-7B-Instruct`)*
