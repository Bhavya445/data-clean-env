import os
import uuid
from typing import Optional, Any
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import DataCleanAction, DataCleanObservation

TASKS = {
    "null_hunter": {
        "description": (
            "Inspect the dataset and identify: (1) which columns contain null/missing values "
            "and how many nulls each has, and (2) any duplicate rows. "
            "For each issue found, propose a fix."
        ),
        "dataset_file": "easy.csv",
    },
    "type_fixer": {
        "description": (
            "Inspect the dataset and identify columns with incorrect data types "
            "(e.g. numbers stored as strings, dates in wrong format, booleans as text). "
            "For each column, state the current type, the correct type, and how to fix it."
        ),
        "dataset_file": "medium.csv",
    },
    "full_audit": {
        "description": (
            "Perform a complete data quality audit. Identify ALL of the following issue types: "
            "null/missing values, duplicate rows, wrong data types, outliers, and inconsistent "
            "formatting (e.g. 'USA' vs 'United States'). For every issue, propose a concrete fix."
        ),
        "dataset_file": "hard.csv",
    },
}

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TASK_ORDER = list(TASKS.keys())

GROUND_TRUTH = {
    "null_hunter": {"null_columns": {"age", "email", "score"}, "has_duplicates": True},
    "type_fixer": {"type_issues": {"age", "salary", "hire_date", "is_manager", "rating"}},
    "full_audit": {
        "null_columns": {"age", "country", "department", "salary", "performance_score"},
        "has_duplicates": True,
        "type_issues": {"salary", "join_date", "is_active"},
        "outlier_columns": {"age", "salary"},
        "inconsistent_columns": {"country", "department", "is_active"},
    },
}


def load_dataset(filename):
    with open(os.path.join(DATASETS_DIR, filename)) as f:
        return f.read()


def _kw(text, keywords):
    t = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in t)
    return hits / len(keywords) if keywords else 0.0


def grade_null_hunter(action):
    txt = " ".join(action.issues + action.fixes).lower()
    null_score = _kw(txt, list(GROUND_TRUTH["null_hunter"]["null_columns"]))
    dup_score = 1.0 if any("duplic" in i.lower() for i in action.issues) else 0.0
    fix_kw = ["fill", "drop", "remove", "impute", "median", "mean", "delete"]
    fix_score = min(1.0, sum(1 for f in action.fixes if any(k in f.lower() for k in fix_kw)) / 4)
    prec = 1.0 if 1 <= len(action.issues) <= 20 else 0.0
    score = round(0.40*null_score + 0.25*dup_score + 0.25*fix_score + 0.10*prec, 3)
    return score, f"Nulls: {null_score:.0%} | Dups: {'yes' if dup_score else 'no'} | Fixes: {fix_score:.0%}"


def grade_type_fixer(action):
    txt = " ".join(action.issues + action.fixes).lower()
    type_score = _kw(txt, list(GROUND_TRUTH["type_fixer"]["type_issues"]))
    fix_kw = ["cast", "convert", "parse", "int", "float", "datetime", "bool", "to_"]
    fix_score = min(1.0, sum(1 for f in action.fixes if any(k in f.lower() for k in fix_kw)) / 5)
    prec = 1.0 if 3 <= len(action.issues) <= 15 else 0.5
    score = round(0.50*type_score + 0.35*fix_score + 0.15*prec, 3)
    return score, f"Types: {type_score:.0%} | Fixes: {fix_score:.0%}"


def grade_full_audit(action):
    gt = GROUND_TRUTH["full_audit"]
    txt = " ".join(action.issues + action.fixes).lower()
    null_s = _kw(txt, list(gt["null_columns"]))
    dup_s = 1.0 if any("duplic" in i.lower() for i in action.issues) else 0.0
    type_s = _kw(txt, list(gt["type_issues"]))
    out_s = min(1.0, sum(1 for k in ["outlier","150","-5","999999"] if k in txt) / 3)
    inc_s = _kw(txt, list(gt["inconsistent_columns"]))
    fix_kw = ["fill","drop","cast","convert","standardize","normalize","replace","median"]
    fix_s = min(1.0, sum(1 for f in action.fixes if any(k in f.lower() for k in fix_kw)) / 8)
    score = round(0.20*null_s + 0.15*dup_s + 0.20*type_s + 0.20*out_s + 0.15*inc_s + 0.10*fix_s, 3)
    return score, f"Nulls: {null_s:.0%} | Types: {type_s:.0%} | Outliers: {out_s:.0%} | Inconsistencies: {inc_s:.0%}"


GRADERS = {"null_hunter": grade_null_hunter, "type_fixer": grade_type_fixer, "full_audit": grade_full_audit}


class DataCleanEnvironment(Environment):

    def __init__(self, transform=None, rubric=None):
        super().__init__(transform=transform, rubric=rubric)
        self._task_name = None
        self._dataset_csv = ""
        self._episode_count = 0
        self._current_state = State(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> DataCleanObservation:
        self._episode_count += 1
        self._task_name = TASK_ORDER[(self._episode_count - 1) % len(TASK_ORDER)]
        self._dataset_csv = load_dataset(TASKS[self._task_name]["dataset_file"])
        self._current_state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0
        )
        return DataCleanObservation(
            dataset_csv=self._dataset_csv,
            task_name=self._task_name,
            task_description=TASKS[self._task_name]["description"],
            step_feedback="",
            score_so_far=0.0,
            done=False,
            reward=None,
        )

    def step(self, action: DataCleanAction, timeout_s=None, **kwargs) -> DataCleanObservation:
        self._current_state.step_count += 1
        score, feedback = GRADERS[self._task_name](action)
        return DataCleanObservation(
            dataset_csv=self._dataset_csv,
            task_name=self._task_name,
            task_description=TASKS[self._task_name]["description"],
            step_feedback=feedback,
            score_so_far=score,
            done=True,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._current_state