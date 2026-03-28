from typing import Optional, List
from pydantic import Field, ConfigDict
from openenv.core.env_server.types import Action, Observation


class DataCleanAction(Action):
    model_config = ConfigDict(extra="allow")
    issues: List[str] = Field(default_factory=list)
    fixes: List[str] = Field(default_factory=list)
    raw_response: Optional[str] = Field(default=None)


class DataCleanObservation(Observation):
    model_config = ConfigDict(extra="allow")
    dataset_csv: str = Field(default="")
    task_name: str = Field(default="")
    task_description: str = Field(default="")
    step_feedback: str = Field(default="")
    score_so_far: float = Field(default=0.0)