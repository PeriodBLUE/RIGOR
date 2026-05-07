"""Dataset -> role pool -> tool permissions registry."""

from dataclasses import dataclass
from itertools import cycle
from typing import Dict, List


@dataclass
class DatasetRoleToolSpec:
    role_pool: List[str]
    role_tools: Dict[str, List[str]]

    def build_roles(self, num_agents: int) -> List[str]:
        if not self.role_pool:
            return []
        stream = cycle(self.role_pool)
        return [next(stream) for _ in range(num_agents)]


DEFAULT_ROLE_POOL = ["planner", "solver", "critic", "verifier", "synthesizer"]


DATASET_ROLE_TOOL_SPECS: Dict[str, DatasetRoleToolSpec] = {
    "math": DatasetRoleToolSpec(
        role_pool=["planner", "solver", "critic", "verifier", "synthesizer"],
        role_tools={
            "planner": ["math_helper"],
            "solver": ["python_eval", "math_helper"],
            "critic": ["python_eval", "math_helper"],
            "verifier": ["python_eval", "math_helper"],
            "synthesizer": ["python_eval", "math_helper"],
        },
    ),
    "medmcqa": DatasetRoleToolSpec(
        role_pool=["knowledge_expert", "option_analyst", "fact_checker", "critic", "synthesizer"],
        role_tools={
            "knowledge_expert": ["wiki_search"],
            "option_analyst": [],
            "fact_checker": ["wiki_search"],
            "critic": [],
            "synthesizer": ["wiki_search"],
        },
    ),
}
