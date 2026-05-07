"""Per-episode DAG runtime: agents, adversarial flags, execution."""

import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

from .graph_nodes import FinalDecision, Node, RoleAwareReasoner, SingleNodeAgent
from .utils import make_id



class RuntimeGraphMixin:
    def _clear_runtime(self) -> None:
        self.memory = {}
        self.nodes = {}
        self.graphs = []
        self.input_nodes = []
        self.output_nodes = []
        self.used_agents = []
        self.decision_method = FinalDecision(env=self)
        self.add_node(self.decision_method)
        self.output_nodes = [self.decision_method]

    def add_node(self, node: Node) -> Node:
        node_id = node.id if node.id is not None else make_id(4)
        while node_id in self.nodes:
            node_id = make_id(5)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def _sample_adversarial_agent_indices(self) -> List[int]:
        if not self.method_config.get("adversarial_attack_enabled", False):
            return []
        ratio = float(self.method_config.get("adversarial_attack_ratio", 0.2))
        count = int(self.num_truthful_agents * ratio)
        if count == 0 and self.num_truthful_agents > 0:
            count = 1
        return sorted(random.sample(list(range(self.num_truthful_agents)), count)) if count else []

    def _build_runtime(
        self,
        adjacency: List[List[int]],
        probe_mode: bool = False,
        forced_adversarial_indices: Optional[List[int]] = None,
    ) -> None:
        self._clear_runtime()
        for idx, role in enumerate(self.roles):
            node = RoleAwareReasoner(role_name=role, dataset_mode=self.dataset_mode, probe_mode=probe_mode, env=self)
            node.agent_index = idx + 1
            self.add_node(node)
            self.used_agents.append(SingleNodeAgent(node))

        for i in range(self.num_truthful_agents):
            for j in range(self.num_truthful_agents):
                if adjacency[i][j]:
                    src_node = next(iter(self.used_agents[i].nodes.values()))
                    dst_node = next(iter(self.used_agents[j].nodes.values()))
                    src_node.add_successor(dst_node)

        agent_nodes = [next(iter(agent.nodes.values())) for agent in self.used_agents]
        sinks = []
        for node in agent_nodes:
            if not any(succ is not self.decision_method for succ in node.successors):
                sinks.append(node)
        if not sinks:
            sinks = agent_nodes[-1:]
        for node in sinks:
            node.add_successor(self.decision_method)

        if bool(self.method_config.get("force-all-to-final-decision", False)):
            for node in agent_nodes:
                if self.decision_method not in node.successors:
                    node.add_successor(self.decision_method)

        self.input_nodes = [node for node in agent_nodes if len(node.predecessors) == 0]
        self._assign_adversarial_flags(agent_nodes, forced_indices=forced_adversarial_indices)

    def _assign_adversarial_flags(self, agent_nodes: List[Node], forced_indices: Optional[List[int]] = None) -> None:
        self.adversarial_node_ids = []
        self.current_adversarial_agent_indices = []
        for node in agent_nodes:
            node.is_adversarial = False
        if forced_indices is None:
            forced_indices = self._sample_adversarial_agent_indices()
        chosen = [agent_nodes[idx] for idx in forced_indices if 0 <= idx < len(agent_nodes)]
        for node in chosen:
            node.is_adversarial = True
        self.current_adversarial_agent_indices = [idx for idx in forced_indices if 0 <= idx < len(agent_nodes)]
        self.adversarial_node_ids = [node.id for node in chosen]

    def _execute_current_runtime(self, task: str) -> Tuple[str, List[Dict[str, Any]]]:
        def is_node_useful(node: Node) -> bool:
            if node in self.output_nodes:
                return True
            return any(is_node_useful(succ) for succ in node.successors)

        useful_node_ids = [node_id for node_id, node in self.nodes.items() if is_node_useful(node)]
        in_degree = {node_id: len(self.nodes[node_id].predecessors) for node_id in useful_node_ids}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]

        for input_node in self.input_nodes:
            input_node.inputs = [{"task": task, "files": []}]

        while queue:
            current_node_id = queue.pop(0)
            asyncio.run(self.nodes[current_node_id].execute())
            for successor in self.nodes[current_node_id].successors:
                if successor.id in in_degree:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        queue.append(successor.id)

        agent_outputs = []
        for idx, agent in enumerate(self.used_agents, start=1):
            node = next(iter(agent.nodes.values()))
            for output in node.outputs:
                agent_outputs.append(
                    {
                        "agent_index": idx,
                        "node_id": node.id,
                        "role": getattr(node, "role_name", ""),
                        "is_adversarial": bool(getattr(node, "is_adversarial", False)),
                        "output": output.get("output", output),
                    }
                )

        final_outputs = self.decision_method.outputs
        if final_outputs:
            response = str(final_outputs[-1].get("output", ""))
        elif agent_outputs:
            response = str(agent_outputs[-1]["output"])
        else:
            response = "No answer since there are no inputs provided"
        return response, agent_outputs

    def _make_probe_query(self, query: str) -> str:
        probe_schema = (
            'Return strict JSON only with keys: "claim", "tentative_answer", '
            '"evidence", "self_check", "possible_failure", "used_upstream".'
        )
        profile = self._build_task_profile(query)
        is_math = profile[0] > 0.5
        is_medmcqa = profile[1] > 0.5
        has_choice = profile[3] > 0.5
        has_digits = profile[5] > 0.5

        if is_medmcqa or has_choice:
            return (
                f"{query}\n\n"
                f"{probe_schema}\n"
                'For MedMCQA, include medical rationale, option elimination, and weak options in "evidence".'
            )
        if is_math and has_digits:
            return (
                f"{query}\n\n"
                f"{probe_schema}\n"
                'For math, include numeric constraints and one intermediate derivation in "evidence".'
            )
        if is_math:
            return (
                f"{query}\n\n"
                f"{probe_schema}\n"
                'For math, include symbolic relations and consistency checks in "self_check".'
            )
        return f"{query}\n\n{probe_schema}"
