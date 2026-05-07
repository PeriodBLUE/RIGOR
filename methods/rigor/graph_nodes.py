import asyncio
import re
from typing import Any, Dict, List, Optional, Sequence

from . import prompts
from .prompt import (
    MATH_FINAL_PROMPT,
    MATH_PROBE_PROMPT,
    MATH_SYSTEM_PROMPT,
    MEDMCQA_SYSTEM_PROMPT,
)
from .role_tooling import parse_tool_call
from .utils import (
    extract_choice_letter,
    extract_first_json_object,
    extract_last_number,
    make_id,
    normalize_text,
)


class Node:
    def __init__(self, operation_description: str, id: Optional[str], combine_inputs_as_one: bool, env: Any):
        self.id = id if id is not None else make_id(4)
        self.operation_description = operation_description
        self.predecessors: List["Node"] = []
        self.successors: List["Node"] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.combine_inputs_as_one = combine_inputs_as_one
        self.env = env
        self.is_adversarial = False

    def add_predecessor(self, operation: "Node"):
        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.successors.append(self)

    def add_successor(self, operation: "Node"):
        if operation not in self.successors:
            self.successors.append(operation)
            operation.predecessors.append(self)

    def process_input(self, inputs):
        if inputs is None:
            return []
        if not isinstance(inputs, list):
            return [inputs]
        return inputs

    async def execute(self, **kwargs):
        self.outputs = []
        tasks = []
        if not self.inputs and self.predecessors:
            if self.combine_inputs_as_one:
                combined_inputs = []
                for predecessor in self.predecessors:
                    if predecessor.outputs:
                        combined_inputs.extend(predecessor.outputs)
                tasks.append(asyncio.create_task(self._execute(combined_inputs, **kwargs)))
            else:
                for predecessor in self.predecessors:
                    for predecessor_output in predecessor.outputs:
                        tasks.append(asyncio.create_task(self._execute(predecessor_output, **kwargs)))
        elif self.inputs:
            tasks = [asyncio.create_task(self._execute(item, **kwargs)) for item in self.inputs]
        else:
            return

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    continue
                if not isinstance(result, list):
                    result = [result]
                self.outputs.extend(result)


class FinalDecision(Node):
    def __init__(self, operation_description: str = "Refer to all answers and give a final answer.", id=None, env=None):
        super().__init__(operation_description, id, True, env)

    @staticmethod
    def _normalize_math_expr(text: str) -> str:
        value = normalize_text(text)
        if not value:
            return ""
        boxed = re.findall(r"\\boxed\{(.+?)\}", value)
        if boxed:
            value = boxed[-1]
        final_matches = re.findall(
            r"(?:final\s*answer|the\s+answer\s+is)\s*[:\s]\s*([^\n,.;()]+)",
            value,
            flags=re.IGNORECASE,
        )
        if final_matches:
            value = final_matches[-1]
        last_number = extract_last_number(value)
        return last_number or re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _vote_key(dataset_mode: str, text: str) -> str:
        if dataset_mode == "medmcqa":
            return extract_choice_letter(text) or normalize_text(text)
        return FinalDecision._normalize_math_expr(text)

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        dataset_mode = getattr(self.env, "dataset_mode", "")
        if dataset_mode not in ("math", "medmcqa"):
            raise ValueError(f"Unsupported RIGOR dataset_mode: {dataset_mode}")

        agent_risks: List[float] = list(getattr(self.env, "_latest_agent_risks", None) or [])

        def vote_weight(agent_index: Any) -> float:
            if agent_index is None or not agent_risks:
                return 1.0
            try:
                idx0 = int(agent_index) - 1
            except Exception:
                return 1.0
            if idx0 < 0 or idx0 >= len(agent_risks):
                return 1.0
            risk = float(agent_risks[idx0])
            return max(0.05, min(1.0, 1.0 - max(0.0, min(1.0, risk))))

        role_priority = {
            "synthesizer": 0,
            "solver": 1,
            "verifier": 2,
            "knowledge_expert": 0,
            "option_analyst": 1,
            "fact_checker": 2,
            "critic": 3,
            "planner": 4,
        }
        buckets: Dict[str, Dict[str, Any]] = {}
        task = ""
        for item in node_inputs:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task", task))
            raw = str(item.get("output", "")).strip()
            key = self._vote_key(dataset_mode, raw)
            if not key:
                continue
            agent_index = item.get("agent_index")
            role = str(item.get("role", ""))
            weight = vote_weight(agent_index)
            bucket = buckets.setdefault(
                key,
                {
                    "count": 0,
                    "weighted_count": 0.0,
                    "raw": raw,
                    "best_role_prio": role_priority.get(role, 99),
                    "raw_trust": weight,
                    "agent_indices": [],
                    "node_ids": [],
                    "roles": [],
                },
            )
            bucket["count"] += 1
            bucket["weighted_count"] += weight
            if weight > bucket["raw_trust"] + 1e-9 or (
                abs(weight - bucket["raw_trust"]) < 1e-9
                and role_priority.get(role, 99) < bucket["best_role_prio"]
            ):
                bucket["raw"] = raw
                bucket["raw_trust"] = weight
                bucket["best_role_prio"] = role_priority.get(role, 99)
            if agent_index is not None:
                bucket["agent_indices"].append(int(agent_index))
            node_id = item.get("node_id")
            if node_id:
                bucket["node_ids"].append(str(node_id))
            if role:
                bucket["roles"].append(role)

        if buckets:
            chosen_key, chosen = sorted(
                buckets.items(),
                key=lambda item: (
                    -item[1]["count"],
                    -item[1]["weighted_count"],
                    item[1]["best_role_prio"],
                    item[0],
                ),
            )[0]
            response = chosen["raw"]
            if dataset_mode == "medmcqa":
                letter = extract_choice_letter(chosen_key) or extract_choice_letter(response)
                if letter:
                    response = f"The answer is ({letter})"
            selected_agent_indices = sorted(set(chosen.get("agent_indices", [])))
            selected_node_ids = sorted(set(chosen.get("node_ids", [])))
            selected_roles = sorted(set(chosen.get("roles", [])))
        else:
            chosen_key = ""
            response = ""
            selected_agent_indices = []
            selected_node_ids = []
            selected_roles = []

        execution = {
            "task": task,
            "files": [],
            "input": node_inputs,
            "output": response,
            "format": "natural language",
            "selected_vote_key": chosen_key,
            "selected_agent_indices": selected_agent_indices,
            "selected_node_ids": selected_node_ids,
            "selected_roles": selected_roles,
            "selected_agent_count": len(selected_agent_indices),
            "aggregation_meta": {},
        }
        if self.id not in self.env.memory:
            self.env.memory[self.id] = []
        self.env.memory[self.id].append(execution)
        return execution


class RoleAwareReasoner(Node):
    def __init__(self, role_name: str, dataset_mode: str, probe_mode: bool, env: Any, id: Optional[str] = None):
        super().__init__(f"{role_name} reasoner", id, True, env)
        self.role_name = role_name
        self.dataset_mode = dataset_mode
        self.probe_mode = probe_mode

    def _build_system_prompt(self) -> str:
        if self.dataset_mode == "medmcqa":
            base = MEDMCQA_SYSTEM_PROMPT
        elif self.dataset_mode == "math":
            base = MATH_PROBE_PROMPT if self.probe_mode else MATH_FINAL_PROMPT
        else:
            raise ValueError(f"Unsupported RIGOR dataset_mode: {self.dataset_mode}")

        role_text = prompts.ROLE_DESCRIPTIONS.get(self.role_name, "")
        rollout_text = prompts.PROBING_INSTRUCTION if self.probe_mode else prompts.FINAL_INSTRUCTION
        role_block = f"You are now playing this role.\nRole: {self.role_name}\nResponsibility: {role_text}"
        role_prompt = f"{base.strip()}\n\n{role_block}\n\n{rollout_text.strip()}"
        if self.dataset_mode == "math":
            role_prompt = f"{MATH_SYSTEM_PROMPT.strip()}\n\n{role_prompt}"
        if self.is_adversarial:
            return (
                f"{prompts.ADVERSARIAL_SYSTEM_PROMPT.strip()}\n\n"
                f"{role_block}\n\n"
                f"{role_prompt}"
            )
        return role_prompt

    def _tool_instruction_block(self) -> str:
        tool_names = []
        if hasattr(self.env, "get_tools_for_role"):
            tool_names = self.env.get_tools_for_role(self.role_name)
        if not tool_names:
            return ""
        lines = ["Available tools for your role:"]
        for name in tool_names:
            desc = self.env.describe_tool(name) if hasattr(self.env, "describe_tool") else name
            lines.append(f"- {name}: {desc}")
        lines.append(
            'If needed, return ONLY one tool call JSON wrapped in <tool_call>...</tool_call>, '
            'for example: <tool_call>{"tool":"python_eval","args":{"expression":"12*(3+1)"}}</tool_call>'
        )
        lines.append("If no tool is needed, answer normally.")
        return "\n".join(lines)

    def _maybe_parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        parsed = parse_tool_call(response)
        if parsed:
            return parsed
        candidate = extract_first_json_object(response)
        if not candidate:
            return None
        return parse_tool_call(candidate)

    def _build_user_prompt(self, task: str, upstream_messages: Sequence[str]) -> str:
        upstream_block = "\n".join(
            f"- Upstream {idx + 1}: {message}"
            for idx, message in enumerate(upstream_messages)
            if normalize_text(message)
        )
        if not upstream_block:
            upstream_block = "- No upstream messages."

        if self.dataset_mode == "medmcqa":
            return (
                f"{task}\n\n"
                f"Upstream signals (optional):\n{upstream_block}\n\n"
                "Think step by step from medical evidence and option elimination, then state your final answer letter."
            )
        if self.dataset_mode == "math":
            if self.probe_mode:
                return (
                    f"Task:\n{task}\n\n"
                    f"Upstream messages:\n{upstream_block}\n\n"
                    "Return one concise intermediate insight only; do not give the final answer."
                )
            return (
                f"Task:\n{task}\n\n"
                f"Upstream messages:\n{upstream_block}\n\n"
                "Solve robustly with essential steps and end with exactly one line:\n"
                "Final answer: <value or expression>"
            )
        raise ValueError(f"Unsupported RIGOR dataset_mode: {self.dataset_mode}")

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        outputs = []
        task = ""
        upstream_messages: List[str] = []

        for item in node_inputs:
            if isinstance(item, dict):
                task = item.get("task", task)
                upstream_output = item.get("output")
                if upstream_output is not None:
                    upstream_messages.append(str(upstream_output))
            else:
                upstream_messages.append(str(item))

        task = task or kwargs.get("task", "")
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task, upstream_messages)
        tool_block = self._tool_instruction_block()
        if tool_block:
            system_prompt = f"{system_prompt}\n\n{tool_block}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.env.call_llm(messages=messages)
        tool_call = self._maybe_parse_tool_call(response)
        if tool_call and hasattr(self.env, "run_tool"):
            tool_output = self.env.run_tool(
                tool_call.get("tool", ""),
                tool_call.get("args", {}),
                context={
                    "task": task,
                    "role": self.role_name,
                    "dataset_mode": self.dataset_mode,
                    "upstream_messages": list(upstream_messages),
                },
            )
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool `{tool_call.get('tool', '')}` returned:\n{tool_output}\n\n"
                        "Now provide your final answer directly."
                    ),
                }
            )
            response = self.env.call_llm(messages=messages)

        execution = {
            "task": task,
            "files": [],
            "input": user_prompt,
            "output": response,
            "format": "natural language",
            "role": self.role_name,
            "agent_index": getattr(self, "agent_index", None),
            "node_id": self.id,
            "probe_mode": self.probe_mode,
        }
        if self.id not in self.env.memory:
            self.env.memory[self.id] = []
        self.env.memory[self.id].append(execution)
        outputs.append(execution)
        return outputs


class SingleNodeAgent:
    def __init__(self, node: Node):
        self.nodes = {node.id: node}
