"""Prototype bank, persistence, and topology matrix seeds."""

import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from . import prompts
from .utils import PrototypeRecord, normalize_text, safe_float



class PrototypeStateMixin:
    def _should_load_saved_optimization_state(self) -> bool:
        return bool(self.method_config.get("load-saved-optimization-state", False))

    def _should_persist_optimization_state(self) -> bool:
        return bool(self.method_config.get("persist-optimization-state", False))

    def _init_prototype_bank(self) -> List[PrototypeRecord]:
        n = self.num_truthful_agents
        return [
            PrototypeRecord("chain", self._chain_matrix(n), source="manual", note="Sequential pipeline."),
            PrototypeRecord("star", self._star_matrix(n), source="manual", note="Single coordinator broadcast."),
            PrototypeRecord("sparse_regular", self._sparse_regular_matrix(n), source="manual", note="Locally redundant sparse graph."),
            PrototypeRecord("modular", self._modular_matrix(n), source="manual", note="Clustered collaboration with bridge edge."),
            PrototypeRecord("diamond", self._diamond_matrix(n), source="manual", note="Parallel branches with reconvergent checking."),
            PrototypeRecord("dual_hub", self._dual_hub_matrix(n), source="manual", note="Overlapping early hubs for redundant broadcast."),
            PrototypeRecord("committee", self._committee_matrix(n), source="manual", note="Independent proposers routed to verifier and aggregator."),
            PrototypeRecord("auditor", self._auditor_matrix(n), source="manual", note="Chain backbone with final auditor visibility."),
        ]

    def _prototype_cache_path(self) -> str:
        return os.path.join(self.state_dir, f"{self.dataset_mode}_prototype_seed.json")

    def _sanitize_matrix(self, matrix: Any) -> Optional[List[List[float]]]:
        if not isinstance(matrix, list) or len(matrix) != self.num_truthful_agents:
            return None
        cleaned: List[List[float]] = []
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != self.num_truthful_agents:
                return None
            cleaned_row: List[float] = []
            for j, value in enumerate(row):
                weight = max(0.0, min(1.0, safe_float(value, 0.0)))
                if i == j or j <= i:
                    weight = 0.0
                cleaned_row.append(weight)
            cleaned.append(cleaned_row)
        if sum(sum(row) for row in cleaned) <= 0.0:
            return None
        return cleaned

    def _heuristic_generated_prototypes(self, count: int) -> List[PrototypeRecord]:
        n = self.num_truthful_agents
        heuristics: List[PrototypeRecord] = []

        ladder = [[0.0] * n for _ in range(n)]
        for i in range(n - 1):
            ladder[i][i + 1] = 1.0
            if i + 2 < n:
                ladder[i][i + 2] = 0.55
        heuristics.append(PrototypeRecord("llm_fallback_ladder", ladder, source="heuristic", note="Fallback ladder with skip edges."))

        reviewer_hub = [[0.0] * n for _ in range(n)]
        verifier_idx = min(n - 1, 3)
        synth_idx = min(n - 1, 4)
        for i in range(n):
            if i < verifier_idx:
                reviewer_hub[i][verifier_idx] = 0.9
            if i < synth_idx:
                reviewer_hub[i][synth_idx] = max(reviewer_hub[i][synth_idx], 0.7 if i != verifier_idx else 1.0)
        heuristics.append(PrototypeRecord("llm_fallback_review_hub", reviewer_hub, source="heuristic", note="Routes through verifier and synthesizer."))

        braided = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for step, weight in ((1, 0.9), (2, 0.45)):
                j = i + step
                if j < n:
                    braided[i][j] = max(braided[i][j], weight)
        heuristics.append(PrototypeRecord("llm_fallback_braided", braided, source="heuristic", note="Braided sparse redundancy."))

        expanded: List[PrototypeRecord] = []
        for idx in range(count):
            base = heuristics[idx % len(heuristics)]
            expanded.append(
                PrototypeRecord(
                    name=f"{base.name}_{idx + 1}",
                    matrix=deepcopy(base.matrix),
                    source=base.source,
                    note=base.note,
                )
            )
        return expanded[:count]

    def _parse_llm_prototype_response(self, response: str, expected_count: int) -> List[PrototypeRecord]:
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(response)
        except Exception:
            fenced = re.findall(r"```(?:json)?\n(.*?)\n```", response, re.DOTALL)
            if fenced:
                try:
                    parsed = json.loads(fenced[-1])
                except Exception:
                    parsed = {}
        prototypes = parsed.get("prototypes", []) if isinstance(parsed, dict) else []
        records: List[PrototypeRecord] = []
        for idx, item in enumerate(prototypes):
            if not isinstance(item, dict):
                continue
            matrix = self._sanitize_matrix(item.get("adjacency"))
            if matrix is None:
                continue
            name = normalize_text(item.get("name")) or f"llm_proto_{idx + 1}"
            reason = normalize_text(item.get("reason"))
            records.append(PrototypeRecord(name=name[:48], matrix=matrix, source="llm", note=reason))
            if len(records) >= expected_count:
                break
        return records

    def _generate_llm_prototypes(self, count: int) -> List[PrototypeRecord]:
        if count <= 0:
            return []
        prompt = prompts.ROBUST_TOPOLOGY_PROTOTYPE_PROMPT.format(
            dataset_mode=self.dataset_mode,
            num_agents=self.num_truthful_agents,
            roles=", ".join(self.roles),
            num_candidates=count,
        )
        try:
            response = self.call_llm(
                messages=[
                    {"role": "system", "content": "You design robust graph topologies and return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return self._parse_llm_prototype_response(response, count)
        except Exception:
            return []

    def _load_or_generate_llm_prototypes(self) -> List[PrototypeRecord]:
        count = int(self.method_config.get("llm-generated-prototype-count", 0))
        if count <= 0:
            return []
        cache_path = self._prototype_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cached_records = []
                for item in data.get("prototypes", []):
                    matrix = self._sanitize_matrix(item.get("matrix"))
                    if matrix is None:
                        continue
                    cached_records.append(
                        PrototypeRecord(
                            name=item.get("name", "cached_proto"),
                            matrix=matrix,
                            score=safe_float(item.get("score"), 0.0),
                            visits=safe_float(item.get("visits"), 0.0),
                            source=item.get("source", "llm"),
                            note=item.get("note", ""),
                        )
                    )
                if len(cached_records) >= count:
                    return cached_records[:count]
            except Exception:
                pass

        records = self._generate_llm_prototypes(count)
        if len(records) < count:
            fallback = self._heuristic_generated_prototypes(count - len(records))
            records.extend(fallback)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"prototypes": [record.__dict__ for record in records]}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return records[:count]

    def _chain_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n - 1):
            matrix[i][i + 1] = 1.0
        return matrix

    def _star_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        for j in range(1, n):
            matrix[0][j] = 1.0
        return matrix

    def _tree_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        for child in range(1, n):
            parent = (child - 1) // 2
            if parent < child:
                matrix[parent][child] = 1.0
        return matrix

    def _sparse_regular_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for step in (1, 2):
                j = i + step
                if j < n:
                    matrix[i][j] = 1.0
        return matrix

    def _modular_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        midpoint = max(1, n // 2)
        for start in (0, midpoint):
            end = midpoint if start == 0 else n
            for i in range(start, end):
                for j in range(i + 1, end):
                    matrix[i][j] = 1.0
        if midpoint < n:
            matrix[0][midpoint] = 1.0
        return matrix

    def _diamond_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        if n <= 1:
            return matrix
        sink = n - 1
        branch_heads = [idx for idx in (1, 2) if idx < n]
        merge = max(branch_heads[-1] if branch_heads else 0, n - 2)
        for head in branch_heads:
            matrix[0][head] = 1.0
            if head < merge:
                matrix[head][merge] = 1.0
            if head < sink:
                matrix[head][sink] = 1.0
        if merge < sink:
            matrix[merge][sink] = 1.0
        return matrix

    def _dual_hub_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        for hub in range(min(2, n)):
            for j in range(hub + 1, n):
                matrix[hub][j] = 1.0
        return matrix

    def _committee_matrix(self, n: int) -> List[List[float]]:
        matrix = [[0.0] * n for _ in range(n)]
        if n <= 1:
            return matrix
        sink = n - 1
        verifier = max(1, n - 2)
        for i in range(verifier):
            matrix[i][verifier] = 1.0
            if i < sink:
                matrix[i][sink] = 1.0
        if verifier < sink:
            matrix[verifier][sink] = 1.0
        return matrix

    def _auditor_matrix(self, n: int) -> List[List[float]]:
        matrix = self._chain_matrix(n)
        if n <= 1:
            return matrix
        sink = n - 1
        for i in range(sink):
            matrix[i][sink] = 1.0
        return matrix

    def _save_state(self) -> None:
        stats_path = os.path.join(self.state_dir, "episode_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.episode_stats, f, ensure_ascii=False, indent=2)
        if not self._should_persist_optimization_state():
            return
        # v1 fixed bank: do not persist mutable bank stats/state.
        model_path = os.path.join(self.state_dir, "hypernetwork.pt")
        torch.save(
            {
                "prototype_selector": self.prototype_selector.state_dict(),
                "risk_editor": self.risk_editor.state_dict(),
                "editor_hypernet": (
                    self.editor_hypernet.state_dict()
                    if getattr(self, "editor_hypernet", None) is not None
                    else None
                ),
                "descriptor_text_projector": self.descriptor_text_projector.state_dict(),
            },
            model_path,
        )

    def _load_state_if_available(self) -> None:
        if not self._should_load_saved_optimization_state():
            return
        device = self._get_descriptor_device()
        model_path = os.path.join(self.state_dir, "hypernetwork.pt")
        stats_path = os.path.join(self.state_dir, "episode_stats.json")
        # v1 fixed bank: ignore persisted prototype_bank.json even if present.
        if os.path.exists(model_path):
            payload = torch.load(model_path, map_location=device)
            if isinstance(payload, dict) and "prototype_selector" in payload:
                hn_state = payload["prototype_selector"]
                current = self.prototype_selector.state_dict()
                compatible = {k: v for k, v in hn_state.items() if k in current and current[k].shape == v.shape}
                if compatible:
                    current.update(compatible)
                    self.prototype_selector.load_state_dict(current)
                editor_state = payload.get("risk_editor")
                if isinstance(editor_state, dict):
                    current_editor = self.risk_editor.state_dict()
                    compatible_editor = {k: v for k, v in editor_state.items() if k in current_editor and current_editor[k].shape == v.shape}
                    if compatible_editor:
                        current_editor.update(compatible_editor)
                        self.risk_editor.load_state_dict(current_editor)
                hyper_state = payload.get("editor_hypernet")
                if isinstance(hyper_state, dict) and getattr(self, "editor_hypernet", None) is not None:
                    current_hyper = self.editor_hypernet.state_dict()
                    compatible_hyper = {k: v for k, v in hyper_state.items() if k in current_hyper and current_hyper[k].shape == v.shape}
                    if compatible_hyper:
                        current_hyper.update(compatible_hyper)
                        self.editor_hypernet.load_state_dict(current_hyper)
                proj_state = payload.get("descriptor_text_projector")
                if isinstance(proj_state, dict):
                    cur_p = self.descriptor_text_projector.state_dict()
                    ok = {k: v for k, v in proj_state.items() if k in cur_p and cur_p[k].shape == v.shape}
                    if ok:
                        cur_p.update(ok)
                        self.descriptor_text_projector.load_state_dict(cur_p)
            else:
                current_state = self.prototype_selector.state_dict()
                compatible_state = {}
                for key, value in payload.items():
                    if key in current_state and current_state[key].shape == value.shape:
                        compatible_state[key] = value
                if compatible_state:
                    current_state.update(compatible_state)
                    self.prototype_selector.load_state_dict(current_state)
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                loaded_stats = json.load(f)
            if isinstance(loaded_stats, list):
                self.episode_stats = loaded_stats
    def _update_prototype_bank(self, matrix: List[List[float]], score: float) -> None:
        # Fixed bank in v1: keep stable priors; no online replacement.
        return
