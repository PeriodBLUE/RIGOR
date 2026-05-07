"""Hypernetwork outputs, topology edits, and structural graph metrics."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import normalize_text, safe_float



class HyperTopologyMixin:
    def _select_prior_top1(self, selector_logits: torch.Tensor) -> Tuple[List[List[float]], int, torch.Tensor]:
        select_mode = str(self.method_config.get("prototype-select-mode", "top1")).strip().lower()
        if select_mode != "top1":
            raise ValueError(f"Unsupported prototype-select-mode: {select_mode}. v1 only supports top1.")
        probs = torch.softmax(selector_logits, dim=-1)
        proto_idx = int(torch.argmax(selector_logits).item())
        alpha = F.one_hot(torch.tensor(proto_idx, device=selector_logits.device), num_classes=len(self.prototype_bank)).float()
        log_prob = torch.log(torch.clamp(probs[proto_idx], min=1e-6))
        return deepcopy(self.prototype_bank[proto_idx].matrix), proto_idx, log_prob

    def _apply_residual_edits(
        self,
        prior_matrix: List[List[float]],
        editor_outputs: Dict[str, torch.Tensor],
        risks: List[float],
        uncertainties: List[float],
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        matrix = deepcopy(prior_matrix)
        quarantine_logits = editor_outputs["quarantine_logits"]
        mask_logits = editor_outputs["mask_logits"]
        quarantine_scores = torch.sigmoid(quarantine_logits)
        mask_scores = torch.sigmoid(mask_logits)
        tau_q = float(self.method_config.get("tau-quarantine", 0.55))
        tau_m = float(self.method_config.get("tau-mask", 0.60))
        max_q = int(self.method_config.get("max-quarantine", 1))
        max_m = int(self.method_config.get("max-mask", 1))
        quarantine_strength = float(self.method_config.get("quarantine-strength", 0.75))
        applied = []
        quarantined_nodes = set()
        policy_log_prob = None
        edit_info_extra = {}
        stochastic_actions = bool(getattr(self, "_stochastic_editor_actions", False))

        def _threshold_shifted_prob(logit: torch.Tensor, threshold: float) -> torch.Tensor:
            threshold_t = torch.tensor(threshold, device=logit.device, dtype=logit.dtype)
            threshold_t = torch.clamp(threshold_t, min=1e-6, max=1.0 - 1e-6)
            return torch.sigmoid(logit - torch.logit(threshold_t))

        if max_q > 0 and quarantine_scores.numel() > 0:
            if stochastic_actions:
                q_dist = torch.distributions.Categorical(logits=quarantine_logits.view(-1))
                q_sample = q_dist.sample()
                q_idx = int(q_sample.item())
                q_activate_prob = _threshold_shifted_prob(quarantine_logits[q_idx], tau_q)
                q_activate_dist = torch.distributions.Bernoulli(
                    probs=torch.clamp(q_activate_prob, min=1e-6, max=1.0 - 1e-6)
                )
                q_activate = q_activate_dist.sample()
                policy_log_prob = q_dist.log_prob(q_sample) + q_activate_dist.log_prob(q_activate)
                q_active = bool(q_activate.item() >= 0.5)
            else:
                q_idx = int(torch.argmax(quarantine_scores).item())
                q_active = float(quarantine_scores[q_idx].item()) >= tau_q
            if q_active:
                quarantined_nodes.add(q_idx)
                for other in range(self.num_truthful_agents):
                    matrix[q_idx][other] *= 1.0 - quarantine_strength
                    matrix[other][q_idx] *= 1.0 - quarantine_strength
                applied.append({"type": "soft_quarantine", "node": q_idx, "score": float(quarantine_scores[q_idx].item())})

        if max_m > 0 and mask_scores.numel() > 0:
            m_idx = -1
            m_active = False
            edge_noop = False
            if stochastic_actions:
                m_logits_for_sampling = mask_logits.view(-1)
                if quarantined_nodes:
                    eligible = [
                        idx
                        for idx, (a, b) in enumerate(self.edge_pairs)
                        if a not in quarantined_nodes and b not in quarantined_nodes
                    ]
                    if eligible:
                        masked_logits = torch.full_like(m_logits_for_sampling, -1e9)
                        masked_logits[torch.tensor(eligible, device=m_logits_for_sampling.device)] = m_logits_for_sampling[
                            torch.tensor(eligible, device=m_logits_for_sampling.device)
                        ]
                        m_logits_for_sampling = masked_logits
                    else:
                        edge_noop = True
                if edge_noop:
                    edit_info_extra = {"edge_noop": True}
                else:
                    m_dist = torch.distributions.Categorical(logits=m_logits_for_sampling)
                    m_sample = m_dist.sample()
                    m_idx = int(m_sample.item())
                    m_activate_prob = _threshold_shifted_prob(mask_logits.view(-1)[m_idx], tau_m)
                    m_activate_dist = torch.distributions.Bernoulli(
                        probs=torch.clamp(m_activate_prob, min=1e-6, max=1.0 - 1e-6)
                    )
                    m_activate = m_activate_dist.sample()
                    m_lp = m_dist.log_prob(m_sample) + m_activate_dist.log_prob(m_activate)
                    policy_log_prob = m_lp if policy_log_prob is None else policy_log_prob + m_lp
                    m_active = bool(m_activate.item() >= 0.5)
            else:
                if quarantined_nodes:
                    eligible = [
                        idx
                        for idx, (a, b) in enumerate(self.edge_pairs)
                        if a not in quarantined_nodes and b not in quarantined_nodes
                    ]
                    if eligible:
                        masked_scores = torch.full_like(mask_scores.view(-1), -1.0)
                        eligible_idx = torch.tensor(eligible, device=mask_scores.device)
                        masked_scores[eligible_idx] = mask_scores.view(-1)[eligible_idx]
                        m_idx = int(torch.argmax(masked_scores).item())
                        m_active = float(mask_scores[m_idx].item()) >= tau_m
                    else:
                        edit_info_extra = {"edge_noop": True}
                else:
                    m_idx = int(torch.argmax(mask_scores).item())
                    m_active = float(mask_scores[m_idx].item()) >= tau_m
            if m_idx >= 0:
                i, j = self.edge_pairs[m_idx]
                if m_active and i not in quarantined_nodes and j not in quarantined_nodes:
                    matrix[i][j] = 0.0
                    applied.append({"type": "mask_edge", "edge": [i, j], "score": float(mask_scores[m_idx].item())})
        edit_info = {"actions": applied}
        edit_info.update(edit_info_extra)
        if policy_log_prob is not None:
            edit_info["policy_log_prob"] = policy_log_prob
            edit_info["stochastic"] = True
        return matrix, edit_info

    def _discretize_topology(self, matrix: List[List[float]]) -> List[List[int]]:
        threshold = float(self.method_config.get("edge-threshold", 0.45))
        discrete = [[0] * self.num_truthful_agents for _ in range(self.num_truthful_agents)]
        for i in range(self.num_truthful_agents):
            for j in range(i + 1, self.num_truthful_agents):
                discrete[i][j] = 1 if matrix[i][j] >= threshold else 0
        return discrete
    def _graph_distance(self, a: List[List[float]], b: List[List[float]]) -> torch.Tensor:
        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        return torch.mean(torch.abs(ta - tb))

    def _diversity_bonus(self, matrix: List[List[float]]) -> float:
        distances = []
        for proto in self.prototype_bank:
            distances.append(float(self._graph_distance(matrix, proto.matrix).item()))
        return sum(distances) / max(1, len(distances))

    def _struct_regularization(self, matrix: List[List[float]], risks: Optional[List[float]] = None) -> float:
        connectivity, cent_conc, bridge_risk = self._structural_components(matrix, risks)
        return connectivity - cent_conc - bridge_risk

    def _structural_components(
        self,
        matrix: List[List[float]],
        risks: Optional[List[float]] = None,
    ) -> Tuple[float, float, float]:
        risks = risks or [0.0] * self.num_truthful_agents
        n = len(matrix)
        out_degrees = [sum(row) for row in matrix]
        if not out_degrees:
            return 0.0, 0.0, 0.0
        total_weight = max(1.0, sum(out_degrees))
        centralization = max(out_degrees) / total_weight
        bridge_risk = 0.0
        for node_idx in range(n):
            if out_degrees[node_idx] > 0.5 * max(1.0, n - 1):
                bridge_risk += risks[node_idx]
        edge_mass = sum(out_degrees) / max(1.0, n * (n - 1) / 2.0)
        sinks = sum(1 for deg in out_degrees if deg == 0) / max(1, n)
        connectivity = 0.6 * edge_mass + 0.4 * (1.0 - sinks)
        return connectivity, centralization, bridge_risk / max(1.0, n)
