"""Scoring, policy rollouts, training loop, and public inference API."""

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import random
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from . import prompts
from ..utils import compute_effective_interaction_network
from .utils import (
    extract_choice_letter,
    extract_first_json_object,
    extract_last_number,
    normalize_text,
    safe_float,
)
class ScoringTrainingMixin:
    def _cvar_tail_mass(self) -> float:
        raw = self.method_config.get("cvar-tail-mass", self.method_config.get("cvar-alpha", 0.2))
        rho = float(raw)
        return max(1e-6, min(1.0, rho))

    def _cvar_tail_from_rewards(self, raw_rewards: List[float], rho: Optional[float] = None) -> Tuple[List[int], float, List[float]]:
        """Select a fixed-size worst-loss tail for empirical RU-CVaR PG updates."""
        n = len(raw_rewards)
        if n == 0:
            return [], 0.0, []
        rho = self._cvar_tail_mass() if rho is None else max(1e-6, min(1.0, float(rho)))
        adapt_losses = [-float(r) for r in raw_rewards]
        tail_k = max(1, int(math.ceil(rho * n)))
        ranked = sorted(range(n), key=lambda idx: (-adapt_losses[idx], idx))
        tail_indices = ranked[:tail_k]
        eta = min(adapt_losses[idx] for idx in tail_indices)
        return tail_indices, float(eta), adapt_losses

    def _normalized_reward_advantages(self, raw_rewards: List[float]) -> List[float]:
        if not raw_rewards:
            return []
        mu = sum(raw_rewards) / max(1, len(raw_rewards))
        variance = sum((r - mu) ** 2 for r in raw_rewards) / max(1, len(raw_rewards))
        sigma = math.sqrt(variance) + 1e-6
        return [(r - mu) / sigma for r in raw_rewards]

    def _editor_forward(
        self,
        query: str,
        sample: Optional[Dict[str, Any]],
        prior_matrix: List[List[float]],
        risks: List[float],
        uncertainties: List[float],
    ) -> Dict[str, torch.Tensor]:
        node_feats, edge_feats = self._editor_features(prior_matrix, risks, uncertainties)
        film = None
        delta_norm = None
        if bool(self.method_config.get("editor-hypernet-enabled", True)) and getattr(self, "editor_hypernet", None) is not None:
            context = self._editor_hyper_context(query, sample or {}, prior_matrix, risks, uncertainties)
            film = self.editor_hypernet(context)
            delta_norm = film.get("delta_norm")
        editor_outputs = self.risk_editor(
            node_feats,
            edge_feats,
            film=film,
            delta_scale=float(self.method_config.get("editor-delta-scale", 0.1)),
        )
        if delta_norm is not None:
            editor_outputs["hyper_delta_norm"] = delta_norm
        return editor_outputs

    def _rollout_with_fixed_prototype(
        self,
        sample: Dict[str, Any],
        proto_idx: int,
        forced_adversarial_indices: Optional[List[int]] = None,
        use_editor: bool = True,
    ) -> Dict[str, Any]:
        query = sample["query"]
        adversarial_indices = (
            list(forced_adversarial_indices)
            if forced_adversarial_indices is not None
            else self._sample_adversarial_agent_indices()
        )
        proto_idx = max(0, min(int(proto_idx), len(self.prototype_bank) - 1))
        prior_matrix = deepcopy(self.prototype_bank[proto_idx].matrix)
        prior_adj = self._discretize_topology(prior_matrix)
        self._build_runtime(prior_adj, probe_mode=True, forced_adversarial_indices=adversarial_indices)
        probe_response, probe_outputs = self._execute_current_runtime(self._make_probe_query(query))
        try:
            risks, uncertainties, judge_result = self._judge_probe_rollout_with_llm(query, probe_outputs, prior_adj=prior_adj)
        except Exception:
            neutral_len = len(probe_outputs)
            risks = [0.5 for _ in range(neutral_len)]
            uncertainties = [0.5 for _ in range(neutral_len)]
            judge_result = {"query": query, "judge_type": "llm_fallback_neutral"}
        editor_outputs: Dict[str, torch.Tensor] = {}
        if use_editor:
            editor_outputs = self._editor_forward(query, sample, prior_matrix, risks, uncertainties)
            new_matrix, edit_info = self._apply_residual_edits(prior_matrix, editor_outputs, risks, uncertainties)
        else:
            new_matrix = deepcopy(prior_matrix)
            edit_info = {"actions": []}
        new_adj = self._discretize_topology(new_matrix)
        return {
            "probe_response": probe_response,
            "probe_outputs": probe_outputs,
            "risks": risks,
            "uncertainties": uncertainties,
            "judge_result": judge_result,
            "prior_matrix": prior_matrix,
            "prior_adj": prior_adj,
            "new_matrix": new_matrix,
            "new_adj": new_adj,
            "edit_info": edit_info,
            "adversarial_agent_indices": adversarial_indices,
            "proto_idx": proto_idx,
            "editor_outputs": editor_outputs,
        }

    def _best_prototype_label(self, sample: Dict[str, Any]) -> int:
        return self._best_prototype_label_with_forced_adversary(sample, forced_adversarial_indices=None)

    def _best_prototype_label_with_forced_adversary(
        self,
        sample: Dict[str, Any],
        forced_adversarial_indices: Optional[List[int]] = None,
    ) -> int:
        fixed_adversarial_indices = (
            list(forced_adversarial_indices)
            if forced_adversarial_indices is not None
            else self._sample_adversarial_agent_indices()
        )
        best_idx = 0
        best_acc = -1.0
        best_total = -1e18
        for idx in range(len(self.prototype_bank)):
            rollout = self._rollout_with_fixed_prototype(
                sample,
                idx,
                forced_adversarial_indices=fixed_adversarial_indices,
                use_editor=False,
            )
            token_before = deepcopy(self.get_token_stats())
            self._build_runtime(
                rollout["prior_adj"],
                probe_mode=False,
                forced_adversarial_indices=rollout["adversarial_agent_indices"],
            )
            prior_response, _ = self._execute_current_runtime(sample["query"])
            ret_prior = self._compute_return(
                prior_response,
                sample,
                rollout["prior_matrix"],
                0,
                token_before,
            )
            acc = float(ret_prior["acc"])
            total = float(ret_prior["total"])
            if (acc > best_acc) or (acc == best_acc and total > best_total):
                best_acc = acc
                best_total = total
                best_idx = idx
        return int(best_idx)

    def _supervised_step(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        query = sample["query"]
        fixed_adversarial_indices = self._sample_adversarial_agent_indices()
        selector_x = self._selector_features(query=query, sample=sample)
        selector_logits = self.prototype_selector(selector_x)
        target_proto = self._best_prototype_label_with_forced_adversary(
            sample,
            forced_adversarial_indices=fixed_adversarial_indices,
        )
        target_proto_t = torch.tensor([target_proto], dtype=torch.long, device=selector_logits.device)
        selector_loss = F.cross_entropy(selector_logits.view(1, -1), target_proto_t)

        fixed_rollout = self._rollout_with_fixed_prototype(
            sample,
            target_proto,
            forced_adversarial_indices=fixed_adversarial_indices,
            use_editor=True,
        )
        risks = fixed_rollout["risks"]
        prior_matrix = fixed_rollout["prior_matrix"]
        editor_outputs = fixed_rollout["editor_outputs"]

        top_risk_idx = int(max(range(len(risks)), key=lambda idx: float(risks[idx]))) if risks else -1
        target_q_vals = [1.0 if idx == top_risk_idx else 0.0 for idx in range(len(risks))]
        target_q = torch.tensor(target_q_vals, dtype=torch.float32, device=selector_logits.device)
        target_m_vals = []
        for i, j in self.edge_pairs:
            is_adjacent_to_top_risk = (i == top_risk_idx) or (j == top_risk_idx)
            has_prior_edge = float(prior_matrix[i][j]) > 0.0
            target_m_vals.append(1.0 if is_adjacent_to_top_risk and has_prior_edge else 0.0)
        target_m = torch.tensor(target_m_vals, dtype=torch.float32, device=selector_logits.device)

        loss_q = F.binary_cross_entropy_with_logits(editor_outputs["quarantine_logits"], target_q)
        loss_m = F.binary_cross_entropy_with_logits(editor_outputs["mask_logits"], target_m)
        loss_sparse = torch.mean(torch.sigmoid(editor_outputs["quarantine_logits"])) + torch.mean(torch.sigmoid(editor_outputs["mask_logits"]))
        loss_sparse_weight = float(self.method_config.get("loss-sparse-weight", 0.0))
        editor_loss = loss_q + 0.7 * loss_m + loss_sparse_weight * loss_sparse
        delta_reg = float(self.method_config.get("editor-delta-reg", 0.0))
        if delta_reg > 0.0 and editor_outputs.get("hyper_delta_norm") is not None:
            editor_loss = editor_loss + delta_reg * editor_outputs["hyper_delta_norm"]
        q_max = (
            float(torch.max(torch.sigmoid(editor_outputs["quarantine_logits"])).detach().item())
            if editor_outputs["quarantine_logits"].numel() > 0
            else 0.0
        )
        m_max = (
            float(torch.max(torch.sigmoid(editor_outputs["mask_logits"])).detach().item())
            if editor_outputs["mask_logits"].numel() > 0
            else 0.0
        )
        print(
            f"[editor-supervision] sample='{str(query)[:120]}' q_max={q_max:.4f} m_max={m_max:.4f} "
            f"target_q_pos={int(target_q.sum().detach().item())} target_m_pos={int(target_m.sum().detach().item())}",
            flush=True,
        )
        total_loss = selector_loss + editor_loss
        return total_loss, {
            "selector_loss": float(selector_loss.detach().item()),
            "editor_loss": float(editor_loss.detach().item()),
            "target_proto": int(target_proto),
            "adversarial_agent_indices": list(fixed_adversarial_indices),
        }

    def _optimizing_supervised(
        self,
        val_dataset: List[Dict[str, Any]],
        rounds_override: Optional[int] = None,
        episode_offset: int = 0,
    ) -> None:
        self.inference_flag = False
        if episode_offset == 0:
            self.episode_stats = []
        rounds = (
            int(rounds_override)
            if rounds_override is not None
            else int(self.method_config.get("optimization-rounds", 1))
        )
        support_size = min(int(self.method_config.get("support-size", 3)), len(val_dataset))
        pbar = tqdm(total=rounds * support_size, desc="supervised optimization", unit="step", leave=True)
        for episode_idx in range(rounds):
            support_set, target_set = self._sample_episode(val_dataset)
            support_set = support_set[:support_size]
            batch_losses: List[torch.Tensor] = []
            support_rollouts: List[Dict[str, Any]] = []
            for sample in support_set:
                loss, logs = self._supervised_step(sample)
                batch_losses.append(loss)
                rollout = self._evaluate_policy_pair(sample)
                rollout["query"] = sample.get("query", "")
                support_rollouts.append(rollout)
                pbar.set_postfix(ep=episode_offset + episode_idx + 1, phase="support", s_loss=logs["selector_loss"], e_loss=logs["editor_loss"])
                pbar.update(1)
            self.optimizer.zero_grad()
            total_loss = sum(batch_losses) / max(1, len(batch_losses))
            total_loss.backward()
            grad_clip = float(self.method_config.get("grad-clip-norm", 1.0))
            grad_params = (
                list(self.prototype_selector.parameters())
                + list(self.risk_editor.parameters())
                + list(self.descriptor_text_projector.parameters())
            )
            if getattr(self, "editor_hypernet", None) is not None:
                grad_params += list(self.editor_hypernet.parameters())
            torch.nn.utils.clip_grad_norm_(grad_params, grad_clip)
            self.optimizer.step()
            target_rollouts = []
            for sample in target_set:
                rollout = self._evaluate_policy_pair(sample)
                rollout["query"] = sample.get("query", "")
                target_rollouts.append(rollout)
            episode_summary = self._build_episode_summary(
                episode_offset + episode_idx + 1, support_rollouts, target_rollouts
            )
            episode_summary["support"]["final_loss"] = float(total_loss.detach().item())
            episode_summary["phase"] = "supervised_warmup"
            self._persist_episode_summary(episode_summary)
        pbar.close()
        self._save_state()
        self.inference_flag = True

    def _judge_probe_rollout_with_llm(
        self,
        query: str,
        probe_outputs: List[Dict[str, Any]],
        prior_adj: List[List[int]],
        sample: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[float], List[float], Dict[str, Any]]:
        if not probe_outputs:
            return [], [], {"query": query, "per_agent": [], "judge_type": "llm"}

        serialized_outputs = []
        for idx, item in enumerate(probe_outputs):
            output_text = str(item.get("output", ""))
            evidence_json = {}
            candidate = extract_first_json_object(output_text)
            if candidate:
                try:
                    parsed_output = json.loads(candidate)
                    if isinstance(parsed_output, dict):
                        evidence_json = {
                            "claim": parsed_output.get("claim", ""),
                            "tentative_answer": parsed_output.get("tentative_answer", ""),
                            "evidence": parsed_output.get("evidence", []),
                            "self_check": parsed_output.get("self_check", []),
                            "possible_failure": parsed_output.get("possible_failure", []),
                            "used_upstream": parsed_output.get("used_upstream", []),
                        }
                except Exception:
                    evidence_json = {}
            serialized_outputs.append(
                {
                    "agent_index": int(item.get("agent_index", idx + 1)),
                    "node_id": str(item.get("node_id", "")),
                    "role": str(item.get("role", self.roles[idx] if idx < len(self.roles) else "")),
                    "output": output_text,
                    "probe_evidence": evidence_json,
                }
            )

        prompt_payload = {
            "query": query,
            "dataset_mode": self.dataset_mode,
            "known_roles": self.roles,
            "num_agents": len(serialized_outputs),
            "single_adversary_assumption": False,
            "prior_adjacency": prior_adj,
            "agent_outputs": serialized_outputs,
            "instructions": {
                "score_range": "risk and uncertainty must be float in [0,1]",
                "score_definition": {
                    "unreliability_risk": "absolute unreliability score for this agent (independent, not normalized)",
                    "uncertainty": "confidence instability/ambiguity in this agent output",
                },
                "required_json_schema": {
                    "per_agent": [
                        {
                            "agent_index": 1,
                            "unreliability_risk": 0.0,
                            "uncertainty": 0.0,
                            "reason": "short reason",
                        }
                    ],
                    "abstain": False,
                    "evidence_strength": 0.0,
                    "none_suspicious_prob": 0.0,
                    "insufficient_evidence": False,
                },
                "calibration_rules": [
                    "provide one row for every agent_index from 1..num_agents",
                    "do not force a single top suspect under weak evidence",
                    "ranking can abstain when evidence is insufficient",
                    "do not punish normal role disagreement unless there is evidence of malicious misleading behavior",
                ],
            },
        }
        user_prompt = (
            "Return strict JSON only. No markdown.\n"
            "If unsure, still provide numeric values.\n"
            "Ensure per_agent is complete and sorted by agent_index ascending.\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False)}"
        )
        raw_response = self.call_llm(
            messages=[
                {"role": "system", "content": prompts.ADVERSARIAL_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        candidate = extract_first_json_object(raw_response)
        parsed = json.loads(candidate) if candidate else {}
        per_agent = parsed.get("per_agent", []) if isinstance(parsed, dict) else []

        size = len(serialized_outputs)
        risk_map = {idx + 1: 0.5 for idx in range(size)}
        uncertainty_map = {idx + 1: 0.5 for idx in range(size)}
        reason_map = {idx + 1: "" for idx in range(size)}
        for item in per_agent:
            if not isinstance(item, dict):
                continue
            agent_index = int(safe_float(item.get("agent_index"), 0))
            if agent_index < 1 or agent_index > size:
                continue
            risk_map[agent_index] = max(0.0, min(1.0, safe_float(item.get("unreliability_risk", item.get("risk")), 0.5)))
            uncertainty_map[agent_index] = max(0.0, min(1.0, safe_float(item.get("uncertainty"), 0.5)))
            reason_map[agent_index] = normalize_text(item.get("reason", ""))

        raw_risks = [risk_map[idx + 1] for idx in range(size)]
        raw_uncertainties = [uncertainty_map[idx + 1] for idx in range(size)]
        self._latest_raw_risks = list(raw_risks)
        self._latest_raw_uncertainties = list(raw_uncertainties)
        risks = list(raw_risks)
        uncertainties = list(raw_uncertainties)
        explanations = []
        for idx in range(size):
            explanations.append(
                {
                    "agent_index": idx + 1,
                    "role": serialized_outputs[idx].get("role", ""),
                    "risk": risks[idx],
                    "uncertainty": uncertainties[idx],
                    "reason": reason_map[idx + 1],
                }
            )
        return risks, uncertainties, {"query": query, "per_agent": explanations, "judge_type": "llm", "raw_output": raw_response, "raw_risks": raw_risks, "raw_uncertainties": raw_uncertainties, "abstain": bool(parsed.get("abstain", False)), "evidence_strength": max(0.0, min(1.0, safe_float(parsed.get("evidence_strength"), 0.0))), "none_suspicious_prob": max(0.0, min(1.0, safe_float(parsed.get("none_suspicious_prob"), 0.0))), "insufficient_evidence": bool(parsed.get("insufficient_evidence", False))}
    def _score_answer(self, prediction: str, sample: Dict[str, Any]) -> float:
        gt = sample.get("gt")
        if gt is None:
            return 0.0
        raw_prediction = prediction if isinstance(prediction, str) else str(prediction)
        prediction = normalize_text(raw_prediction)
        gt_text = normalize_text(gt)
        if self.dataset_mode == "medmcqa":
            return 1.0 if extract_choice_letter(prediction) == extract_choice_letter(gt_text) else 0.0
        pred_number = extract_last_number(prediction)
        gt_number = extract_last_number(gt_text)
        if pred_number and gt_number:
            return 1.0 if pred_number == gt_number else 0.0
        return float(gt_text.lower() in prediction.lower() or prediction.lower() in gt_text.lower())

    def _compute_return(
        self,
        response: str,
        sample: Dict[str, Any],
        matrix: List[List[float]],
        edit_actions_count: int,
        token_before: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        current_stats = self.get_token_stats()[self.model_name]
        used_prompt = current_stats["prompt_tokens"] - token_before[self.model_name]["prompt_tokens"]
        used_completion = current_stats["completion_tokens"] - token_before[self.model_name]["completion_tokens"]
        token_cost = used_prompt + used_completion
        acc = self._score_answer(response, sample)
        edit_cost = float(edit_actions_count)
        struct_reg = self._struct_regularization(matrix)
        div = self._diversity_bonus(matrix)
        total = (
            acc
            - float(self.method_config.get("lambda-token", 2e-4)) * token_cost
            - float(self.method_config.get("lambda-edit", 0.03)) * edit_cost
            + float(self.method_config.get("lambda-struct", 0.10)) * struct_reg
            + float(self.method_config.get("lambda-div", 0.08)) * div
        )
        return {
            "total": total,
            "acc": acc,
            "token_cost": float(token_cost),
            "edit_cost": edit_cost,
            "struct_reg": struct_reg,
            "div": div,
        }

    def _record_prototype_feedback(self, alpha: torch.Tensor, reward: float) -> None:
        # v1 uses a fixed prototype bank with immutable per-prototype stats.
        return

    def _sample_episode(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not dataset:
            return [], []
        support_size = min(int(self.method_config.get("support-size", 3)), len(dataset))
        default_target_size = max(1, support_size // 2)
        target_size = min(int(self.method_config.get("target-size", default_target_size)), len(dataset))
        shuffled = random.sample(dataset, len(dataset))
        support_set = shuffled[:support_size]
        remaining = shuffled[support_size:]
        if not remaining:
            target_set = support_set[:target_size]
        else:
            target_set = remaining[: min(target_size, len(remaining))]
            if len(target_set) < target_size:
                target_set = target_set + support_set[: target_size - len(target_set)]
        return support_set, target_set

    def _evaluate_policy_pair(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        rollout = self._run_policy(sample)

        token_before = deepcopy(self.get_token_stats())
        self._build_runtime(
            rollout["prior_adj"],
            probe_mode=False,
            forced_adversarial_indices=rollout["adversarial_agent_indices"],
        )
        prior_response, _ = self._execute_current_runtime(sample["query"])
        ret_prior = self._compute_return(prior_response, sample, rollout["prior_matrix"], 0, token_before)

        token_before = deepcopy(self.get_token_stats())
        self._build_runtime(
            rollout["new_adj"],
            probe_mode=False,
            forced_adversarial_indices=rollout["adversarial_agent_indices"],
        )
        new_response, _ = self._execute_current_runtime(sample["query"])
        edit_count = len(rollout["edit_info"]["actions"])
        ret_new = self._compute_return(new_response, sample, rollout["new_matrix"], edit_count, token_before)

        j_prior = ret_prior["total"]
        j_new = ret_new["total"]
        acc_prior = ret_prior["acc"]
        acc_new = ret_new["acc"]

        # gain_aux: lambda-weighted auxiliary improvement, matching _compute_return() coefficients
        # so that gain_aux is dimensionally comparable to gain_acc ∈ {-1,0,1}
        lam_token = float(self.method_config.get("lambda-token", 2e-4))
        lam_edit = float(self.method_config.get("lambda-edit", 0.03))
        lam_struct = float(self.method_config.get("lambda-struct", 0.10))
        lam_div = float(self.method_config.get("lambda-div", 0.08))
        gain_aux = (
            - lam_token * (ret_new["token_cost"] - ret_prior["token_cost"])
            - lam_edit  * (ret_new["edit_cost"]  - ret_prior["edit_cost"])
            + lam_struct * (ret_new["struct_reg"] - ret_prior["struct_reg"])
            + lam_div   * (ret_new["div"]         - ret_prior["div"])
        )

        rollout.update(
            {
                "prior_response": prior_response,
                "new_response": new_response,
                "j_prior": j_prior,
                "j_new": j_new,
                "acc_prior": acc_prior,
                "acc_new": acc_new,
                "gain_acc": acc_new - acc_prior,
                "gain_total": j_new - j_prior,
                "gain_aux": gain_aux,
                "edit_count": edit_count,
            }
        )
        return rollout

    def _sample_rollout(self, sample: Dict[str, Any], update_bank: bool = True) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Run one rollout and return (policy_log_prob, raw_reward, rollout).

        Raw reward is NOT yet normalized; normalization is done at the batch level
        in optimizing() so that advantage standardization can use batch statistics.
        Bank updates are gated by update_bank to avoid double-counting when the
        caller handles them explicitly.
        """
        delta_perf = float(self.method_config.get("delta-perf", 0.10))
        delta_novel = float(self.method_config.get("delta-novel", 0.15))
        cvar_beta = float(self.method_config.get("cvar-beta", 0.05))

        rollout = self._evaluate_policy_pair(sample)

        if update_bank:
            self._record_prototype_feedback(rollout["prior_alpha"], rollout["j_prior"])
            self._record_prototype_feedback(rollout["final_alpha"], rollout["j_new"])
            if rollout["j_new"] >= rollout["j_prior"] + delta_perf:
                novelty = float(self._graph_distance(rollout["new_matrix"], rollout["prior_matrix"]).item())
                if novelty >= delta_novel:
                    self._update_prototype_bank(rollout["new_matrix"], rollout["j_new"])

        raw_reward = rollout["gain_acc"] + cvar_beta * rollout["gain_aux"]
        log_prob: torch.Tensor = rollout["policy_log_prob"]
        return log_prob, raw_reward, rollout

    def _sample_training_loss(self, sample: Dict[str, Any], update_bank: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Thin wrapper kept for backwards-compatibility; used only when update_bank=True (e.g. standalone calls)."""
        log_prob, raw_reward, rollout = self._sample_rollout(sample, update_bank=update_bank)
        reward_t = torch.tensor(raw_reward, dtype=torch.float32)
        loss = -reward_t * log_prob
        return loss, rollout

    def _build_episode_summary(
        self,
        episode_index: int,
        support_rollouts: List[Dict[str, Any]],
        target_rollouts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _mean(values: List[float]) -> float:
            return float(sum(values) / max(1, len(values))) if values else 0.0

        def _tail_mean(values: List[float], alpha: float) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            tail_n = max(1, int(math.ceil(len(sorted_vals) * alpha)))
            return float(sum(sorted_vals[:tail_n]) / tail_n)

        cvar_alpha = self._cvar_tail_mass()

        support_j_prior = [float(item["j_prior"]) for item in support_rollouts]
        support_j_new = [float(item["j_new"]) for item in support_rollouts]
        support_gain = [new - prior for prior, new in zip(support_j_prior, support_j_new)]
        support_gain_acc = [float(item.get("gain_acc", 0.0)) for item in support_rollouts]
        support_edit_count = [float(item["edit_count"]) for item in support_rollouts]
        support_risk = [float(sum(item["risks"]) / max(1, len(item["risks"]))) for item in support_rollouts]
        support_uncertainty = [
            float(sum(item["uncertainties"]) / max(1, len(item["uncertainties"]))) for item in support_rollouts
        ]

        target_j_prior = [float(item["j_prior"]) for item in target_rollouts]
        target_j_new = [float(item["j_new"]) for item in target_rollouts]
        target_gain = [new - prior for prior, new in zip(target_j_prior, target_j_new)]
        target_gain_acc = [float(item.get("gain_acc", 0.0)) for item in target_rollouts]
        target_edit_count = [float(item["edit_count"]) for item in target_rollouts]
        target_risk = [float(sum(item["risks"]) / max(1, len(item["risks"]))) for item in target_rollouts]
        target_uncertainty = [
            float(sum(item["uncertainties"]) / max(1, len(item["uncertainties"]))) for item in target_rollouts
        ]

        return {
            "episode_index": int(episode_index),
            "support_size": len(support_rollouts),
            "target_size": len(target_rollouts),
            "support": {
                "avg_j_prior": _mean(support_j_prior),
                "avg_j_new": _mean(support_j_new),
                "avg_gain": _mean(support_gain),
                "avg_gain_acc": _mean(support_gain_acc),
                "tail_gain_acc": _tail_mean(support_gain_acc, cvar_alpha),
                "avg_edit_count": _mean(support_edit_count),
                "avg_risk": _mean(support_risk),
                "avg_uncertainty": _mean(support_uncertainty),
                "queries": [item.get("query", "") for item in support_rollouts],
            },
            "target": {
                "avg_j_prior": _mean(target_j_prior),
                "avg_j_new": _mean(target_j_new),
                "avg_gain": _mean(target_gain),
                "avg_gain_acc": _mean(target_gain_acc),
                "tail_gain_acc": _tail_mean(target_gain_acc, cvar_alpha),
                "avg_edit_count": _mean(target_edit_count),
                "avg_risk": _mean(target_risk),
                "avg_uncertainty": _mean(target_uncertainty),
                "queries": [item.get("query", "") for item in target_rollouts],
            },
            "prototypes": [
                {
                    "name": record.name,
                    "score": float(record.score),
                    "visits": float(record.visits),
                }
                for record in self.prototype_bank
            ],
        }

    def _persist_episode_summary(self, summary: Dict[str, Any]) -> None:
        self.episode_stats.append(summary)
        stats_path = os.path.join(self.state_dir, "episode_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.episode_stats, f, ensure_ascii=False, indent=2)

    def _run_policy(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = sample["query"]
        adversarial_indices = self._sample_adversarial_agent_indices()
        selector_x = self._selector_features(query=query, sample=sample)
        selector_logits = self.prototype_selector(selector_x)
        prior_matrix, proto_idx, selector_log_prob = self._select_prior_top1(selector_logits)
        prior_alpha = torch.nn.functional.one_hot(
            torch.tensor(proto_idx, device=selector_logits.device),
            num_classes=len(self.prototype_bank),
        ).float()
        prior_alpha_dist = torch.softmax(selector_logits, dim=-1)
        prior_adj = self._discretize_topology(prior_matrix)

        self._build_runtime(prior_adj, probe_mode=True, forced_adversarial_indices=adversarial_indices)
        probe_response, probe_outputs = self._execute_current_runtime(self._make_probe_query(query))
        try:
            risks, uncertainties, judge_result = self._judge_probe_rollout_with_llm(
                query, probe_outputs, prior_adj=prior_adj
            )
        except Exception as llm_judge_error:
            neutral_len = len(probe_outputs)
            risks = [0.5 for _ in range(neutral_len)]
            uncertainties = [0.5 for _ in range(neutral_len)]
            judge_result = {
                "query": query,
                "per_agent": [
                    {
                        "agent_index": idx + 1,
                        "role": str(probe_outputs[idx].get("role", self.roles[idx] if idx < len(self.roles) else "")),
                        "risk": 0.5,
                        "uncertainty": 0.5,
                        "reason": "llm_judge_failed",
                    }
                    for idx in range(neutral_len)
                ],
                "judge_type": "llm_fallback_neutral",
                "judge_error": str(llm_judge_error),
                "raw_risks": [0.5] * neutral_len,
                "raw_uncertainties": [0.5] * neutral_len,
            }

        if bool(self.method_config.get("probe-judge-only", False)):
            device = selector_logits.device
            zero = torch.tensor(0.0, device=device)
            new_matrix = deepcopy(prior_matrix)
            new_adj = self._discretize_topology(new_matrix)
            final_alpha = prior_alpha.detach().clone()
            final_alpha_dist = prior_alpha_dist.detach().clone()
            gate_values_prior = {"selector": float(torch.max(prior_alpha_dist).item())}
            return {
                "probe_response": probe_response,
                "probe_outputs": probe_outputs,
                "risks": risks,
                "uncertainties": uncertainties,
                "judge_result": judge_result,
                "prior_matrix": prior_matrix,
                "prior_adj": prior_adj,
                "prior_alpha": prior_alpha,
                "prior_alpha_dist": prior_alpha_dist,
                "final_alpha": final_alpha,
                "final_alpha_dist": final_alpha_dist,
                "new_matrix": new_matrix,
                "new_adj": new_adj,
                "edit_info": {"actions": []},
                "adversarial_agent_indices": adversarial_indices,
                "gate_values_prior": gate_values_prior,
                "gate_values_final": dict(gate_values_prior),
                "entropy": zero,
                "policy_log_prob": selector_log_prob + zero,
                "prior_policy_log_prob": selector_log_prob,
                "final_policy_log_prob": selector_log_prob + zero,
                "final_selector_used": False,
                "editor_outputs": {},
            }
        editor_outputs = self._editor_forward(query, sample, prior_matrix, risks, uncertainties)
        new_matrix, edit_info = self._apply_residual_edits(prior_matrix, editor_outputs, risks, uncertainties)
        new_adj = self._discretize_topology(new_matrix)
        gate_values_prior = {"selector": float(torch.max(prior_alpha_dist).item())}
        gate_values_final = {
            "quarantine": float(torch.max(torch.sigmoid(editor_outputs["quarantine_logits"])).item()) if editor_outputs["quarantine_logits"].numel() > 0 else 0.0,
            "mask": float(torch.max(torch.sigmoid(editor_outputs["mask_logits"])).item()) if editor_outputs["mask_logits"].numel() > 0 else 0.0,
        }
        zero = torch.tensor(0.0, device=selector_logits.device)
        editor_policy_log_prob = self._editor_realized_policy_log_prob(
            {
                "editor_outputs": editor_outputs,
                "edit_info": edit_info,
                "prior_alpha": prior_alpha,
            }
        )

        return {
            "probe_response": probe_response,
            "probe_outputs": probe_outputs,
            "risks": risks,
            "uncertainties": uncertainties,
            "judge_result": judge_result,
            "prior_matrix": prior_matrix,
            "prior_adj": prior_adj,
            "prior_alpha": prior_alpha,
            "prior_alpha_dist": prior_alpha_dist,
            "final_alpha": prior_alpha,
            "final_alpha_dist": prior_alpha_dist,
            "new_matrix": new_matrix,
            "new_adj": new_adj,
            "edit_info": edit_info,
            "adversarial_agent_indices": adversarial_indices,
            "gate_values_prior": gate_values_prior,
            "gate_values_final": gate_values_final,
            # The episodic reward is defined on improvement from prior -> new.
            # We only reinforce the decision variables that create the new topology,
            # otherwise a strong prior can be penalized simply because there is less
            # headroom left for improvement.
            "entropy": zero,
            "policy_log_prob": selector_log_prob + editor_policy_log_prob,
            "prior_policy_log_prob": selector_log_prob,
            "final_policy_log_prob": selector_log_prob + editor_policy_log_prob,
            "editor_policy_log_prob": editor_policy_log_prob,
            "final_selector_used": False,
            "editor_outputs": editor_outputs,
        }

    def _editor_realized_policy_log_prob(self, rollout: Dict[str, Any]) -> torch.Tensor:
        """Differentiable log-prob surrogate for the realized quarantine/mask decisions.

        Treats each node/edge Bernoulli with p=sigmoid(logit); z is 1 on realized
        quarantine node / masked edge and 0 elsewhere (including when no edit fires).
        """
        editor_outputs = rollout.get("editor_outputs") or {}
        q_logits = editor_outputs.get("quarantine_logits")
        m_logits = editor_outputs.get("mask_logits")
        edit_info = rollout.get("edit_info") or {}
        if isinstance(edit_info.get("policy_log_prob"), torch.Tensor):
            return edit_info["policy_log_prob"]
        actions = edit_info.get("actions") or []
        device = rollout["prior_alpha"].device
        if q_logits is None or m_logits is None:
            return torch.tensor(0.0, device=device)
        q_logits = q_logits.view(-1)
        m_logits = m_logits.view(-1)
        z_q = torch.zeros_like(q_logits)
        z_m = torch.zeros_like(m_logits)
        for act in actions:
            if not isinstance(act, dict):
                continue
            if act.get("type") == "soft_quarantine":
                node = int(act.get("node", -1))
                if 0 <= node < z_q.numel():
                    z_q[node] = 1.0
            elif act.get("type") == "mask_edge":
                edge = act.get("edge") or []
                if len(edge) >= 2:
                    i, j = int(edge[0]), int(edge[1])
                    for idx, (a, b) in enumerate(self.edge_pairs):
                        if (a == i and b == j) or (a == j and b == i):
                            if idx < z_m.numel():
                                z_m[idx] = 1.0
                            break
        p_q = torch.sigmoid(q_logits)
        p_m = torch.sigmoid(m_logits)
        eps = 1e-6
        lp_q = z_q * torch.log(p_q + eps) + (1.0 - z_q) * torch.log(1.0 - p_q + eps)
        lp_m = z_m * torch.log(p_m + eps) + (1.0 - z_m) * torch.log(1.0 - p_m + eps)
        return lp_q.sum() + lp_m.sum()

    def _optimizing_cvar_finetune(
        self,
        val_dataset: List[Dict[str, Any]],
        rounds: int = 1,
        episode_offset: int = 0,
    ) -> None:
        """CVaR fine-tune after supervised warm-up.

        The modeled risk variable is adaptation loss, L_adapt = -reward. We
        define the empirical Rockafellar-Uryasev CVaR tail over this black-box
        loss, then update the editor through a likelihood-ratio policy-gradient
        surrogate on the fixed-size worst-loss tail. The reward itself is never
        differentiated through.

        Does not read ``warmup-rounds`` (that key applies only to the pure-RL optimizing branch); warm-up is solely
        the supervised phase in ``optimizing()``.
        """
        scope = str(self.method_config.get("cvar-update-scope", "editor_only")).strip().lower()
        if scope not in ("editor_only", "editor-only"):
            raise ValueError(f"Unsupported cvar-update-scope for supervised hybrid: {scope}")

        prev_sel = [p.requires_grad for p in self.prototype_selector.parameters()]
        prev_proj = [p.requires_grad for p in self.descriptor_text_projector.parameters()]
        prev_ed = [p.requires_grad for p in self.risk_editor.parameters()]
        hypernet = getattr(self, "editor_hypernet", None)
        prev_hyper = [p.requires_grad for p in hypernet.parameters()] if hypernet is not None else []
        prev_stochastic_editor_actions = bool(getattr(self, "_stochastic_editor_actions", False))
        try:
            self._stochastic_editor_actions = True
            for p in self.prototype_selector.parameters():
                p.requires_grad_(False)
            for p in self.descriptor_text_projector.parameters():
                p.requires_grad_(False)
            for p in self.risk_editor.parameters():
                p.requires_grad_(True)
            if hypernet is not None:
                for p in hypernet.parameters():
                    p.requires_grad_(True)

            lr = float(self.method_config.get("cvar-learning-rate", 1e-4))
            cvar_params = list(self.risk_editor.parameters())
            if hypernet is not None:
                cvar_params += list(hypernet.parameters())
            cvar_optimizer = torch.optim.Adam(cvar_params, lr=lr)

            self.inference_flag = False
            support_updates = int(self.method_config.get("support-updates-per-episode", 1))
            support_size = min(int(self.method_config.get("support-size", 3)), len(val_dataset))
            target_size = min(int(self.method_config.get("target-size", max(1, support_size // 2))), len(val_dataset))
            cvar_alpha = self._cvar_tail_mass()
            cvar_lambda = float(self.method_config.get("cvar-lambda", 1.0))
            delta_perf = float(self.method_config.get("delta-perf", 0.10))
            delta_novel = float(self.method_config.get("delta-novel", 0.15))

            total_steps = rounds * (support_updates * support_size + target_size)
            pbar = tqdm(total=total_steps, desc="cvar editor finetune", unit="step", leave=True)

            for episode_idx in range(rounds):
                support_set, target_set = self._sample_episode(val_dataset)
                episode_loss_value = 0.0
                tail_metric = "adapt_loss_topk"
                tail_selected_count = 0
                tail_eta = 0.0
                support_rollouts: List[Dict[str, Any]] = []
                target_rollouts: List[Dict[str, Any]] = []

                for _ in range(support_updates):
                    cvar_optimizer.zero_grad()
                    batch_items: List[Tuple[float, torch.Tensor, float, Dict[str, Any]]] = []

                    for sample in support_set:
                        _, raw_reward, rollout = self._sample_rollout(sample, update_bank=False)
                        rollout["query"] = sample.get("query", "")
                        editor_lp = self._editor_realized_policy_log_prob(rollout)
                        support_rollouts.append(rollout)
                        batch_items.append((0.0, editor_lp, raw_reward, rollout))
                        pbar.set_postfix(ep=episode_offset + episode_idx + 1, phase="cvar_support", collecting=True)
                        pbar.update(1)

                    for _, _, _, rollout in batch_items:
                        self._record_prototype_feedback(rollout["prior_alpha"], rollout["j_prior"])
                        self._record_prototype_feedback(rollout["final_alpha"], rollout["j_new"])
                        if rollout["gain_total"] >= delta_perf:
                            novelty = float(self._graph_distance(rollout["new_matrix"], rollout["prior_matrix"]).item())
                            if novelty >= delta_novel:
                                self._update_prototype_bank(rollout["new_matrix"], rollout["j_new"])

                    raw_rewards = [r for _, _, r, _ in batch_items]
                    advantages = self._normalized_reward_advantages(raw_rewards)
                    tail_indices, tail_eta, _ = self._cvar_tail_from_rewards(raw_rewards, cvar_alpha)
                    tail_selected_count = len(tail_indices)

                    per_losses = [-adv * log_prob for adv, (_, log_prob, _, _) in zip(advantages, batch_items)]

                    n_batch = len(per_losses)
                    mean_loss = sum(per_losses) / max(1, n_batch)
                    cvar_loss = (
                        sum(per_losses[idx] for idx in tail_indices) / max(1, len(tail_indices))
                        if tail_indices
                        else mean_loss
                    )
                    total_loss = (1.0 - cvar_lambda) * mean_loss + cvar_lambda * cvar_loss
                    delta_reg = float(self.method_config.get("editor-delta-reg", 0.0))
                    if delta_reg > 0.0:
                        delta_terms = [
                            rollout["editor_outputs"].get("hyper_delta_norm")
                            for _, _, _, rollout in batch_items
                            if isinstance(rollout.get("editor_outputs"), dict)
                            and rollout["editor_outputs"].get("hyper_delta_norm") is not None
                        ]
                        if delta_terms:
                            total_loss = total_loss + delta_reg * (sum(delta_terms) / len(delta_terms))

                    entropy_coeff = float(self.method_config.get("entropy-coeff", 0.0))
                    batch_entropies = [rollout.get("entropy", torch.tensor(0.0)) for _, _, _, rollout in batch_items]
                    mean_entropy = sum(batch_entropies) / max(1, len(batch_entropies))
                    total_loss = total_loss - entropy_coeff * mean_entropy

                    episode_loss_value = float(total_loss.detach().item())
                    total_loss.backward()
                    grad_clip = float(self.method_config.get("grad-clip-norm", 1.0))
                    torch.nn.utils.clip_grad_norm_(cvar_params, grad_clip)
                    cvar_optimizer.step()
                    pbar.set_postfix(ep=episode_offset + episode_idx + 1, phase="cvar_finetune", loss=episode_loss_value)

                target_gain_total_list: List[float] = []
                target_gain_acc_list: List[float] = []
                self._stochastic_editor_actions = False
                for sample in target_set:
                    rollout = self._evaluate_policy_pair(sample)
                    rollout["query"] = sample.get("query", "")
                    target_rollouts.append(rollout)
                    target_gain_total_list.append(rollout["gain_total"])
                    target_gain_acc_list.append(rollout["gain_acc"])
                    n = len(target_gain_total_list)
                    pbar.set_postfix(
                        ep=episode_offset + episode_idx + 1,
                        phase="cvar_target",
                        gain=float(sum(target_gain_total_list) / n),
                        gain_acc=float(sum(target_gain_acc_list) / n),
                    )
                    pbar.update(1)
                self._stochastic_editor_actions = True
                target_gain = float(sum(target_gain_total_list) / max(1, len(target_gain_total_list))) if target_gain_total_list else 0.0
                episode_summary = self._build_episode_summary(
                    episode_offset + episode_idx + 1, support_rollouts, target_rollouts
                )
                episode_summary["support"]["final_loss"] = float(episode_loss_value)
                episode_summary["target"]["avg_gain"] = float(target_gain)
                episode_summary["phase"] = "cvar_finetune"
                episode_summary["cvar_alpha"] = float(cvar_alpha)
                episode_summary["cvar_lambda"] = float(cvar_lambda)
                episode_summary["tail_metric"] = tail_metric
                episode_summary["tail_selected_count"] = int(tail_selected_count)
                episode_summary["tail_eta"] = float(tail_eta)
                episode_summary["cvar_objective"] = "empirical_ru_pg"
                episode_summary["cvar_update_scope"] = "editor_and_hypernet" if hypernet is not None else "editor_only"
                self._persist_episode_summary(episode_summary)
                pbar.set_postfix(
                    ep=episode_offset + episode_idx + 1,
                    phase="cvar_done",
                    loss=episode_loss_value,
                    gain=target_gain,
                )

            pbar.close()
            self._save_state()
            self.inference_flag = True
        finally:
            for p, rg in zip(self.prototype_selector.parameters(), prev_sel):
                p.requires_grad_(rg)
            for p, rg in zip(self.descriptor_text_projector.parameters(), prev_proj):
                p.requires_grad_(rg)
            for p, rg in zip(self.risk_editor.parameters(), prev_ed):
                p.requires_grad_(rg)
            if hypernet is not None:
                for p, rg in zip(hypernet.parameters(), prev_hyper):
                    p.requires_grad_(rg)
            self._stochastic_editor_actions = prev_stochastic_editor_actions

    def inference(self, sample):
        self.inference_flag = True

        rollout = self._run_policy(sample)

        if bool(self.method_config.get("probe-judge-only", False)):
            jr = rollout["judge_result"]
            effective_network = compute_effective_interaction_network(
                self.used_agents,
                self.nodes,
                self.adversarial_node_ids,
                is_decision_node=lambda n: n == self.decision_method,
            )
            return {
                "response": rollout.get("probe_response") or "",
                "agent_outputs": rollout["probe_outputs"],
                "judge_result": jr,
                "risk_scores": rollout["risks"],
                "uncertainty_scores": rollout["uncertainties"],
                "raw_risks": jr.get("raw_risks", []),
                "raw_uncertainties": jr.get("raw_uncertainties", []),
                "prototype_weights": [float(x) for x in rollout["final_alpha"].tolist()],
                "prototype_weights_prior": [float(x) for x in rollout["prior_alpha"].tolist()],
                "prototype_weights_final": [],
                "probe_response": rollout["probe_response"],
                "topology_prior": rollout["prior_adj"],
                "topology_new": rollout["new_adj"],
                "edit_actions": [],
                "adversarial_agent_indices": rollout["adversarial_agent_indices"],
                "adversarial_nodes": self.adversarial_node_ids,
                "num_adversarial": len(self.adversarial_node_ids),
                "total_nodes": self.num_truthful_agents,
                "effective_interaction_network": effective_network,
                "gate_values_prior": rollout["gate_values_prior"],
                "gate_values_final": rollout["gate_values_final"],
                "probe_judge_only": True,
                "final_selector_used": bool(rollout.get("final_selector_used", False)),
            }

        # Use the supernet's learned topology for the final round.
        # The hypernetwork refines the prior topology using probe-round
        # risk assessments to quarantine/isolate suspected adversaries
        # and strengthen trusted agent communication paths.
        self._build_runtime(
            rollout["new_adj"],
            probe_mode=False,
            forced_adversarial_indices=rollout["adversarial_agent_indices"],
        )
        # Use judge-provided absolute risk scores directly (no sum-to-1
        # normalization). Judge semantics are now per-agent unreliability
        # estimation instead of forced relative ranking.
        self._latest_agent_risks = list(rollout.get("risks") or [])
        self._probe_code_candidates: List[Dict[str, Any]] = []
        response, agent_outputs = self._execute_current_runtime(sample["query"])
        effective_network = compute_effective_interaction_network(
            self.used_agents,
            self.nodes,
            self.adversarial_node_ids,
            is_decision_node=lambda n: n == self.decision_method,
        )
        return {
            "response": response,
            "agent_outputs": agent_outputs,
            "judge_result": rollout["judge_result"],
            "risk_scores": rollout["risks"],
            "uncertainty_scores": rollout["uncertainties"],
            "prototype_weights": [float(x) for x in rollout["final_alpha"].tolist()],
            "prototype_weights_prior": [float(x) for x in rollout["prior_alpha"].tolist()],
            "prototype_weights_final": [],
            "probe_response": rollout["probe_response"],
            "topology_prior": rollout["prior_adj"],
            "topology_new": rollout["new_adj"],
            "edit_actions": rollout["edit_info"]["actions"],
            "adversarial_agent_indices": rollout["adversarial_agent_indices"],
            "adversarial_nodes": self.adversarial_node_ids,
            "num_adversarial": len(self.adversarial_node_ids),
            "total_nodes": self.num_truthful_agents,
            "effective_interaction_network": effective_network,
            "gate_values_prior": rollout["gate_values_prior"],
            "gate_values_final": rollout["gate_values_final"],
            "final_selector_used": bool(rollout.get("final_selector_used", False)),
        }

    def optimizing(self, val_dataset):
        if not val_dataset:
            return
        if bool(self.method_config.get("skip-optimization", False)):
            self.inference_flag = True
            self.episode_stats = []
            print(">> [RIGOR] skip-optimization=true; bypassing optimizing() entirely.", flush=True)
            return
        selector_supervised = bool(self.method_config.get("selector-supervised", True))
        editor_supervised = bool(self.method_config.get("editor-supervised", True))
        rl_finetune_enabled = bool(self.method_config.get("rl-finetune-enabled", False))
        if selector_supervised != editor_supervised:
            raise ValueError(
                "v1 requires selector-supervised and editor-supervised to be both true or both false."
            )

        if selector_supervised and editor_supervised:
            warmup_rounds = int(self.method_config.get("warmup-rounds-supervised", 0))
            cvar_rounds = int(self.method_config.get("cvar-finetune-rounds", 0))
            total_rounds = int(self.method_config.get("optimization-rounds", 1))
            if warmup_rounds <= 0:
                warmup_rounds = total_rounds
            self._optimizing_supervised(
                val_dataset,
                rounds_override=warmup_rounds,
                episode_offset=0,
            )
            if cvar_rounds > 0:
                self._optimizing_cvar_finetune(
                    val_dataset,
                    rounds=cvar_rounds,
                    episode_offset=warmup_rounds,
                )
            return
        if not rl_finetune_enabled:
            self.inference_flag = True
            self.episode_stats = []
            print(">> [RIGOR] rl-finetune-enabled=false; skipping RL optimizing().", flush=True)
            return
        self.inference_flag = False
        # Each validation-time optimization run should start with fresh stats
        # instead of appending to a previous execution on disk.
        self.episode_stats = []
        rounds = int(self.method_config.get("optimization-rounds", 1))
        support_updates = int(self.method_config.get("support-updates-per-episode", 1))
        support_size = min(int(self.method_config.get("support-size", 3)), len(val_dataset))
        target_size = min(int(self.method_config.get("target-size", max(1, support_size // 2))), len(val_dataset))
        cvar_alpha = self._cvar_tail_mass()
        cvar_lambda_full = float(self.method_config.get("cvar-lambda", 0.0))
        warmup_rounds = int(self.method_config.get("warmup-rounds", 0))
        delta_perf = float(self.method_config.get("delta-perf", 0.10))
        delta_novel = float(self.method_config.get("delta-novel", 0.15))

        total_steps = rounds * (support_updates * support_size + target_size)
        pbar = tqdm(
            total=total_steps,
            desc="episodic optimization",
            unit="step",
            leave=True,
        )
        for episode_idx in range(rounds):
            # warmup phase: pure mean-gain (cvar_lambda=0); CVaR activates after warmup_rounds
            cvar_lambda = 0.0 if episode_idx < warmup_rounds else cvar_lambda_full
            support_set, target_set = self._sample_episode(val_dataset)
            episode_loss_value = 0.0
            target_gain = 0.0
            support_rollouts: List[Dict[str, Any]] = []
            target_rollouts: List[Dict[str, Any]] = []

            for support_update_idx in range(support_updates):
                self.optimizer.zero_grad()

                # 1. collect rollouts; bank update deferred to step 2
                batch_items: List[tuple] = []   # (tail_score, log_prob, raw_reward, rollout)
                for sample in support_set:
                    log_prob, raw_reward, rollout = self._sample_rollout(sample, update_bank=False)
                    rollout["query"] = sample.get("query", "")
                    # Keep the CVaR tail criterion aligned with the reward used in the
                    # policy update. Ranking by gain_total here can over-focus on
                    # auxiliary structure/token terms even when accuracy does not improve.
                    tail_score = raw_reward
                    batch_items.append((tail_score, log_prob, raw_reward, rollout))
                    support_rollouts.append(rollout)
                    pbar.set_postfix(ep=episode_idx + 1, phase="support", collecting=True)
                    pbar.update(1)

                # 2. prototype bank: full-set update, unaffected by CVaR truncation
                for _, _, _, rollout in batch_items:
                    self._record_prototype_feedback(rollout["prior_alpha"], rollout["j_prior"])
                    self._record_prototype_feedback(rollout["final_alpha"], rollout["j_new"])
                    if rollout["gain_total"] >= delta_perf:
                        novelty = float(self._graph_distance(rollout["new_matrix"], rollout["prior_matrix"]).item())
                        if novelty >= delta_novel:
                            self._update_prototype_bank(rollout["new_matrix"], rollout["j_new"])

                # 3. batch reward-side advantage normalization (reduces REINFORCE variance)
                raw_rewards = [r for _, _, r, _ in batch_items]
                advantages = self._normalized_reward_advantages(raw_rewards)

                # 4. per-sample losses using normalized advantages
                per_losses = [-adv * log_prob for adv, (_, log_prob, _, _) in zip(advantages, batch_items)]

                # 5. empirical RU-CVaR tail over adaptation losses L_adapt=-R
                tail_indices, tail_eta, _ = self._cvar_tail_from_rewards(raw_rewards, cvar_alpha)

                # 6. mean + likelihood-ratio PG-CVaR mixed objective
                n_batch = len(per_losses)
                mean_loss = sum(per_losses) / max(1, n_batch)
                cvar_loss = (
                    sum(per_losses[idx] for idx in tail_indices) / max(1, len(tail_indices))
                    if tail_indices
                    else mean_loss
                )
                total_loss = (1.0 - cvar_lambda) * mean_loss + cvar_lambda * cvar_loss
                delta_reg = float(self.method_config.get("editor-delta-reg", 0.0))
                if delta_reg > 0.0:
                    delta_terms = [
                        rollout["editor_outputs"].get("hyper_delta_norm")
                        for _, _, _, rollout in batch_items
                        if isinstance(rollout.get("editor_outputs"), dict)
                        and rollout["editor_outputs"].get("hyper_delta_norm") is not None
                    ]
                    if delta_terms:
                        total_loss = total_loss + delta_reg * (sum(delta_terms) / len(delta_terms))

                # Entropy bonus encourages exploration, preventing premature policy collapse
                entropy_coeff = float(self.method_config.get("entropy-coeff", 0.0))
                batch_entropies = [rollout.get("entropy", torch.tensor(0.0)) for _, _, _, rollout in batch_items]
                mean_entropy = sum(batch_entropies) / max(1, len(batch_entropies))
                total_loss = total_loss - entropy_coeff * mean_entropy

                episode_loss_value = float(total_loss.detach().item())
                total_loss.backward()
                # Gradient clipping prevents catastrophic updates from noisy REINFORCE
                grad_clip = float(self.method_config.get("grad-clip-norm", 1.0))
                grad_params = (
                    list(self.prototype_selector.parameters())
                    + list(self.risk_editor.parameters())
                    + list(self.descriptor_text_projector.parameters())
                )
                if getattr(self, "editor_hypernet", None) is not None:
                    grad_params += list(self.editor_hypernet.parameters())
                torch.nn.utils.clip_grad_norm_(grad_params, grad_clip)
                self.optimizer.step()
                train_phase = "warmup" if episode_idx < warmup_rounds else "cvar"
                pbar.set_postfix(ep=episode_idx + 1, phase=train_phase, loss=episode_loss_value)

            target_gain_total_list: List[float] = []
            target_gain_acc_list: List[float] = []
            for sample in target_set:
                rollout = self._evaluate_policy_pair(sample)
                rollout["query"] = sample.get("query", "")
                target_rollouts.append(rollout)
                target_gain_total_list.append(rollout["gain_total"])
                target_gain_acc_list.append(rollout["gain_acc"])
                n = len(target_gain_total_list)
                avg_gain_acc = float(sum(target_gain_acc_list) / n)
                sorted_acc = sorted(target_gain_acc_list)
                tail_n = max(1, int(math.ceil(n * cvar_alpha)))
                tail_gain_acc = float(sum(sorted_acc[:tail_n]) / tail_n)
                pbar.set_postfix(
                    ep=episode_idx + 1,
                    phase="target",
                    gain=float(sum(target_gain_total_list) / n),
                    gain_acc=avg_gain_acc,
                    tail_acc=tail_gain_acc,
                )
                pbar.update(1)
            target_gain = float(sum(target_gain_total_list) / max(1, len(target_gain_total_list))) if target_gain_total_list else 0.0
            episode_summary = self._build_episode_summary(episode_idx + 1, support_rollouts, target_rollouts)
            episode_summary["support"]["final_loss"] = float(episode_loss_value)
            episode_summary["target"]["avg_gain"] = float(target_gain)
            episode_summary["cvar_tail_mass"] = float(cvar_alpha)
            episode_summary["cvar_objective"] = "empirical_ru_pg"
            episode_summary["tail_metric"] = "adapt_loss_topk"
            episode_summary["tail_selected_count"] = int(max(1, math.ceil(cvar_alpha * max(1, support_size))))
            self._persist_episode_summary(episode_summary)
            pbar.set_postfix(ep=episode_idx + 1, phase="done", loss=episode_loss_value, gain=target_gain)

        pbar.close()
        self._save_state()
        self.inference_flag = True
