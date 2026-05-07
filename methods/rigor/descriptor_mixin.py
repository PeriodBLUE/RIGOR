"""Task/dataset descriptors and hypernetwork input features."""

import hashlib
import json
import os
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import math
from . import prompts
from .utils import (
    extract_first_json_object,
    normalize_text,
    safe_float,
    tokenize,
)


_SENTENCE_TRANSFORMER_LOAD_LOCK = threading.Lock()


class DescriptorMixin:
    def _hash_text_tensor(self, text: str) -> torch.Tensor:
        """Dependency-free fallback embedding when MiniLM backend is unavailable."""
        device = self._get_descriptor_device()
        vec = self._hashed_bow_embedding(normalize_text(text), self._minilm_embedding_dim)
        return torch.tensor(vec, dtype=torch.float32, device=device)

    def _build_task_profile(self, query: str) -> List[float]:
        """Heuristic flags for shaping probe queries only (not used in the hypernetwork descriptor)."""
        query = normalize_text(query)
        return [
            float(self.dataset_mode == "math"),
            float(self.dataset_mode == "medmcqa"),
            float("medical" in query.lower() or "clinical" in query.lower()),
            float("choose the correct answer" in query.lower()),
            float(any(token in query.lower() for token in ("diagnosis", "treatment", "patient", "drug"))),
            float(any(ch.isdigit() for ch in query)),
        ]

    def _descriptor_text_proj_dim(self) -> int:
        return int(self.method_config.get("descriptor-text-proj-dim", 64))

    def _descriptor_minilm_model_name(self) -> str:
        return str(self.method_config.get("descriptor-minilm-model", "all-MiniLM-L6-v2"))

    def _dataset_profile_cache_path(self) -> str:
        return os.path.join(self.state_dir, "dataset_profile_descriptor.json")

    def _get_descriptor_device(self) -> torch.device:
        return next(self.prototype_selector.parameters()).device

    def _lazy_sentence_transformer(self) -> Any:
        if getattr(self, "_use_hash_text_encoder", False):
            return None
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                # Keep inference runnable even when local HF/transformers versions conflict.
                self._use_hash_text_encoder = True
                return None
            try:
                with _SENTENCE_TRANSFORMER_LOAD_LOCK:
                    st = SentenceTransformer(self._descriptor_minilm_model_name())
                    st.eval()
                    for param in st.parameters():
                        param.requires_grad_(False)
                    self._sentence_transformer = st.to(self._get_descriptor_device())
            except Exception:
                self._use_hash_text_encoder = True
                return None
        dev = self._get_descriptor_device()
        if next(self._sentence_transformer.parameters()).device != dev:
            try:
                with _SENTENCE_TRANSFORMER_LOAD_LOCK:
                    self._sentence_transformer = self._sentence_transformer.to(dev)
            except Exception:
                self._use_hash_text_encoder = True
                self._sentence_transformer = None
                return None
        return self._sentence_transformer

    def _minilm_encode_detached(self, text: str) -> torch.Tensor:
        device = self._get_descriptor_device()
        model = self._lazy_sentence_transformer()
        if model is None:
            return self._hash_text_tensor(text)
        payload = (text or "").strip() or " "
        with torch.no_grad():
            vec = model.encode(payload, convert_to_tensor=True, show_progress_bar=False)
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        vec = vec.detach().float().view(-1).to(device)
        if vec.shape[0] != self._minilm_embedding_dim:
            return self._hash_text_tensor(text)
        # sentence_transformers may produce inference-mode tensors that cannot be saved
        # for backward; clone() yields a regular tensor.
        return vec.clone()

    def _fallback_dataset_description_text(self) -> str:
        return (
            f"Dataset {self.dataset_name} under mode {self.dataset_mode} with {self.num_truthful_agents} "
            f"role-ordered agents. Tasks require multi-step reasoning; evaluation stresses correctness when "
            f"exactly one agent may be adversarial."
        )

    def _llm_dataset_description(self) -> str:
        try:
            user = prompts.DATASET_DESCRIPTOR_USER_PROMPT.format(
                dataset_name=self.dataset_name,
                dataset_mode=self.dataset_mode,
                num_agents=self.num_truthful_agents,
                roles=", ".join(self.roles),
            )
            raw = self.call_llm(
                messages=[
                    {"role": "system", "content": prompts.DATASET_DESCRIPTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            candidate = extract_first_json_object(raw)
            parsed = json.loads(candidate) if candidate else {}
            desc = normalize_text(parsed.get("description", "")) if isinstance(parsed, dict) else ""
            if len(desc) >= 12:
                return desc
        except Exception:
            pass
        return normalize_text(self._fallback_dataset_description_text())

    def _save_dataset_profile_cache(self, description: str, minilm_embedding: List[float], minilm_model: str) -> None:
        try:
            with open(self._dataset_profile_cache_path(), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_name": self.dataset_name,
                        "dataset_mode": self.dataset_mode,
                        "minilm_model": minilm_model,
                        "minilm_dim": len(minilm_embedding),
                        "description": description,
                        "minilm_embedding": minilm_embedding,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    def _ensure_dataset_profile_minilm_vector(self) -> torch.Tensor:
        """LLM dataset description → frozen MiniLM 384-d; result is detached on the same device as hypernetwork."""
        device = self._get_descriptor_device()
        model_tag = self._descriptor_minilm_model_name()
        dim_expect = self._minilm_embedding_dim

        if (
            self._dataset_profile_minilm_cpu is not None
            and self._dataset_profile_text
            and self._dataset_profile_minilm_cpu.shape[0] == dim_expect
        ):
            return self._dataset_profile_minilm_cpu.to(device)

        path = self._dataset_profile_cache_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if (
                    data.get("dataset_name") == self.dataset_name
                    and data.get("dataset_mode") == self.dataset_mode
                    and data.get("minilm_model") == model_tag
                    and int(data.get("minilm_dim", -1)) == dim_expect
                ):
                    emb = data.get("minilm_embedding")
                    if isinstance(emb, list) and len(emb) == dim_expect:
                        self._dataset_profile_text = str(data.get("description", ""))
                        t_cpu = torch.tensor([float(x) for x in emb], dtype=torch.float32)
                        self._dataset_profile_minilm_cpu = t_cpu
                        return t_cpu.to(device)
            except Exception:
                pass

        text = self._llm_dataset_description()
        self._dataset_profile_text = text
        raw = self._minilm_encode_detached(text)
        self._dataset_profile_minilm_cpu = raw.detach().cpu()
        self._save_dataset_profile_cache(text, raw.detach().cpu().tolist(), model_tag)
        return raw

    def _selector_dataset_flags(self, sample: Optional[Dict[str, Any]] = None) -> List[float]:
        sample = sample or {}
        query = normalize_text(sample.get("query", ""))
        return [
            float(self.dataset_mode == "math"),
            float(self.dataset_mode == "medmcqa"),
            float("medical" in query.lower() or "clinical" in query.lower()),
            float(any(ch.isdigit() for ch in query)),
            float(any(token in query.lower() for token in ("diagnosis", "treatment", "patient", "drug"))),
            float("choose the correct answer" in query.lower()),
        ]

    def _graph_stats(self, matrix: List[List[float]]) -> List[float]:
        n = len(matrix)
        density = sum(sum(row) for row in matrix) / max(1, n * (n - 1) / 2)
        out_degrees = [sum(row) for row in matrix]
        in_degrees = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
        max_out = max(out_degrees) if out_degrees else 0.0
        max_in = max(in_degrees) if in_degrees else 0.0
        sinks = sum(1 for deg in out_degrees if deg == 0)
        return [density, max_out / max(1, n - 1), max_in / max(1, n - 1), sinks / max(1, n)]

    def _hashed_bow_embedding(self, text: str, dim: int) -> List[float]:
        import hashlib
        buckets = [0.0] * dim
        tokens = tokenize(text)
        if not tokens:
            return buckets
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % dim
            buckets[bucket] += 1.0
        total = sum(buckets)
        return [value / max(1.0, total) for value in buckets]

    def _encode_role_features(self) -> List[float]:
        features: List[float] = []
        denom = max(1, self.num_truthful_agents - 1)
        role_vocab = getattr(self, "role_vocabulary", None) or []
        for idx, role in enumerate(self.roles):
            one_hot = [1.0 if role == role_name else 0.0 for role_name in role_vocab]
            features.extend(one_hot + [idx / denom])
        return features

    def _encode_relation_features(self, prior_matrix: List[List[float]]) -> List[float]:
        n = len(prior_matrix)
        out_degrees = [sum(row) for row in prior_matrix]
        in_degrees = [sum(prior_matrix[i][j] for i in range(n)) for j in range(n)]
        features: List[float] = []
        norm = max(1.0, n - 1)
        for idx in range(n):
            features.extend(
                [
                    out_degrees[idx] / norm,
                    in_degrees[idx] / norm,
                    float(in_degrees[idx] == 0.0),
                    float(out_degrees[idx] == 0.0),
                ]
            )
        return features

    def _encode_prior_features(
        self,
        prior_matrix: List[List[float]],
        risks: Optional[List[float]] = None,
    ) -> List[float]:
        graph_stats = self._graph_stats(prior_matrix)
        diversity = self._diversity_bonus(prior_matrix)
        structural = self._struct_regularization(prior_matrix, risks)
        distances = [float(self._graph_distance(prior_matrix, proto.matrix).item()) for proto in self.prototype_bank]
        flat_edges = [prior_matrix[i][j] for i, j in self.edge_pairs]
        return graph_stats + [
            diversity,
            structural,
            min(distances) if distances else 0.0,
            max(distances) if distances else 0.0,
        ] + flat_edges

    def _selector_features(self, query: str, sample: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        sample = sample or {}
        device = self._get_descriptor_device()
        q_raw = self._minilm_encode_detached(normalize_text(query))
        query_embed = self.descriptor_text_projector(q_raw)
        flags = self._selector_dataset_flags({"query": query, **sample})
        flag_tensor = torch.tensor(flags, dtype=torch.float32, device=device)
        selector_x = torch.cat([query_embed, flag_tensor], dim=0)
        expected = int(self.method_config.get("selector-query-dim", 8)) + len(flags)
        if selector_x.shape[0] < expected:
            selector_x = F.pad(selector_x, (0, expected - selector_x.shape[0]))
        elif selector_x.shape[0] > expected:
            selector_x = selector_x[:expected]
        return selector_x

    def _editor_features(
        self,
        prior_matrix: List[List[float]],
        risks: List[float],
        uncertainties: List[float],
    ) -> Any:
        n = self.num_truthful_agents
        risks = list(risks[:n]) + [0.0] * max(0, n - len(risks))
        uncertainties = list(uncertainties[:n]) + [0.0] * max(0, n - len(uncertainties))
        out_degrees = [sum(prior_matrix[i][j] for j in range(n)) for i in range(n)]
        in_degrees = [sum(prior_matrix[i][j] for i in range(n)) for j in range(n)]
        norm = max(1.0, n - 1.0)
        node_feats: List[List[float]] = []
        for i in range(n):
            node_feats.append(
                [
                    float(risks[i]),
                    float(uncertainties[i]),
                    float(out_degrees[i] / norm),
                    float(in_degrees[i] / norm),
                    float(out_degrees[i] == 0.0),
                    float(in_degrees[i] == 0.0),
                ]
            )
        edge_feats: List[List[float]] = []
        for i, j in self.edge_pairs:
            edge_feats.append(
                [
                    float(prior_matrix[i][j]),
                    float(risks[i]),
                    float(risks[j]),
                    float(uncertainties[i]),
                    float(uncertainties[j]),
                    float(max(risks[i], risks[j])),
                    float(abs(risks[i] - risks[j])),
                ]
            )
        device = self._get_descriptor_device()
        return (
            torch.tensor(node_feats, dtype=torch.float32, device=device),
            torch.tensor(edge_feats, dtype=torch.float32, device=device),
        )

    def _editor_hyper_context(
        self,
        query: str,
        sample: Optional[Dict[str, Any]],
        prior_matrix: List[List[float]],
        risks: List[float],
        uncertainties: List[float],
    ) -> torch.Tensor:
        """Context vector used by the editor hypernetwork to generate FiLM deltas."""
        n = self.num_truthful_agents
        risks = list(risks[:n]) + [0.0] * max(0, n - len(risks))
        uncertainties = list(uncertainties[:n]) + [0.0] * max(0, n - len(uncertainties))
        flat_edges = [float(prior_matrix[i][j]) for i, j in self.edge_pairs]
        graph_stats = self._graph_stats(prior_matrix)
        selector_x = self._selector_features(query=query, sample=sample or {})
        extras = torch.tensor(
            graph_stats + risks + uncertainties + flat_edges,
            dtype=torch.float32,
            device=selector_x.device,
        )
        context = torch.cat([selector_x, extras], dim=0)
        expected = int(getattr(self, "editor_hyper_context_dim", context.shape[0]))
        if context.shape[0] < expected:
            context = F.pad(context, (0, expected - context.shape[0]))
        elif context.shape[0] > expected:
            context = context[:expected]
        return context
