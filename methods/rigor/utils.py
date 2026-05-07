import hashlib
import math
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ID / seed helpers
# ---------------------------------------------------------------------------

def make_id(length: int = 4) -> str:
    return uuid.uuid4().hex[:length]


def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Text / numeric helpers
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: Any) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", normalize_text(text).lower())


def sigmoid_like(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def extract_choice_letter(text: str) -> str:
    """Extract the answer choice letter (A-J) from a response.

    Priority order:
    1. "answer is (X)" or "answer is X" patterns (most explicit)
    2. Last standalone (X) in parentheses
    3. Last standalone letter A-D on its own line
    4. First standalone letter A-J (fallback)
    """
    upper = text.upper()
    # 1. Explicit "answer is" patterns
    m = re.findall(r"(?:ANSWER|CORRECT)\s*(?:IS|:)\s*\(?([A-J])\)?", upper)
    if m:
        return m[-1]
    # 2. Last parenthesized letter like (A), (B), etc.
    m = re.findall(r"\(([A-J])\)", upper)
    if m:
        return m[-1]
    # 3. Last standalone letter on its own line or at end
    m = re.findall(r"(?:^|\n)\s*([A-J])\s*\.?\s*$", upper, re.MULTILINE)
    if m:
        return m[-1]
    # 4. Fallback: last standalone letter A-J
    m = re.findall(r"\b([A-J])\b", upper)
    if m:
        return m[-1]
    return ""


def extract_last_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def extract_first_json_object(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""

# ---------------------------------------------------------------------------
# Prototype record
# ---------------------------------------------------------------------------

@dataclass
class PrototypeRecord:
    name: str
    matrix: List[List[float]]
    score: float = 0.0
    visits: float = 0.0
    source: str = "manual"
    note: str = ""


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """Two-layer MLP with skip connection for gradient flow.

    Projects the input to hidden_dim via a skip linear, then adds the
    residual from a LayerNorm+ReLU+Dropout block. This prevents gradient
    vanishing when compressing high-dim descriptors (e.g. 206→64).
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.skip = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


def mlp_backbone(feature_dim: int, hidden_dim: int, dropout: float = 0.1) -> nn.Module:
    return ResidualMLP(feature_dim, hidden_dim, dropout)


class PrototypeSelector(nn.Module):
    """Small selector network for strict top-1 prototype routing."""

    def __init__(self, in_dim: int, hidden_dim: int, num_prototypes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_prototypes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskEditor(nn.Module):
    """Node/edge local editor for quarantine/mask logits."""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _apply_film(
        hidden: torch.Tensor,
        gamma: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
    ) -> torch.Tensor:
        if gamma is None or beta is None:
            return hidden
        while gamma.dim() < hidden.dim():
            gamma = gamma.unsqueeze(0)
        while beta.dim() < hidden.dim():
            beta = beta.unsqueeze(0)
        return hidden * (1.0 + scale * gamma) + scale * beta

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        film: Optional[Dict[str, torch.Tensor]] = None,
        delta_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        film = film or {}
        node_hidden = self.node_mlp[1](self.node_mlp[0](node_feats))
        node_hidden = self._apply_film(
            node_hidden,
            film.get("node_gamma"),
            film.get("node_beta"),
            float(delta_scale),
        )
        edge_hidden = self.edge_mlp[1](self.edge_mlp[0](edge_feats))
        edge_hidden = self._apply_film(
            edge_hidden,
            film.get("edge_gamma"),
            film.get("edge_beta"),
            float(delta_scale),
        )
        quarantine_logits = self.node_mlp[2](node_hidden).squeeze(-1)
        mask_logits = self.edge_mlp[2](edge_hidden).squeeze(-1)
        return {
            "quarantine_logits": quarantine_logits,
            "mask_logits": mask_logits,
        }


class EditorHyperNetwork(nn.Module):
    """Generate per-sample FiLM deltas for the risk editor hidden states."""

    def __init__(self, context_dim: int, hidden_dim: int, editor_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.editor_hidden_dim = int(editor_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 4 * editor_hidden_dim),
        )

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        if context.dim() != 1:
            context = context.view(-1)
        raw = self.net(context)
        node_gamma, node_beta, edge_gamma, edge_beta = torch.chunk(raw, 4, dim=-1)
        return {
            "node_gamma": node_gamma,
            "node_beta": node_beta,
            "edge_gamma": edge_gamma,
            "edge_beta": edge_beta,
            "delta_norm": torch.mean(raw.pow(2)),
        }


class DescriptorTextProjector(nn.Module):
    """Frozen MiniLM 384-d → trainable bottleneck for hypernetwork descriptor (query + dataset profile)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        if hidden is not None and int(hidden) > 0:
            h = int(hidden)
            self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, out_dim))
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_runtime_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
