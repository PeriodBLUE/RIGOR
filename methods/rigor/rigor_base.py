"""Core RIGOR multi-agent method: wires mixins and constructs trainable modules."""

import os
from itertools import chain
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..mas_base import MAS
from ..utils import load_config
from . import prompts
from .descriptor_mixin import DescriptorMixin
from .graph_nodes import FinalDecision, Node, RoleAwareReasoner, SingleNodeAgent
from .hyper_topology_mixin import HyperTopologyMixin
from .prototype_state_mixin import PrototypeStateMixin
from .runtime_graph_mixin import RuntimeGraphMixin
from .scoring_training_mixin import ScoringTrainingMixin
from .utils import (
    DescriptorTextProjector,
    EditorHyperNetwork,
    PrototypeRecord,
    PrototypeSelector,
    RiskEditor,
    get_runtime_device,
    seed_everything,
)
from .role_tooling import DATASET_ROLE_TOOL_SPECS, TOOL_DESCRIPTIONS, TOOL_REGISTRY

DATASET_MODE_CONFIGS = {
    "math": "config_math",
    "medmcqa": "config_medmcqa",
}


# MAS must be listed last so mixin `inference` / `optimizing` override the base `MAS`
# implementations (MAS defines stub `optimizing` and default `inference` before mixins in MRO).
class RigorBase(
    PrototypeStateMixin,
    HyperTopologyMixin,
    DescriptorMixin,
    RuntimeGraphMixin,
    ScoringTrainingMixin,
    MAS,
):
    dataset_mode = "math"

    @staticmethod
    def _move_module_to_device(module: nn.Module, device: torch.device) -> nn.Module:
        """Move module to device, handling meta tensors on newer PyTorch versions."""
        try:
            return module.to(device)
        except NotImplementedError as exc:
            if "meta tensor" not in str(exc):
                raise
            module = module.to_empty(device=device)
            for child in module.modules():
                reset_fn = getattr(child, "reset_parameters", None)
                if callable(reset_fn):
                    reset_fn()
            return module

    def __init__(self, general_config, method_config_name=None):
        MAS.__init__(self, general_config)
        user_seed = general_config.get("random_seed", None)
        if user_seed is not None:
            seed_everything(int(user_seed))
        expected_config = DATASET_MODE_CONFIGS.get(self.dataset_mode)
        if expected_config is None:
            raise ValueError(f"Unsupported RIGOR dataset_mode: {self.dataset_mode}")
        if method_config_name in (None, "", "config"):
            method_config_name = expected_config
        if method_config_name != expected_config:
            raise ValueError(
                f"RIGOR {self.dataset_mode} requires `{expected_config}`; got `{method_config_name}`."
            )
        self.method_config_name = method_config_name
        self.method_config = load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml")
        )
        overrides = general_config.get("method_config_overrides")
        if isinstance(overrides, dict) and overrides:
            self.method_config = {**self.method_config, **overrides}
        self.dataset_name = general_config["test_dataset_name"]
        self.num_truthful_agents = int(self.method_config["num-truthful-agent"])
        if self.dataset_mode not in DATASET_ROLE_TOOL_SPECS:
            raise ValueError(f"Unsupported RIGOR dataset_mode: {self.dataset_mode}")
        dataset_spec = DATASET_ROLE_TOOL_SPECS[self.dataset_mode]
        self.role_pool = list(dataset_spec.role_pool)
        self.roles = dataset_spec.build_roles(self.num_truthful_agents)
        self.role_tools = {role: list(tools) for role, tools in dataset_spec.role_tools.items()}
        self.role_vocabulary = list(dict.fromkeys(self.role_pool + self.roles))
        self.tools = {name: TOOL_REGISTRY[name] for name in TOOL_REGISTRY.keys()}
        self.tool_descriptions = dict(TOOL_DESCRIPTIONS)
        self.edge_pairs = [(i, j) for i in range(self.num_truthful_agents) for j in range(i + 1, self.num_truthful_agents)]
        self.hidden_dim = int(self.method_config.get("editor-hidden-dim", 16))
        self.state_dir = os.path.join(
            "results",
            self.dataset_name,
            "rigor",
            self.model_name.replace("/", "_"),
            str(self.method_config_name).replace("/", "_"),
        )
        os.makedirs(self.state_dir, exist_ok=True)
        self._minilm_embedding_dim = int(self.method_config.get("descriptor-minilm-dim", 384))
        proj_hidden = self.method_config.get("descriptor-text-proj-hidden")
        proj_hidden_int: Optional[int] = None
        if proj_hidden is not None and str(proj_hidden).strip() not in ("", "0"):
            proj_hidden_int = int(proj_hidden)
        self.descriptor_text_projector = DescriptorTextProjector(
            in_dim=self._minilm_embedding_dim,
            out_dim=int(self.method_config.get("selector-query-dim", 8)),
            hidden=proj_hidden_int,
        )
        self.prototype_bank: List[PrototypeRecord] = self._init_prototype_bank()
        selector_in_dim = int(self.method_config.get("selector-query-dim", 8)) + 6
        self.prototype_selector = PrototypeSelector(
            in_dim=selector_in_dim,
            hidden_dim=int(self.method_config.get("selector-hidden-dim", 16)),
            num_prototypes=len(self.prototype_bank),
        )
        self.editor_hyper_context_dim = (
            selector_in_dim
            + 4
            + 2 * self.num_truthful_agents
            + len(self.edge_pairs)
        )
        self.risk_editor = RiskEditor(
            node_feat_dim=int(self.method_config.get("editor-node-feat-dim", 6)),
            edge_feat_dim=int(self.method_config.get("editor-edge-feat-dim", 7)),
            hidden_dim=int(self.method_config.get("editor-hidden-dim", 16)),
        )
        self.editor_hypernet = None
        if bool(self.method_config.get("editor-hypernet-enabled", True)):
            mode = str(self.method_config.get("editor-hypernet-mode", "film")).strip().lower()
            if mode != "film":
                raise ValueError(f"Unsupported editor-hypernet-mode: {mode}. v1 supports film only.")
            self.editor_hypernet = EditorHyperNetwork(
                context_dim=self.editor_hyper_context_dim,
                hidden_dim=int(self.method_config.get("editor-hypernet-hidden-dim", 32)),
                editor_hidden_dim=int(self.method_config.get("editor-hidden-dim", 16)),
                dropout=float(self.method_config.get("editor-hypernet-dropout", 0.0)),
            )
        runtime_device = get_runtime_device()
        self.prototype_selector = self._move_module_to_device(self.prototype_selector, runtime_device)
        self.risk_editor = self._move_module_to_device(self.risk_editor, runtime_device)
        if self.editor_hypernet is not None:
            self.editor_hypernet = self._move_module_to_device(self.editor_hypernet, runtime_device)
        self.descriptor_text_projector = self._move_module_to_device(self.descriptor_text_projector, runtime_device)

        lr = float(self.method_config.get("learning-rate", 1e-3))
        optim_params = list(
            chain(
                self.prototype_selector.parameters(),
                self.risk_editor.parameters(),
                self.descriptor_text_projector.parameters(),
            )
        )
        if self.editor_hypernet is not None:
            optim_params.extend(self.editor_hypernet.parameters())
        self.optimizer = torch.optim.Adam(
            optim_params,
            lr=lr,
        )
        self.prompts = prompts
        self.memory: Dict[str, List[Dict[str, Any]]] = {}
        self.nodes: Dict[str, Node] = {}
        self.graphs: List[Any] = []
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.used_agents: List[SingleNodeAgent] = []
        self.decision_method: Optional[FinalDecision] = None
        self.adversarial_node_ids: List[str] = []
        self.current_adversarial_agent_indices: List[int] = []
        self.inference_flag = True
        self.episode_stats: List[Dict[str, Any]] = []
        self._dataset_profile_text: Optional[str] = None
        self._dataset_profile_minilm_cpu: Optional[torch.Tensor] = None
        self._sentence_transformer: Any = None
        self._load_state_if_available()

    def get_tools_for_role(self, role_name: str) -> List[str]:
        return list(self.role_tools.get(role_name, []))

    def describe_tool(self, tool_name: str) -> str:
        return self.tool_descriptions.get(tool_name, tool_name)

    def run_tool(self, tool_name: str, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        fn = self.tools.get(tool_name)
        if fn is None:
            return f"Unknown tool: {tool_name}"
        try:
            return str(fn(args or {}, context or {}))
        except Exception as exc:
            return f"Tool {tool_name} execution error: {exc}"
