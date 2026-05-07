import yaml
from typing import Any, Dict, List

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def handle_retry_error(retry_state):
    exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
    if exc is not None:
        print(f"Retry failed: {type(exc).__name__}: {exc}")
        raise exc
    raise RuntimeError("Retry failed without an exception")


def compute_effective_interaction_network(used_agents: List[Any], nodes: Dict[str, Any],
                                         adversarial_node_ids: List[str],
                                         is_decision_node: Any) -> Dict[str, Any]:
    """
    Compute Effective Interaction Network: W_eff = A * diag(V).
    A[i,j] = 1 if there is an edge from some node in agent j to some node in agent i (agent-level adjacency).
    V[k] = 1 if agent k is good (no adversarial node), else 0.
    W_eff[i,j] = A[i,j] * V[j]: only edges from good agents count.

    Returns JSON-serializable dict with keys: A, V, W_eff, N, agent_node_ids.
    """
    N = len(used_agents)
    if N == 0:
        return {"N": 0, "A": [], "V": [], "W_eff": [], "agent_node_ids": []}

    # node_id -> agent index (0..N-1)
    node_to_agent: Dict[str, int] = {}
    for idx, agent in enumerate(used_agents):
        for nid in agent.nodes:
            node_to_agent[nid] = idx

    # Build physical adjacency A: A[i,j]=1 if exists edge from agent j to agent i
    A = [[0] * N for _ in range(N)]
    for idx, agent in enumerate(used_agents):
        for nid in agent.nodes:
            node = nodes.get(nid)
            if node is None:
                continue
            for succ in getattr(node, "successors", []):
                succ_id = getattr(succ, "id", None)
                if succ_id is None:
                    continue
                if is_decision_node(succ):
                    continue
                j = node_to_agent.get(nid)
                i = node_to_agent.get(succ_id)
                if i is not None and j is not None:
                    A[i][j] = 1

    # V[k] = 1 if agent k is good (no node in agent k is adversarial), else 0
    adversarial_set = set(adversarial_node_ids)
    V = []
    for agent in used_agents:
        agent_has_bad = any(nid in adversarial_set for nid in agent.nodes)
        V.append(0 if agent_has_bad else 1)

    # W_eff[i,j] = A[i,j] * V[j]
    W_eff = [[A[i][j] * V[j] for j in range(N)] for i in range(N)]

    # Optional: list of node ids per agent (for reference)
    agent_node_ids = [[nid for nid in agent.nodes] for agent in used_agents]

    return {
        "N": N,
        "A": A,
        "V": V,
        "W_eff": W_eff,
        "agent_node_ids": agent_node_ids,
    }
