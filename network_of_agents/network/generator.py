"""
Network generator using the consolidated mathematics module.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.mathematics import create_network, get_network_info
from .graph_model import NetworkModel

def create_network_model(topology: str, topology_params: Dict[str, Any], 
                        random_seed: Optional[int] = None) -> NetworkModel:
    """Create a NetworkModel with the specified topology"""
    adjacency_matrix = create_network(topology, topology_params, random_seed)
    n_agents = adjacency_matrix.shape[0]
    network_model = NetworkModel(n_agents, random_seed)
    network_model.update_adjacency_matrix(adjacency_matrix)
    return network_model

def get_network_params(topology: str, n_agents: int) -> Dict[str, Any]:
    """Get default parameters for a network topology"""
    params = {"n_agents": n_agents}
    
    if topology == "smallworld":
        params.update({"k": 4, "beta": 0.1})
    elif topology == "scalefree":
        params.update({"m": 2})
    elif topology == "random":
        params.update({"p": 0.1})
    elif topology == "echo":
        params.update({"n_communities": 2, "p_intra": 0.3, "p_inter": 0.05})
    elif topology == "karate":
        params = {"n_agents": 34}
    elif topology == "stubborn":
        params.update({"k": 4, "beta": 0.1})
    elif topology == "complete":
        params = {"n_agents": n_agents}
    
    return params
