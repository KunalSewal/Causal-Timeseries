"""
DAG Utilities

Helper functions for working with Directed Acyclic Graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List


class DAGValidator:
    """
    Validate and check DAG properties.
    """
    
    @staticmethod
    def is_dag(adjacency_matrix: np.ndarray) -> bool:
        """
        Check if adjacency matrix represents a valid DAG (no cycles).
        
        Args:
            adjacency_matrix: (n, n) adjacency matrix
            
        Returns:
            True if DAG, False if has cycles
        """
        G = nx.DiGraph(adjacency_matrix)
        return nx.is_directed_acyclic_graph(G)
    
    @staticmethod
    def count_edges(adjacency_matrix: np.ndarray) -> int:
        """Count number of edges in graph."""
        return np.count_nonzero(adjacency_matrix)


def visualize_dag(
    adjacency_matrix: np.ndarray,
    variable_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Causal DAG",
):
    """
    Visualize a causal DAG.
    
    Args:
        adjacency_matrix: (n, n) adjacency matrix where A[i,j] != 0 means i -> j
        variable_names: Names for variables (default: var_0, var_1, ...)
        save_path: Path to save figure (optional)
        title: Plot title
    """
    num_vars = adjacency_matrix.shape[0]
    
    if variable_names is None:
        variable_names = [f"var_{i}" for i in range(num_vars)]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, name in enumerate(variable_names):
        G.add_node(i, label=name)
    
    # Add edges
    for i in range(num_vars):
        for j in range(num_vars):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(i, j, weight=abs(adjacency_matrix[i, j]))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=1500,
        alpha=0.9
    )
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=[w * 3 for w in weights],
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw labels
    labels = {i: variable_names[i] for i in range(num_vars)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved DAG visualization to: {save_path}")
    
    plt.show()


def dag_to_adjacency(edges: List[tuple], num_vars: int) -> np.ndarray:
    """
    Convert list of edges to adjacency matrix.
    
    Args:
        edges: List of (source, target) or (source, target, weight) tuples
        num_vars: Number of variables
        
    Returns:
        Adjacency matrix
    """
    adj = np.zeros((num_vars, num_vars))
    
    for edge in edges:
        if len(edge) == 2:
            i, j = edge
            adj[i, j] = 1.0
        elif len(edge) == 3:
            i, j, weight = edge
            adj[i, j] = weight
    
    return adj


if __name__ == "__main__":
    # Example usage
    print("Testing DAG utilities...")
    
    # Create sample DAG
    adj = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    
    print("Is DAG:", DAGValidator.is_dag(adj))
    print("Number of edges:", DAGValidator.count_edges(adj))
    
    visualize_dag(adj, variable_names=['A', 'B', 'C', 'D'])
