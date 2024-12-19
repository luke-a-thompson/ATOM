import torch
import networkx as nx
from typing import Dict


def pretty_print_graph_data(
    batch_element: Dict[str, torch.Tensor], print_node_features: bool = True, precision: int = 4
) -> None:
    """
    Pretty print a single batch element from the graph data.

    Args:
        batch_element (dict): A dictionary containing graph data with tensors.
    """
    assert isinstance(batch_element, Dict), "batch_element must be a dictionary"

    torch.set_printoptions(precision=precision)  # Set precision for tensor printing

    print("\n===== Graph Batch Keys =====")
    # Handle types that might not have a `.shape` attribute
    print(
        "\nBatch Keys:\n"
        + "\n".join(
            f" - {key} (shape: {value.shape if isinstance(value, torch.Tensor) else type(value)})"
            for key, value in batch_element.items()
        )
    )

    print("\n===== Graph Batch Data =====")

    # Timestep
    timestep = batch_element.get("timestep")
    if timestep is not None:
        print(f"Timestep: {timestep.item()}")

    # Energy
    energy = batch_element.get("energy")
    if energy is not None:
        print(f"Energy: {energy.item():.{precision}f}")

    # Nuclear Charges
    nuclear_charges = batch_element.get("nuclear_charges")
    if nuclear_charges is not None:
        print("\nNuclear Charges:")
        print(" ", nuclear_charges.squeeze())

    # Coordinates
    coords = batch_element.get("coords")
    if coords is not None:
        print("\nCoordinates (x, y, z):")
        for idx, coord in enumerate(coords.squeeze()):
            print(f" Atom {idx+1}: {coord}")

    # Forces
    forces = batch_element.get("forces")
    if forces is not None:
        print("\nForces (fx, fy, fz):")
        for idx, force in enumerate(forces.squeeze()):
            print(f" Atom {idx+1}: {force}")

    # Edge Features
    edge_features = batch_element.get("edge_features")
    if edge_features is not None:
        print("\nEdge Features:")
        print(edge_features.squeeze())

    # Node Features
    node_features = batch_element.get("node_features")
    if node_features is not None and print_node_features is True:
        print("\nNode Features:")
        node_features_list = node_features.squeeze()
        num_nodes = nuclear_charges.size(1) if nuclear_charges is not None else 0

        # Select random atom index
        random_atom_idx = int(torch.randint(0, num_nodes - 1, (1,)).item())
        print(f" Random Atom {random_atom_idx+1} node features: {node_features_list[random_atom_idx]}")

        # print(" Additional Features:")
        # for idx, feat in enumerate(node_features_list[num_nodes:]):
        #     print(f"  Feature {idx+1}: {feat:.{precision}f}")

    print("\n==============================")


def draw_graph(nx_graph: nx.Graph, filename: str = "debug_graph.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph, seed=42)
    labels = {node: f"{data['features']}" for node, data in nx_graph.nodes(data=True)}
    nx.draw(
        nx_graph,
        pos,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )

    # Add edge labels showing distances
    edge_labels = nx.get_edge_attributes(nx_graph, "distance")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)

    plt.title("2-Hop Graph Structure")
    plt.savefig(filename)
    plt.close()
