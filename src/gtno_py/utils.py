import torch
import networkx as nx
import pandas as pd
from torch.utils.data import random_split
import random


def pretty_print_graph_data(
    batch_element: dict[str, torch.Tensor], print_node_features: bool = True, precision: int = 4
) -> None:
    """
    Pretty print a single batch element from the graph data.

    Args:
        batch_element (dict): A dictionary containing graph data with tensors.
    """
    assert isinstance(batch_element, dict), "batch_element must be a dictionary"

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


def get_data_split_indices_MD17_prescribed(
    split_number: int, val_ratio: float = 0.2
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits indices for train, validation, and test sets based on index files.

    Args:
        split_number (int): The split number (e.g., 1 for index_test_01 and index_train_01).
        val_ratio (float): The proportion of the training set to use for validation.

    Returns:
        Tuple[List[int], List[int], List[int]]: Train, validation, and test indices.
    """
    # Construct file names for the split
    test_file = f"data/splits/index_test_0{split_number}.csv"
    train_file = f"data/splits/index_train_0{split_number}.csv"

    # Load indices
    test_indices = pd.read_csv(test_file, header=None).iloc[:, 0].tolist()
    train_indices = pd.read_csv(train_file, header=None).iloc[:, 0].tolist()

    # Split train indices into train and validation subsets
    num_train = int((1 - val_ratio) * len(train_indices))
    num_val = len(train_indices) - num_train
    train_indices_split, val_indices_split = random_split(train_indices, [num_train, num_val])

    # Convert splits to lists (random_split returns Subset objects)
    train_indices_split = list(train_indices_split)
    val_indices_split = list(val_indices_split)

    return train_indices_split, val_indices_split, test_indices


def get_data_split_indices_custom(dataset_size: int) -> tuple[list[int], list[int], list[int]]:
    """
    Returns indices of size 500 for train, 2000 for val, 2000 for test,
    randomly chosen from dataset_size frames. Adjust as needed.
    """

    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    train_count = 500
    val_count = 2000
    test_count = 2000
    assert dataset_size >= train_count + val_count + test_count, "Dataset size is too small for the specified splits."

    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count : train_count + val_count]
    test_indices = all_indices[train_count + val_count : train_count + val_count + test_count]
    return train_indices, val_indices, test_indices
