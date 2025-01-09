import numpy as np
import torch
import pickle as pkl
import os
from enum import Enum
from typing import final
import numpy.typing as npt


@final
class DataPartition(str, Enum):
    train = "train"
    val = "val"
    test = "test"


@final
class MoleculeType(str, Enum):
    aspirin = "aspirin"
    benzene_old = "benzene_old"
    ethanol = "ethanol"
    malonaldehyde = "malonaldehyde"
    naphthalene = "naphthalene"
    salicylic = "salicylic"
    toluene = "toluene"
    uracil = "uracil"


class MD17Dataset:
    """
    MD17 Dataset

    """

    def __init__(
        self,
        partition: DataPartition,
        max_samples: int,
        delta_frame: int,
        data_dir: str,
        split_dir: str,
        molecule_type: MoleculeType,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
    ):
        """
        Args:
            partition (str): The partition to load ('train', 'val', 'test').
            max_samples (int): The maximum number of samples to load into the initial frame.
            delta_frame (int): The number of frames to skip between the initial and target frames.
            data_dir (str): The directory to load the data from.
            split_dir (str): The directory to load or store splits.
            molecule_type (str): The type of molecule to load ('aspirin', 'benzene_old', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil').
            train_par (float): The percentage of the data to use for training.
            val_par (float): The percentage of the data to use for validation.
            test_par (float): The percentage of the data to use for testing.
        """

        # setup a split, tentative setting
        train_par, val_par, test_par = train_par, val_par, test_par
        full_dir = os.path.join(data_dir + "md17_" + molecule_type + ".npz")
        split_dir = os.path.join(split_dir + molecule_type + "_split.pkl")
        data: np.lib.npyio.NpzFile = np.load(full_dir)

        if molecule_type in MoleculeType.__members__.values():
            self.molecule_type: MoleculeType = molecule_type
        else:
            raise ValueError(f"Invalid molecule type: {molecule_type}, select from one of {MoleculeType.__members__.keys()}")
        x: npt.NDArray[np.float64] = data["R"]
        v: npt.NDArray[np.float64] = x[1:] - x[:-1]
        x = x[:-1]

        # Load or generate split
        split = self._get_or_generate_split(
            split_dir=split_dir,
            x=x,
            train_par=train_par,
            val_par=val_par,
            test_par=test_par,
        )

        self.partition: DataPartition = partition
        match partition:
            case DataPartition.train:
                st = split[0]
            case DataPartition.val:
                st = split[1]
            case DataPartition.test:
                st = split[2]
            case _:
                raise ValueError(f"Invalid partition: {partition}, select from one of {DataPartition.__members__.keys()}")

        #  st is the index of the first frame to load
        st = st[:max_samples]

        z = data["z"]
        print("mol idx:", z)
        # Select all atoms with atomic number 'z' greater than 1
        x = x[:, z > 1, ...]
        v = v[:, z > 1, ...]
        z = z[z > 1]

        x_0, v_0 = x[st], v[st]
        x_t, v_t = x[st + delta_frame], v[st + delta_frame]

        print("Got {:d} samples!".format(x_0.shape[0]))

        mole_idx = z
        n_node = mole_idx.shape[0]
        self.n_node = n_node

        # Build edges
        atom_edges, atom_edges2 = self._compute_edges(x=x, z=z)
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2

        edge_attr, edges = self._build_edge_attributes(atom_edges=atom_edges, atom_edges2=atom_edges2, mole_idx=mole_idx)
        self.edge_attr = edge_attr
        self.edges = edges

        # Build conf_edges
        all_edges = self._compute_all_edges(x=x, z=z)
        conf_edges = self._compute_conf_edges(all_edges=all_edges)
        self.conf_edges: list[list[int]] = conf_edges

        # Convert to tensors
        self.x_0 = torch.cat([torch.Tensor(x_0), torch.norm(torch.Tensor(x_0), dim=-1, keepdim=True)], dim=-1)
        self.v_0 = torch.cat([torch.Tensor(v_0), torch.norm(torch.Tensor(v_0), dim=-1, keepdim=True)], dim=-1)
        self.x_t = torch.Tensor(x_t)
        self.v_t = torch.Tensor(v_t)
        self.mole_idx = torch.Tensor(mole_idx)
        self.Z = torch.Tensor(z)

        self.cfg = self.sample_cfg()

    def _get_or_generate_split(
        self, split_dir: str, x: npt.NDArray[np.float64], train_par: float, val_par: float, test_par: float
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        Load the train/val/test split from split_dir if it exists; otherwise generate it.
        Returns:
            A tuple (train_idx, val_idx, test_idx) of arrays containing split indices.
        """
        try:
            with open(split_dir, "rb") as f:
                print("Got Split!")
                split: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]] = pkl.load(f)
        except Exception as e:
            print(f"Error loading split file: {e}")
            np.random.seed(100)

            _x = x[10000:-10000]

            train_idx = np.random.choice(np.arange(_x.shape[0]), size=int(train_par * _x.shape[0]), replace=False)
            flag = np.zeros(_x.shape[0])
            for _ in train_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            val_idx = np.random.choice(rest, size=int(val_par * _x.shape[0]), replace=False)
            for _ in val_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            test_idx = np.random.choice(rest, size=int(test_par * _x.shape[0]), replace=False)

            train_idx += 10000
            val_idx += 10000
            test_idx += 10000

            split = (train_idx, val_idx, test_idx)

            with open(split_dir, "wb") as f:
                pkl.dump(split, f)
            print("Generate and save split!")

        return split

    def _compute_edges(self, x: npt.NDArray[np.float64], z: npt.NDArray[np.int_], threshold: float = 1.6) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes adjacency (atom_edges) and squared adjacency (atom_edges2) matrices
        based on inter-atomic distances at the first frame.
        """

        def d(_i: int, _j: int, _t: int) -> float:
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = z.shape[0]
        atom_edges = torch.zeros(n, n).int()
        for i in range(n):
            for j in range(n):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < threshold:
                        atom_edges[i][j] = 1

        atom_edges2 = atom_edges @ atom_edges
        return atom_edges, atom_edges2

    def _build_edge_attributes(self, atom_edges: torch.Tensor, atom_edges2: torch.Tensor, mole_idx: npt.NDArray[np.int_]) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Build edge_attr (torch.Tensor) and edges (list) based on atom_edges and atom_edges2.
        The edge_attr array stores [atom_type1, atom_type2, path_distance].
        The edges list has two sublists [rows, cols] for edges.
        """

        n_node = mole_idx.shape[0]
        edge_attr = []
        rows, cols = [], []
        for i in range(n_node):
            for j in range(n_node):
                if i != j:
                    if atom_edges[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 1])
                        assert not atom_edges2[i][j]
                    if atom_edges2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 2])
                        assert not atom_edges[i][j]

        edges = [rows, cols]
        edge_attr_tensor = torch.Tensor(np.array(edge_attr))

        return edge_attr_tensor, edges

    def _compute_all_edges(self, x: npt.NDArray[np.float64], z: npt.NDArray[np.int_], threshold: float = 1.6) -> dict[tuple[int, int], list[list[int]]]:
        """
        Builds a dictionary of edges keyed by (atom type i, atom type j), where each value
        is a list of pairs of atom indices that are close (distance < threshold).
        """

        def d(_i: int, _j: int, _t: int) -> float:
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = z.shape[0]
        all_edges: dict[tuple[int, int], list[list[int]]] = {}
        for i in range(n):
            for j in range(i + 1, n):
                _d = d(i, j, 0)
                if _d < threshold:
                    idx_i, idx_j = z[i], z[j]
                    if idx_i < idx_j:
                        idx_i, idx_j = idx_j, idx_i
                    if (idx_i, idx_j) in all_edges:
                        all_edges[(idx_i, idx_j)].append([i, j])
                    else:
                        all_edges[(idx_i, idx_j)] = [[i, j]]

        return all_edges

    def _compute_conf_edges(self, all_edges: dict[tuple[int, int], list[list[int]]]) -> list[list[int]]:
        """
        Based on all_edges, select and combine edges that preserve bond constraints.
        """
        conf_edges = []
        for key in all_edges:
            # if True:
            assert abs(key[0] - key[1]) <= 2
            conf_edges.extend(all_edges[key])
        return conf_edges

    def sample_cfg(self) -> dict[str, list[tuple[int, int]] | list[list[int]]]:
        """
        Kinematics Decomposition
        """
        cfg = {}
        if self.molecule_type == "benzene_old":
            cfg["Stick"] = [(0, 1), (2, 3), (4, 5)]
        elif self.molecule_type == "aspirin":
            cfg["Stick"] = [(0, 2), (1, 3), (5, 6), (7, 10), (11, 12)]
        elif self.molecule_type == "ethanol":
            cfg["Stick"] = [(0, 1)]
        elif self.molecule_type == "malonaldehyde":
            cfg["Stick"] = [(1, 2)]
        elif self.molecule_type == "naphthalene":
            cfg["Stick"] = [(0, 1), (2, 3), (4, 9), (5, 6), (7, 8)]
        elif self.molecule_type == "salicylic":
            cfg["Stick"] = [(0, 9), (1, 2), (4, 5), (6, 7)]
        elif self.molecule_type == "toluene":
            cfg["Stick"] = [(2, 3), (5, 6), (0, 1)]
        elif self.molecule_type == "uracil":
            cfg["Stick"] = [(0, 1), (3, 4)]
        else:
            raise NotImplementedError()
        cur_selected = []
        for _ in cfg["Stick"]:
            cur_selected.append(_[0])
            cur_selected.append(_[1])
        cfg["Isolated"] = [[_] for _ in range(self.n_node) if _ not in cur_selected]
        if len(cfg["Isolated"]) == 0:
            cfg.pop("Isolated")

        return cfg

    def __getitem__(self, i) -> dict[str, torch.Tensor]:

        cfg = self.cfg

        edge_attr = self.edge_attr
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        edges = self.edges

        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if "Stick" in cfg:
                for stick in cfg["Stick"]:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                        stick_ind[m] = 1
            if "Hinge" in cfg:
                for hinge in cfg["Hinge"]:
                    if (row, col) in [
                        (hinge[0], hinge[1]),
                        (hinge[1], hinge[0]),
                        (hinge[0], hinge[2]),
                        (hinge[2], hinge[0]),
                    ]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)  # [edge, 4]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}
        cfg_tensors = {key: torch.from_numpy(np.array(cfg[key])) for key in cfg}

        return {
            # 'x_0': [n_nodes, 4]: 3D coords (x,y,z, norm(x)) in starting frame.
            "x_0": self.x_0[i],
            # 'v_0': [n_nodes, 4]: 3D velocity (vx,vy,vz, norm(v)) in starting frame.
            "v_0": self.v_0[i],
            # 'edge_attr': [n_edges, 4]: edge attributes.
            # [atom type 1, atom type 2, path distance (1 or 2), stick indicator].
            "edge_attr": edge_attr,
            # 'mole_idx': [n_nodes, 1]: molecule ID for each atom.
            "mole_idx": self.mole_idx.unsqueeze(-1),
            # 'x_t': [n_nodes, 3]: 3D coords at future timestep.
            "x_t": self.x_t[i],
            # 'v_t': [n_nodes, 3]: 3D velocity at future timestep.
            "v_t": self.v_t[i],
            # 'Z': [n_nodes, 1]: atomic number for each atom.
            "Z": self.Z.unsqueeze(-1),
            # 'cfg': special groupings or constraints.
            # 'Stick': [n_sticks, 2]: stick constraint by atom indices.
            # 'Isolated': [n_isolated, 1]: index of isolated atoms.
            "cfg": cfg_tensors,
            # # 'concatenated_features': [n_nodes, 9]: concatenated (x,y,z,norm(x),vx,vy,vz,norm(v),Z)
            "concatenated_features": torch.cat([self.x_0[i], self.v_0[i], self.Z.unsqueeze(-1)], dim=-1),
        }

    def __len__(self):
        return len(self.x_0)

    def get_edges(self, batch_size: int, n_nodes: int) -> list[torch.LongTensor]:
        edges: list[torch.LongTensor] = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges

        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows).long(), torch.cat(cols).long()]
        return edges

    @staticmethod
    def get_cfg(batch_size: int, n_nodes: int, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == "Isolated":
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


@final
class MD17DynamicsDataset(MD17Dataset):
    """
    MD17 Dynamics Dataset

    """

    def __init__(
        self,
        partition: DataPartition,
        max_samples: int,
        delta_frame: int,
        data_dir: str,
        split_dir: str,
        molecule_type: MoleculeType,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        num_timesteps: int = 8,
    ):
        # First call the parent constructor so we inherit shared logic
        super().__init__(
            partition=partition,
            max_samples=max_samples,
            delta_frame=delta_frame,
            data_dir=data_dir,
            split_dir=split_dir,
            molecule_type=molecule_type,
            train_par=train_par,
            val_par=val_par,
            test_par=test_par,
        )

        # setup a split, tentative setting
        train_par, val_par, test_par = train_par, val_par, test_par
        full_dir = os.path.join(data_dir + "md17_" + molecule_type + ".npz")
        split_dir = os.path.join(split_dir + molecule_type + "_split.pkl")
        data: np.lib.npyio.NpzFile = np.load(full_dir)

        self.partition: DataPartition = partition
        self.molecule_type: MoleculeType = molecule_type

        x: npt.NDArray[np.float64] = data["R"]
        v: npt.NDArray[np.float64] = x[1:] - x[:-1]
        x = x[:-1]

        # Attempt to load or generate the split
        try:
            with open(split_dir, "rb") as f:
                print("Got Split!")
                split: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]] = pkl.load(f)
        except Exception as e:
            print(f"Error loading split file: {e}")
            np.random.seed(100)

            _x = x[10000:-10000]
            train_idx: np.ndarray = np.random.choice(np.arange(_x.shape[0]), size=int(train_par * _x.shape[0]), replace=False)
            flag: np.ndarray = np.zeros(_x.shape[0])
            for _ in train_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            val_idx: np.ndarray = np.random.choice(rest, size=int(val_par * _x.shape[0]), replace=False)
            for _ in val_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            test_idx: np.ndarray = np.random.choice(rest, size=int(test_par * _x.shape[0]), replace=False)

            train_idx += 10000
            val_idx += 10000
            test_idx += 10000
            split = (train_idx, val_idx, test_idx)

            with open(split_dir, "wb") as f:
                pkl.dump(split, f)
            print("Generate and save split!")

        match partition:
            case DataPartition.train:
                st = split[0]
            case DataPartition.val:
                st = split[1]
            case DataPartition.test:
                st = split[2]
            case _:
                raise ValueError(f"Invalid partition: {partition}")

        st = st[:max_samples]

        z: npt.NDArray[np.int_] = data["z"]
        print("mol idx:", z)
        # Filter out atoms with atomic number <= 1
        x = x[:, z > 1, ...]
        v = v[:, z > 1, ...]
        z = z[z > 1]

        # Select starting and velocity frames
        x_0: npt.NDArray[np.float64] = x[st]
        v_0: npt.NDArray[np.float64] = v[st]

        # Create multi-step x_t, v_t sequences
        x_t: list[npt.NDArray[np.float64]] = [x[st + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        x_t = np.stack(x_t, axis=2)
        v_t: list[npt.NDArray[np.float64]] = [v[st + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        v_t = np.stack(v_t, axis=2)

        print(f"Got {x_0.shape[0]} samples!")

        mole_idx: npt.NDArray[np.int_] = z
        n_node: int = mole_idx.shape[0]
        self.n_node: int = n_node

        # Convert arrays to torch tensors
        self.x_0 = torch.cat([torch.Tensor(x_0), torch.norm(torch.Tensor(x_0), dim=-1, keepdim=True)], dim=-1)
        self.v_0 = torch.cat([torch.Tensor(v_0), torch.norm(torch.Tensor(v_0), dim=-1, keepdim=True)], dim=-1)
        self.x_t = torch.Tensor(x_t)
        self.v_t = torch.Tensor(v_t)
        self.mole_idx = torch.Tensor(mole_idx)
        self.Z = torch.Tensor(z)

        # Use the parent's helper functions for edge construction and bond constraints
        atom_edges, atom_edges2 = self._compute_edges(x, z, threshold=1.6)
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2

        edge_attr, edges = self._build_edge_attributes(atom_edges, atom_edges2, mole_idx)
        self.edge_attr = edge_attr
        self.edges = edges

        all_edges = self._compute_all_edges(x, z, threshold=1.6)
        conf_edges = self._compute_conf_edges(all_edges)
        self.conf_edges = conf_edges

        # Re-sample configuration from the parent
        self.cfg = self.sample_cfg()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=5000,
        delta_frame=5000,
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=MoleculeType.aspirin,
    )

    dataloader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        for key in data:
            if key not in ["cfg", "edge_attr"]:
                print(f"{key}:", data[key].shape)

        print("cfg shapes:")
        for key in data["cfg"]:
            print(f"  {key}:", data["cfg"][key].shape)
        break
