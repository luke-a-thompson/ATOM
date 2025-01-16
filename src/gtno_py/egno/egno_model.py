from gtno_py.egno.basic import EGNN
from gtno_py.egno.layer_no import TimeConv, get_timestep_embedding, TimeConv_x
import torch.nn as nn
import torch
from typing import final, override, Literal
from tensordict import TensorDict
from gtno_py.dataloaders.egno_dataloder import MD17DynamicsDataset


@final
class EGNO(EGNN):
    def __init__(
        self,
        n_layers: int,
        in_node_num_feats: int,
        in_edge_num_feats: int,
        hidden_num_feats: int,
        activation: nn.Module = nn.SiLU(),
        device: Literal["cpu", "cuda"] = "cpu",
        with_v: bool = False,
        flat: bool = False,
        norm: bool = False,
        use_time_conv: bool = True,
        num_modes: int = 2,
        num_timesteps: int = 8,
        time_emb_dim: int = 32,
    ):
        self.time_emb_dim = time_emb_dim
        in_node_num_feats = in_node_num_feats + self.time_emb_dim

        super(EGNO, self).__init__(n_layers, in_node_num_feats, in_edge_num_feats, hidden_num_feats, activation, device, with_v, flat, norm)
        self.use_time_conv = use_time_conv
        self.num_timesteps = num_timesteps
        self.device = device
        self.hidden_nf = hidden_num_feats

        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()
            self.time_conv_x_modules = nn.ModuleList()
            for _ in range(n_layers):
                self.time_conv_modules.append(TimeConv(hidden_num_feats, hidden_num_feats, num_modes, activation, with_nin=False))
                self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

        self.to(self.device)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, edge_index: torch.LongTensor, edge_fea: torch.Tensor, v: torch.Tensor | None = None, loc_mean: torch.Tensor | None = None
    ):  # [BN, H]

        assert max(len(x.shape), len(h.shape)) == 2, f"All tensors must be 2D (B*N, feat). Got: x.shape: {x.shape}, h.shape: {h.shape}"

        T = self.num_timesteps

        num_nodes = h.shape[0]
        num_edges = edge_index[0].shape[0]

        cumsum: torch.Tensor = torch.arange(0, T).to(self.device) * num_nodes
        # cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)

        time_emb = get_timestep_embedding(torch.arange(T).to(x), embedding_dim=self.time_emb_dim, max_positions=10000)  # [T, H_t]
        h = h.unsqueeze(0).repeat(T, 1, 1)  # [T, BN, H]
        time_emb = time_emb.unsqueeze(1).repeat(1, num_nodes, 1)  # [T, BN, H_t]
        h = torch.cat((h, time_emb), dim=-1)  # [T, BN, H+H_t]
        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        h = self.embedding(h)
        x = x.repeat(T, 1)
        loc_mean = loc_mean.to(self.device).repeat(T, 1)
        edges_0 = edge_index[0].to(self.device).repeat(T) + cumsum_edges
        edges_1 = edge_index[1].to(self.device).repeat(T) + cumsum_edges
        edge_index = [edges_0, edges_1]
        v = v.to(self.device).repeat(T, 1)

        edge_fea = edge_fea.to(self.device).repeat(T, 1)

        for i in range(self.n_layers):
            if self.use_time_conv:
                time_conv = self.time_conv_modules[i]
                h = time_conv(h.view(T, num_nodes, self.hidden_nf)).view(T * num_nodes, self.hidden_nf)
                x_translated = x - loc_mean
                time_conv_x = self.time_conv_x_modules[i]
                # v shape: [timesteps * num_nodes, 3 (x, y, z) + 1 (norm)]
                # x_translated shape: [timesteps * num_nodes, 3 (x, y, z) + 1 (norm)]
                X = torch.stack((x_translated, v), dim=-1)  # Combine x and v into a single tensor, shape [timesteps * num_nodes, 4, 2] (pos + vel for each coord)
                temp: torch.Tensor = time_conv_x(X.view(T, num_nodes, 4, 2))  # X.view(timesteps * num_nodes, 4, 2) -> [T, N, 4, 2], separating out the time dimension
                x = temp[..., 0].view(T * num_nodes, 4) + loc_mean
                v = temp[..., 1].view(T * num_nodes, 4)

            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)

        # assert False, f"x: {x.shape}, v: {v.shape}, h: {h.shape}"

        if v is not None:
            return x, v, h
        else:
            return x, h

    @staticmethod
    def reshape_batch(batch: TensorDict, dataset: MD17DynamicsDataset):
        batch_size = int(batch.batch_size[0])
        n_nodes = int(batch["x_0"].shape[1])

        # The velocity norm is already the last element of the velocity vector. We concat it with the normalised atomic numbers.
        z_normalised: torch.Tensor = batch["Z"] / batch["Z"].max().unsqueeze(-1)
        nodes: torch.Tensor = torch.cat([batch["v_0"], z_normalised], dim=-1)  # Shape [B, N, 5], 5 = 3 (x, y, z) + 1 (velocity norm) + 1 (normalised atomic number)
        nodes = nodes.view(-1, nodes.shape[-1])  # [B*N, 5]

        edges = dataset.get_edges(batch_size, n_nodes)

        rows, cols = edges[0], edges[1]
        loc_dist = torch.sum((batch["x_0"].view(-1, batch["x_0"].shape[-1])[rows] - batch["x_0"].view(-1, batch["x_0"].shape[-1])[cols]) ** 2, dim=-1).unsqueeze(
            -1
        )  # Shape [B*E, 1]
        # loc_dist: torch.Tensor = torch.sum((batch["x_0"][:, rows] - batch["x_0"][:, cols]) ** 2, dim=-1).unsqueeze(-1)  # Shape [batch, n_edges, 1]
        edge_attr = batch["edge_attr"].view(-1, batch["edge_attr"].shape[-1])  # [B*E, edge_feats]
        assert loc_dist.shape[0] == edge_attr.shape[0], f"Loc dist must have the same number of edges {loc_dist.shape[0]} as edge attributes {batch['edge_attr'].shape[0]}"
        edge_attr: torch.Tensor = torch.cat([edge_attr, loc_dist], dim=1).detach()  # Shape [B*E, edge_feats + 1]

        loc_mean: torch.Tensor = batch["x_0"].mean(dim=1, keepdim=True).repeat(1, n_nodes, 1)  # shape [B, 1, 4]  # shape [B, N, 4]
        loc_mean = loc_mean.view(-1, loc_mean.shape[-1])  # [B*N, 4]

        x_0: torch.Tensor = batch["x_0"].view(-1, batch["x_0"].shape[-1])
        v_0: torch.Tensor = batch["v_0"].view(-1, batch["v_0"].shape[-1])

        return x_0, nodes, edges, edge_attr, v_0, loc_mean
