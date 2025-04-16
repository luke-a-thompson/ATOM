import torch
import torch.nn as nn
import torch.nn.functional as F
from gtno_py.gtno.mlps import MLP
from typing import override, final
from enum import StrEnum
from gtno_py.training.config_options import FFNActivation
from gtno_py.gtno.activations import get_activation


class AggregationMode(StrEnum):
    SUM = "sum"
    MEAN = "mean"


def aggregate(
    message: torch.Tensor,
    row_index: torch.Tensor,
    n_node: int,
    aggregation_mode: AggregationMode,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    The aggregation function (aggregate edge messages towards nodes)
    :param message: The edge message with shape [M, K]
    :param row_index: The row index of edges with shape [M]
    :param n_node: The number of nodes, N
    :param aggr: aggregation type, sum or mean
    :param mask: the edge mask (used in mean aggregation for counting degree)
    :return: The aggreagated node-wise information with shape [N, K]
    """
    result_shape = (n_node, message.shape[1])
    result: torch.Tensor = message.new_full(result_shape, 0)  # [N, K]
    row_index = row_index.unsqueeze(-1).expand(-1, message.shape[1])  # [M, 1]
    result.scatter_add_(0, row_index, message)  # [N, K]
    match aggregation_mode:
        case AggregationMode.SUM:
            return result
        case AggregationMode.MEAN:
            count = message.new_full(result_shape, 0)  # [N, K]
            ones = torch.ones_like(message)
            if mask is not None:
                ones = ones * mask.unsqueeze(-1)
            count.scatter_add_(0, row_index, ones)
            result = result / count.clamp(min=1)

            return result


class TimeConvMode(StrEnum):
    TIME_CONV = "time_conv"
    TIME_CONV_X = "time_conv_x"


@final
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, fourier_modes: int, num_timesteps: int, conv_mode: TimeConvMode):
        """
        A spectral convolution layer that applies a spectral convolution to the time dimension.

        EGNO equivalences:
            When fixed_scale is True this is equivalent to SpectralConv1d_x
            When fixed_scale is False this is equivalent to SpectralConv1d
        """
        super().__init__()
        self.modes = fourier_modes
        self.num_timesteps = num_timesteps
        self.conv_mode = conv_mode

        if conv_mode == TimeConvMode.TIME_CONV_X:
            self.scale = 0.1
        else:
            self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, fourier_modes, 2, dtype=torch.float))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=0)  # Shape matches
        if self.conv_mode == TimeConvMode.TIME_CONV_X:
            out_ft = torch.einsum("mndi,iom->mndo", x_ft[: self.modes], torch.view_as_complex(self.weights))
        else:
            out_ft = torch.einsum("mni,iom->mno", x_ft[: self.modes], torch.view_as_complex(self.weights))  # Shape matches
        x = torch.fft.irfftn(out_ft, s=self.num_timesteps, dim=0)  # Shape matches
        return x


@final
class TimeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, mode: TimeConvMode, num_timesteps: int):
        """
        A temporal convolution layer that applies a spectral convolution to the time dimension.

        EGNO equivalences:
            When mode is TIME_CONV this is equivalent to TimeConv in EGNO - Activation and SpectralConv scale is 1 / (in_ch * out_ch)
            When mode is TIME_CONV_X this is equivalent to TimeConv_x in EGNO - Activation and SpectralConv scale is 0.1
        """
        super().__init__()
        if mode == TimeConvMode.TIME_CONV:
            self.time_conv = SpectralConv1d(in_channels, out_channels, modes, num_timesteps, mode)
            self.activation = get_activation(FFNActivation.LEAKY_RELU, out_channels)
        else:
            self.time_conv = SpectralConv1d(in_channels, out_channels, modes, num_timesteps, mode)
            self.activation = None

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convolved_x: torch.Tensor = self.time_conv(x)
        if self.activation is not None:
            convolved_x = self.activation(convolved_x)
        return x + convolved_x


# Assume n_vector_input = 1
@final
class InvariantScalarNet(nn.Module):
    def __init__(
        self,
        n_vector_input: int,
        n_scalar_input: int,
        hidden_dim: int,
        output_dim: int,
        activation: nn.Module,
        with_v: bool,
        flat: bool,
        norm: bool,
    ):
        super().__init__()
        self.norm = norm
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.n_scalar_input = n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scalar_mlp = MLP(
            self.input_dim,
            hidden_dim,
            output_dim,
            2,
            activation,
            0.0,
        )

    @override
    def forward(self, vectors: torch.Tensor | list[torch.Tensor], scalars: torch.Tensor) -> torch.Tensor:
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor with shape [N, 3]
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A scalar that is invariant to the O(n) transformations of input vectors  with shape [N, K]

        Shapes:
            N: Batch size
            3: Vector dimension
            K: Number of vector features
            L: Number of scalar features
        """
        if isinstance(vectors, list):
            Z = torch.stack(vectors, dim=-1)
        else:
            Z = vectors
        K: int = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        # This is a pairwise inner product of the vectors, producing a matrix of scalar invariants
        scalar = torch.einsum("bij,bjk->bik", Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]

        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat([scalar, scalars], dim=-1)  # [N, KK + L]

        scalar: torch.Tensor = self.scalar_mlp(scalar)

        return scalar


@final
class EGNN_Layer(nn.Module):
    def __init__(
        self,
        lifting_dim: int,
        activation: nn.Module,
        with_v: bool,
        flat: bool,
        norm: bool,
        h_update: bool,
    ):
        super().__init__()

        self.h_update = h_update

        self.edge_message_net = InvariantScalarNet(
            1,
            2 * lifting_dim + 5,  # 5 edge features
            lifting_dim,
            lifting_dim,
            activation,
            with_v,
            flat,
            norm,
        )
        self.coord_net = MLP(
            lifting_dim,
            lifting_dim,
            1,
            2,
            activation,
            0.0,
        )
        if with_v:
            self.node_v_net = MLP(
                lifting_dim,
                lifting_dim,
                1,
                2,
                activation,
                0.0,
            )
        if h_update:
            self.node_net = MLP(
                lifting_dim + lifting_dim,
                lifting_dim,
                lifting_dim,
                2,
                activation,
                0.0,
            )

    @override
    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, v: torch.Tensor | None = None):
        """ """
        row, col = edge_index[0].to(torch.long), edge_index[1].to(torch.long)
        rij = x[row] - x[col]  # Shape [B*T*E, 3] matches
        hij = torch.cat((h[row], h[col], edge_attr), dim=-1)  # Shape [B*T*E, 2K+T] matches; 1 missing from attr due to no stick indicies

        message: torch.Tensor = self.edge_message_net(vectors=[rij], scalars=hij)  # Shape [BM, 3] matches
        coord_message: torch.Tensor = self.coord_net(message)  # Shape [BM, 1] matches
        f: torch.Tensor = (x[row] - x[col]) * coord_message  # Shape [BM, 3] matches

        tot_f = aggregate(f, row, x.shape[0], AggregationMode.MEAN, None)  # Shape [B*N*T, 3] matches
        tot_f = torch.clamp(tot_f, min=-100, max=100)

        if v is not None:
            x = x + self.node_v_net(h) * v + tot_f
        else:
            x = x + tot_f  # [BN, 3]

        tot_message = aggregate(message, row, x.shape[0], AggregationMode.SUM, None)  # [BN, K]
        node_message = torch.cat((h, tot_message), dim=-1)  # Shape [BN, K+K] matches
        if self.h_update:
            h = h + self.node_net(node_message)  # [BN, K], corrected EGNN residual
        return x, v, h


class EGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        lifting_dim: int,
        num_layers: int,
        activation: nn.Module,
        with_v: bool,
        flat: bool,
        norm: bool,
    ):
        super().__init__()

        self.embedding = nn.Linear(in_dim, lifting_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EGNN_Layer(lifting_dim, activation, with_v=with_v, flat=flat, norm=norm, h_update=True)
            _ = self.layers.append(layer)

    @override
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        v: torch.Tensor | None = None,
        loc_mean: torch.Tensor | None = None,
    ):
        h = self.embedding(h)
        for i in range(len(self.layers)):
            x, v, h = self.layers[i](x, h, edge_index, edge_attr, v, loc_mean)

        if v is not None:
            return x, v, h
        else:
            return x, h
