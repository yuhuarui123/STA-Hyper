import math
from typing import Union, Tuple, Optional

import torch
from torch import Tensor, cat, nn
from torch.nn import init, Parameter, Linear, LayerNorm
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from utils import ccorr


class HypergraphTransformer(MessagePassing):
    r"""Hypergraph Conv containing relation transform、edge fusion(including time fusion)、
    self attention and gated residual connection(or skip connection).

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        attn_heads: int = 4,
        residual_beta: Optional[float] = None,
        learn_beta: bool = False,
        dropout: float = 0.,
        negative_slope: float = 0.2,
        bias: bool = True,
        trans_method: str = 'add',
        edge_fusion_mode: str = 'add',
        time_fusion_mode: str = None,
        head_fusion_mode: str = 'concat',
        residual_fusion_mode: str = None,
        edge_dim: int = None,
        rel_embed_dim: int = None,
        time_embed_dim: int = 0,
        dist_embed_dim: int = 0,
        normalize: bool = True,
        message_mode: str = 'node_edge',
        have_query_feature: bool = False,
        **kwargs
    ):
        super(HypergraphTransformer, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = attn_heads
        self.learn_beta = learn_beta
        self.residual_beta = residual_beta
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.trans_method = trans_method
        self.edge_dim = edge_dim
        self.rel_embed_dim = rel_embed_dim
        self.time_embed_dim = time_embed_dim
        self.dist_embed_dim = dist_embed_dim
        self.edge_fusion_mode = edge_fusion_mode
        self.time_fusion_mode = time_fusion_mode
        self.head_fusion_mode = head_fusion_mode
        self.residual_fusion_mode = residual_fusion_mode
        self.normalize = normalize
        self.message_mode = message_mode
        self.trans_flag = False
        self.have_query_feature = have_query_feature

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            self.in_channels = in_channels
        if in_channels[0] != out_channels and self.trans_method == 'add':
            self.trans_flag = True
            self.lin_trans_x = Linear(in_channels[0], in_channels[1])

        if not self.have_query_feature:
            # V -> hyperedge (E) situation
            self.att_r = Parameter(Tensor(1, attn_heads, out_channels))

        self.attn_in_dim, self.attn_out_dim = self._check_attn_dim(in_channels[1], out_channels)

        self.lin_key = Linear(self.attn_in_dim, attn_heads * out_channels)
        self.lin_query = Linear(in_channels[1], attn_heads * out_channels)
        if self.message_mode == 'node_edge':
            self.lin_value = Linear(self.attn_in_dim, attn_heads * out_channels)
        else:
            self.lin_value = Linear(in_channels[1], attn_heads * out_channels)

        if self.residual_fusion_mode == 'concat':
            self.lin_ffn_0 = Linear(in_channels[1] + self.attn_out_dim, out_channels + 128)
            self.lin_ffn_1 = Linear(out_channels + 128, out_channels)
        elif residual_fusion_mode == 'add':
            if head_fusion_mode == 'concat':
                self.lin_ffn_1 = Linear(attn_heads * out_channels, out_channels, bias=bias)
                self.lin_skip = Linear(in_channels[0], attn_heads * out_channels, bias=bias)
                if learn_beta:
                    self.lin_beta = Linear(3 * attn_heads * out_channels, 1, bias=False)
            else:
                self.lin_skip = Linear(in_channels[0], out_channels, bias=bias)
                if learn_beta:
                    self.lin_beta = Linear(3 * out_channels, 1, bias=False)
        else:
            self.lin_ffn_0 = Linear(self.attn_out_dim, out_channels + 128)
            self.lin_ffn_1 = Linear(out_channels + 128, out_channels)
            if self.head_fusion_mode == 'add':
                self.layer_norm = LayerNorm(out_channels)
            else:
                self.layer_norm = LayerNorm(out_channels * attn_heads)

        self.reset_parameters()

    def reset_parameters(self):
        if self.trans_flag:
            self.lin_trans_x.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if not self.have_query_feature:
            init.xavier_uniform_(self.att_r)
        if self.residual_fusion_mode == 'add':
            self.lin_skip.reset_parameters()
            if self.head_fusion_mode == 'concat':
                self.lin_ffn_1.reset_parameters()
            if self.learn_beta:
                self.lin_beta.reset_parameters()
        else:
            self.lin_ffn_0.reset_parameters()
            self.lin_ffn_1.reset_parameters()
            if not self.residual_fusion_mode:
                self.layer_norm.reset_parameters()

    # the edge_type are stored as edge_index value
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        edge_attr_embed: Tensor,
    ):

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if isinstance(edge_index, SparseTensor):
            out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
                edge_attr_embed=edge_attr_embed,
                edge_time_embed=edge_time_embed,
                edge_dist_embed=edge_dist_embed,
                edge_type_embed=edge_type_embed,
                have_query_feature=self.have_query_feature,
                size=None
            )
        else:
            out = self.propagate(
                edge_index,
                x=(x[0][edge_index[0]], x[1][edge_index[1]]),
                edge_attr_embed=edge_attr_embed,
                edge_time_embed=edge_time_embed,
                edge_dist_embed=edge_dist_embed,
                edge_type_embed=edge_type_embed,
                have_query_feature=self.have_query_feature,
                size=None
            )

        if not self.have_query_feature:
            out += self.att_r

        if self.head_fusion_mode == 'concat':
            out = out.view(-1, self.attn_heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # todo dont use two FC before out
        if self.residual_fusion_mode == 'concat':
            out = cat([out, x[1]], dim=-1)
            out = self.lin_ffn_0(out)
            out = F.relu(out)
            out = self.lin_ffn_1(out)
        elif self.residual_fusion_mode == 'add':
            x_skip = self.lin_skip(x[1])

            if self.learn_beta:
                beta = self.lin_beta(cat([out, x_skip, out - x_skip], -1))
                beta = beta.sigmoid()
                out = beta * x_skip + (1 - beta) * out
            else:
                if self.residual_beta is not None:
                    out = self.residual_beta * x_skip + (1 - self.residual_beta) * out
                else:
                    out += x_skip
            if self.head_fusion_mode == 'concat':
                out = self.lin_ffn_1(out)
        else:
            out = self.layer_norm(out)
            out = self.lin_ffn_0(out)
            out = F.relu(out)
            out = self.lin_ffn_1(out)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(
        self,
        x: OptPairTensor,
        edge_attr_embed: Tensor,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        index: Tensor,
        ptr: OptTensor,
        have_query_feature: bool,
        size_i: Optional[int]
    ) -> Tensor:
        x_j, x_i = x

        if self.trans_flag:
            if have_query_feature:
                x_i = self.lin_trans_x(x_i)
            x_j_raw = self.lin_trans_x(x_j)
            x_j = self.lin_trans_x(x_j)
        else:
            x_j_raw = x_j
        if edge_type_embed is not None:
            x_j = self.rel_transform(x_j, edge_type_embed)

        if self.time_fusion_mode == 'concat':
            x_j = cat([x_j, edge_time_embed, edge_dist_embed], dim=-1)
        elif self.time_fusion_mode == 'add':
            x_j += edge_time_embed + edge_dist_embed

        if edge_attr_embed is not None:
            if self.edge_fusion_mode == 'concat':
                x_j = cat([x_j, edge_attr_embed], dim=-1)
            else:
                x_j += edge_attr_embed

        key = self.lin_key(x_j).view(-1, self.attn_heads, self.out_channels)
        if not have_query_feature:
            query = self.att_r
            alpha = (key * query).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
        else:
            query = self.lin_query(x_i).view(-1, self.attn_heads, self.out_channels)
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.message_mode == 'node_edge':
            out = self.lin_value(x_j).view(-1, self.attn_heads, self.out_channels)
        else:
            out = self.lin_value(x_j_raw).view(-1, self.attn_heads, self.out_channels)

        out *= alpha.view(-1, self.attn_heads, 1)

        return out

    def rel_transform(self, ent_embed, edge_type_embed):
        if self.trans_method == "corr":
            trans_embed = ccorr(ent_embed, edge_type_embed)
        elif self.trans_method == "sub":
            trans_embed = ent_embed - edge_type_embed
        elif self.trans_method == "multi":
            trans_embed = ent_embed * edge_type_embed
        elif self.trans_method == 'add':
            trans_embed = ent_embed + edge_type_embed
        elif self.trans_method == 'concat':
            trans_embed = cat([ent_embed, edge_type_embed], dim=1)
        else:
            raise NotImplementedError
        return trans_embed

    def _check_attn_dim(self, in_channels, out_channels):
        attn_in_dim = in_channels
        attn_out_dim = out_channels * self.attn_heads if self.head_fusion_mode == 'concat' else out_channels
        if self.trans_method == 'concat':
            attn_in_dim += self.rel_embed_dim
        else:
            assert attn_in_dim == self.rel_embed_dim, \
                "[HypergraphTransformer >> Translation Error] Node embedding dimension {} is not equal with relation " \
                "embedding dimension {} when you are using '{}' translation method" \
                ".".format(attn_in_dim, self.rel_embed_dim, self.trans_method)

        if self.time_fusion_mode:
            if self.time_fusion_mode == 'concat':
                attn_in_dim += self.time_embed_dim + self.dist_embed_dim
            else:
                assert attn_in_dim == self.time_embed_dim, \
                    "[HypergraphTransformer >> Time Fusion Error] Time embedding dimension {} is " \
                    "not equal with edge fusion result embedding dimension {} when you are using '{}' " \
                    "time fusion mode.".format(self.time_embed_dim, attn_in_dim, self.time_fusion_mode)
                assert attn_in_dim == self.dist_embed_dim, \
                    "[HypergraphTransformer >> Time Fusion Error] Time embedding dimension {} is " \
                    "not equal with edge fusion result embedding dimension {} when you are using '{}' " \
                    "time fusion mode.".format(self.dist_embed_dim, attn_in_dim, self.time_fusion_mode)

        if self.edge_fusion_mode == 'concat' and self.edge_dim is not None:
            attn_in_dim += self.edge_dim
        else:
            if self.edge_dim is not None:
                assert attn_in_dim == self.edge_dim, \
                    "[HypergraphTransformer >> Edge Fusion Error] Edge embedding dimension {} is " \
                    "not equal with translation result embedding dimension {} when you are using '{}' " \
                    "edge fusion mode.".format(self.edge_dim, attn_in_dim, self.edge_fusion_mode)
        return attn_in_dim, attn_out_dim

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, attn_heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.attn_heads)


class TransformerModel(nn.Module):
    def __init__(self, num, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder = nn.Linear(embed_size, num)
        self.init_weights()
        self.softmax = nn.Softmax()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)

        src = self.pos_encoder(src)
        Poi_feature = self.transformer_encoder(src, src_mask)

        return Poi_feature


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)