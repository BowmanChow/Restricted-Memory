from typing import Iterable, List
import torch.nn.functional as F
from torch import nn
import torch

from networks.layers.basic import DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d
from networks.layers.attention import MultiheadAttention, GatedPropagation, LocalGatedPropagation, silu
from utils.tensor import lbc_2_bchw, bchw_2_lbc


def _get_norm(indim, type='ln', groups=8):
    if type == 'gn' and groups != 1:
        return GroupNorm1D(indim, groups)
    elif type == 'gn' and groups == 1:
        return nn.GroupNorm(groups, indim)
    else:
        return nn.LayerNorm(indim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_can = nn.Conv2d(
            in_channels=input_dim+hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def init_hidden(batch_size, hidden_dim, input_size, dtype):
        height, width = input_size
        return (torch.autograd.Variable(torch.zeros(batch_size, hidden_dim, height, width)).type(dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRUCellOutput(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, output_dim) -> None:
        super(ConvGRUCellOutput, self).__init__()
        self.conv_gru_cell = ConvGRUCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.output_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=1,
        )
    def forward(self, input_tensor, h_cur):
        h_next = self.conv_gru_cell(input_tensor, h_cur)
        return h_next, self.output_conv(h_next)


class LongShortTermTransformer(nn.Module):
    def __init__(
        self,
        num_layers=2,
        d_model=256,
        self_nhead=8,
        att_nhead=8,
        dim_feedforward=1024,
        emb_dropout=0.,
        droppath=0.1,
        lt_dropout=0.,
        st_dropout=0.,
        droppath_lst=False,
        droppath_scaling=False,
        activation="gelu",
        return_intermediate=False,
        intermediate_norm=True,
        final_norm=True,
        linear_q=False,
        norm_inp=False,
        time_encode=False,
        gru_memory=False,
    ):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)
        self.gru_memory = gru_memory

        layers: List[SimplifiedTransformerBlock] = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                SimplifiedTransformerBlock(
                    d_model, self_nhead, att_nhead,
                    dim_feedforward, droppath_rate,
                    activation,
                    linear_q=linear_q,
                    time_encode=time_encode,
                    gru_memory=gru_memory,
                ))
        self.layers: Iterable[SimplifiedTransformerBlock] = nn.ModuleList(
            layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model, type='ln') for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

        self.clear_memory()

    def forward(
        self,
        tgt,
        curr_id_emb=None,
        self_pos=None,
        size_2d=None,
        temporal_encoding=None,
        is_outer_memory=False,
        outer_long_memories=None,
        outer_short_memories=None,
        save_atten_weights=False,
    ):

        output = self.emb_dropout(tgt)

        intermediate = []
        intermediate_memories = []

        for idx, layer in enumerate(self.layers):
            if is_outer_memory:
                output, memories = layer(
                    output,
                    outer_long_memories[idx],
                    outer_short_memories[idx],
                    curr_id_emb=curr_id_emb,
                    self_pos=self_pos,
                    size_2d=size_2d,
                    temporal_encoding=temporal_encoding,
                    save_atten_weights=save_atten_weights,
                )
            else:
                output, memories = layer(
                    output,
                    self.long_term_memories[idx] if
                    self.long_term_memories is not None else None,
                    self.short_term_memories[idx] if
                    self.short_term_memories is not None else None,
                    curr_id_emb=curr_id_emb,
                    self_pos=self_pos,
                    size_2d=size_2d,
                    temporal_encoding=temporal_encoding,
                    save_atten_weights=save_atten_weights,
                )
            # memories : [[curr_K, curr_V], [global_K, global_V], [local_K, local_V]]

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            if not is_outer_memory:
                self.lstt_curr_memories, self.lstt_long_memories, self.lstt_short_memories = zip(
                    *intermediate_memories)
            return intermediate

        return output

    def update_short_memories(
        self,
        curr_id_emb,
        short_term_mem_skip,
        is_update_long_memory,
    ):
        lstt_curr_memories_2d = []
        for layer_idx in range(len(self.lstt_curr_memories)):
            curr_v = self.lstt_curr_memories[layer_idx][1]
            curr_v = self.layers[layer_idx].linear_V(
                curr_v + curr_id_emb)
            self.lstt_curr_memories[layer_idx][1] = curr_v

            curr_v = self.lstt_short_memories[layer_idx][1]
            curr_v = self.layers[layer_idx].linear_VMem(
                curr_v + curr_id_emb)
            self.lstt_short_memories[layer_idx][1] = curr_v

            lstt_curr_memories_2d.append([
                self.lstt_short_memories[layer_idx][0],
                self.lstt_short_memories[layer_idx][1],
            ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        for temp in self.short_term_memories_list[0]:
            for x in temp:
                x.cpu()
        self.short_term_memories_list = self.short_term_memories_list[
            -short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if is_update_long_memory:
            self.update_long_term_memory(
                self.lstt_curr_memories,
            )

    def update_long_term_memory(
        self,
        new_long_term_memories,
    ):
        updated_long_term_memories = []
        max_size = 48840
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(
                new_long_term_memory,
                last_long_term_memory,
            ):
                new_mem = torch.cat([last_e, new_e[None, ...]], dim=0)
                updated_e.append(new_mem)
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def restrict_long_memories(
        self,
        former_memory_len,
        latter_memory_len,
        use_atten_weight=False,
    ):
        to_drop_idx = former_memory_len
        if self.gru_memory:
            to_drop_idx += 1
        if use_atten_weight:
            attn_weight = torch.stack([
                self.layers[0].record_attn_weight,
                self.layers[0].record_attn_weight,
                self.layers[1].record_attn_weight,
                self.layers[2].record_attn_weight,
            ]).mean(dim=0)
            attn_weight = attn_weight.cpu()
            # print(f"{attn_weight = }")
            ignore_former_size = 1
            if self.gru_memory:
                ignore_former_size += 1
            attn_weight_remove_0 = attn_weight[ignore_former_size:]
            if attn_weight_remove_0.size(0) > 0:
                to_drop_idx = torch.argmin(attn_weight_remove_0).item()
                to_drop_idx += ignore_former_size
        # print(f"{to_drop_idx = }")
        for layer_idx in range(len(self.layers)):
            memory_k_v = self.long_term_memories[layer_idx]
            for i in range(len(memory_k_v)):
                mem = memory_k_v[i]
                if mem.size(0) > (former_memory_len + latter_memory_len):
                    if self.gru_memory:
                        gru = self.layers[layer_idx].memory_grus[i]
                        # gru = mean_gru
                        hidden_state = self.long_term_memory_hidden_states[layer_idx][i]
                        size_2d = self.long_term_memory_hidden_states[0][0].size()[2:]
                        gru_input = lbc_2_bchw(mem[to_drop_idx, ...], size_2d)
                        hidden_state, gru_output = gru(gru_input, hidden_state)
                        gru_output = bchw_2_lbc(gru_output)
                        new_mem = torch.cat(
                            [mem[0:1, ...], gru_output[None, ...], mem[2:to_drop_idx, ...], mem[to_drop_idx+1:, ...]], dim=0)
                        self.long_term_memory_hidden_states[layer_idx][i] = hidden_state
                    else:
                        new_mem = torch.cat(
                            [mem[0:to_drop_idx, ...], mem[to_drop_idx+1:, ...]], dim=0)
                    self.long_term_memories[layer_idx][i] = new_mem

    def init_memory(self, size_2d=(30, 30)):
        self.long_term_memories = self.lstt_long_memories
        self.short_term_memories_list = [self.lstt_short_memories]
        self.short_term_memories = self.lstt_short_memories
        if self.gru_memory:
            l, b, c = self.short_term_memories[0][0].size()
            dtype = self.short_term_memories[0][0].dtype
            device = self.short_term_memories[0][0].device
            self.long_term_memory_hidden_states = [
                [
                    ConvGRUCell.init_hidden(b, c, size_2d, dtype).to(device),
                    ConvGRUCell.init_hidden(b, c, size_2d, dtype).to(device),
                ] for _ in range(len(self.layers))
            ]

    def clear_memory(self):
        self.lstt_curr_memories = None
        self.lstt_long_memories = None
        self.lstt_short_memories = None

        self.short_term_memories_list = []
        self.short_term_memories = None
        self.long_term_memories = None
        self.long_term_memory_hidden_states = None


class SimplifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        self_nhead,
        att_nhead,
        dim_feedforward=1024,
        droppath=0.1,
        activation="gelu",
        linear_q=False,
        time_encode=False,
        gru_memory=False,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_QMem = nn.Linear(d_model, d_model)
        self.linear_VMem = nn.Linear(d_model, d_model)
        if not linear_q:
            self.norm4 = _get_norm(d_model)

        self.linear_KMem = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(
            d_model,
            att_nhead,
            use_linear=False,
        )

        self.short_term_attn = MultiheadAttention(
            d_model,
            att_nhead,
            use_linear=False,
        )

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self.linear_q = linear_q
        self._init_weight()

        if time_encode:
            self.Q_time_encode = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model),
                nn.ReLU(),
                nn.Linear(in_features=d_model, out_features=d_model),
            )
            self.K_time_encode = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model),
                nn.ReLU(),
                nn.Linear(in_features=d_model, out_features=d_model),
            )
        if gru_memory:
            self.memory_grus = nn.ModuleList([
                ConvGRUCellOutput(
                    input_dim=d_model,
                    hidden_dim=d_model,
                    kernel_size=(2, 2),
                    bias=True,
                    output_dim=d_model,
                ),
                ConvGRUCellOutput(
                    input_dim=d_model,
                    hidden_dim=d_model,
                    kernel_size=(1, 1),
                    bias=True,
                    output_dim=d_model,
                ),
            ])
    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        long_term_memory=None,
        short_term_memory=None,
        curr_id_emb=None,
        self_pos=None,
        size_2d=(30, 30),
        temporal_encoding=None,
        save_atten_weights=False,
    ):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = curr_Q

        if curr_id_emb is not None:
            global_K = curr_K
            global_V = self.linear_V(curr_V + curr_id_emb)
            local_K = global_K
            local_V = global_V
            global_K = global_K[None, ...]
            global_V = global_V[None, ...]
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        if temporal_encoding is None:
            flatten_global_K = global_K.flatten(0, 1)
            flatten_global_V = global_V.flatten(0, 1)
            curr_Q_add_time = curr_Q
        else:
            Q_temp_encoding = self.Q_time_encode(temporal_encoding[-1, ...])
            K_temp_encoding = self.K_time_encode(temporal_encoding[:-1, ...])
            curr_Q_add_time = curr_Q + Q_temp_encoding
            flatten_global_K = (global_K + K_temp_encoding).flatten(0, 1)
            flatten_global_V = (global_V).flatten(0, 1)

        tgt2, attn = self.long_term_attn(
            curr_Q_add_time, flatten_global_K, flatten_global_V,
            is_return_attn_weight=save_atten_weights,
        )
        if save_atten_weights:
            bs, head, hw, thw = attn.size()
            self.record_T = thw // hw
            attn = attn.view((bs, head, hw, self.record_T, hw))
            attn = attn.mean(dim=1) # bs, hw, T, hw
            assert attn.size(0) == 1 # only for evaluation
            attn = attn.squeeze(0) # hw, T, hw
            self.record_attn_weight = attn.sum(dim=2)
            self.attn_values, self.attn_indices = \
                attn.detach().view((hw, thw)).topk(32, dim=-1)
            self.attn_values, self.attn_indices = \
                self.attn_values.cpu().numpy(), self.attn_indices.cpu().numpy()
            self.attn_indices = np.unravel_index(self.attn_indices, (self.record_T, hw))

        if self.linear_q:
            tgt3 = self.short_term_attn(
                local_Q,
                torch.cat((local_K, curr_K), 0),
                torch.cat((local_V, curr_V), 0),
            )[0]
        else:
            tgt3, short_attn = self.short_term_attn(
                local_Q,
                self.norm4(local_K + curr_K),
                self.norm4(local_V + curr_V),
                is_return_attn_weight=save_atten_weights,
            )
        if save_atten_weights:
            # bs, head, hw, hw
            short_attn = short_attn.mean(dim=1) # bs, hw, hw
            assert short_attn.size(0) == 1 # only for evaluation
            short_attn = short_attn.squeeze(0) # hw, hw
            self.short_attn_values, self.short_attn_indices = \
                short_attn.detach().topk(32, dim=-1)
            self.short_attn_values, self.short_attn_indices = \
                self.short_attn_values.cpu().numpy(), self.short_attn_indices.cpu().numpy()

        _tgt3 = tgt3

        local_K = self.linear_QMem(_tgt3)
        local_V = _tgt3
        if curr_id_emb is not None:
            local_V = self.linear_VMem(local_V + curr_id_emb)

        tgt = tgt + tgt2 + tgt3

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [
            [curr_K, curr_V], [global_K, global_V],
            [local_K, local_V],
        ]

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
