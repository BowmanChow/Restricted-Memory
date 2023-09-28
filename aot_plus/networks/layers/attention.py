import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import DropOutLogit


# Long-term attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head=8, dropout=0., use_linear=True):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.hidden_dim = d_model // num_head
        self.T = (d_model / num_head)**0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V, is_return_attn_weight=False):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        if is_return_attn_weight:
            # Scale
            Q = Q / self.T

            # Multi-head
            Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 3, 0)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

            # Multiplication
            QK = Q @ K

            # Activation
            attn = torch.softmax(QK, dim=-1)

            # Dropouts
            attn = self.dropout(attn)

            # Weighted sum
            outputs = (attn @ V).permute(2, 0, 1, 3)
        else:
            dropout_p = self.drop_prob if self.training else 0.0
            Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            enable_mem_efficient = False if self.training else True
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=enable_mem_efficient):
                outputs = F.scaled_dot_product_attention(Q, K, V, None, dropout_p, is_causal=False)
            outputs = outputs.permute(2, 0, 1, 3)
            attn = None

        # Restore shape
        outputs = outputs.reshape(-1, bs, self.d_model)

        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
