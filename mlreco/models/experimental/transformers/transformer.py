import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class TransformerDecoder(nn.Module):
    
    def __init__(self, d_model, num_heads, 
                 dim_feedforward=1024, dropout=0.0, normalize_before=False):
        super(TransformerDecoder, self).__init__()
        
        self.num_heads = num_heads
        
        self.cross_attention = CrossAttentionLayer(d_model,
                                                   num_heads,
                                                   dropout=dropout,
                                                   normalize_before=normalize_before)
        self.self_attention  = SelfAttentionLayer(d_model, 
                                                  num_heads, 
                                                  dropout=dropout, 
                                                  normalize_before=normalize_before)
        self.ffn_layer       = FFNLayer(d_model, 
                                        dim_feedforward,
                                        dropout=dropout,
                                        normalize_before=normalize_before)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, query_pos, src_pcd, batched_pos_enc, batched_attn):
        """
        queries: B, num_queries, d_model
        
        """
        x = queries.permute((1,0,2))
        memory_mask = batched_attn.repeat_interleave(
            self.num_heads, dim=0).permute(0, 2, 1)
        pos = batched_pos_enc.permute((1,0,2))
        x = self.cross_attention(x, 
                                 src_pcd.permute((1,0,2)), 
                                 memory_mask=memory_mask,
                                 memory_key_padding_mask=None,
                                 pos=pos,
                                 query_pos=query_pos.permute((1,0,2)))
        x = self.self_attention(x, tgt_mask=None, tgt_key_padding_mask=None,
                                query_pos=query_pos.permute((1,0,2)))
        out_queries = self.ffn_layer(x).permute((1,0,2))
        
        return out_queries


class TransformerEncoderLayer(nn.Module):
    '''
    Transformer module (attention mechanism) that takes (N, F_in) feature
    tensors to (N, F_out) feature tensor. 
    '''
    def __init__(self, num_input, num_output, num_hidden=128, num_layers=5, 
                 d_qk=128, d_v=128, num_heads=8, leakiness=0.0, dropout=True, 
                 name='attention_net', norm_layer='layer_norm'):
        super(TransformerEncoderLayer, self).__init__()

        self.ma_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        self.num_layers = num_layers

        for i in range(num_layers):
            self.ma_layers.append(
                MultiHeadAttention(num_input, d_qk, d_v, num_hidden, 
                                   num_heads=num_heads, dropout=dropout,
                                   norm_layer=norm_layer)
            )
            self.ffn_layers.append(
                PositionWiseFFN(num_hidden, num_hidden, num_hidden, 
                                leakiness=leakiness, dropout=dropout,
                                norm_layer=norm_layer)
            )
            num_input = num_hidden

        self.softplus = nn.Softplus()

    def forward(self, x):

        for i in range(self.num_layers):
            # print(x.shape)
            x = self.ma_layers[i](x)
            # print(x.shape)
            x = self.ffn_layers[i](x)
            # print(x.shape)
            
        x = self.softplus(x)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_input, d_qk, d_v, num_output, 
                 num_heads=8, dropout=False, norm_layer='layer_norm'):
        super(MultiHeadAttention, self).__init__()
        
        self.num_input = num_input
        self.num_output = num_output
        self.num_heads = num_heads
        self.d_qk = d_qk
        self.d_v = d_v
        self.T = num_output ** 0.5
        
        if norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm(num_output)
        elif norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm1d(num_output)
        else:
            raise ValueError('Normalization layer {} not recognized!'.format(norm_layer))
        
        self.W_q = nn.Linear(num_input, d_qk * num_heads, bias=False)
        self.W_k = nn.Linear(num_input, d_qk * num_heads, bias=False)
        self.W_v = nn.Linear(num_input, d_v * num_heads, bias=False)
        self.W_o = nn.Linear(d_v * num_heads, num_output, bias=False) 
        
        self.dropout = dropout
        if self.dropout:
            self.dropout1 = nn.Dropout()

        if self.num_input == self.num_output:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(self.num_input, self.num_output)
        
        
    def forward(self, x):
        
        num_output, num_heads = self.num_output, self.num_heads
        residual = self.residual(x)
        
        q = self.W_q(x).view(-1, self.d_qk, num_heads)
        k = self.W_k(x).view(-1, self.d_qk, num_heads)
        v = self.W_v(x).view(-1, self.d_v, num_heads)
        
        a = torch.einsum('ikb,jkb->ijb', q, k)
        attention = F.softmax(a / self.T, dim=1)
        v = torch.einsum('bih,ijh->bjh', attention, v).contiguous()
        v = v.view(-1, num_heads * self.d_v)
        out = self.W_o(v)
        
        out += residual
        if self.dropout:
            out = self.dropout1(out)
        
        return self.norm(out)


class PositionWiseFFN(nn.Module):
    
    def __init__(self, num_input, num_output, num_hidden, 
                 leakiness=0.0, dropout=True, norm_layer='layer_norm'):
        super(PositionWiseFFN, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_output)
        if norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm(num_output)
        elif norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm1d(num_output)
        else:
            raise ValueError('Normalization layer {} not recognized!'.format(norm_layer))
        self.act = nn.LeakyReLU(negative_slope=leakiness)

        self.dropout = dropout
        if self.dropout:
            self.dropout1 = nn.Dropout()

        
    def forward(self, x):
        residual = x
        out = self.act(self.linear1(x))
        out = self.linear2(out)
        if self.dropout:
            out = self.dropout1(out)
        out += residual
        return self.norm(out)


# ---------------------------------------------------------------------------
# From Mask3D/models/mask3d.py by Jonas Schult:
# https://github.com/JonasSchult/Mask3D

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask= None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask= None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


NORM_DICT = {
    # "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output