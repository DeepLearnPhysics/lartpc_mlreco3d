import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    '''
    Transformer module (attention mechanism) that takes (N, F_in) feature
    tensors to (N, F_out) feature tensor. 
    '''
    def __init__(self, num_input, num_output, num_hidden=64, num_layers=3, 
                 d_qk=64, d_v=64, num_heads=8, leakiness=0.0, dropout=True, 
                 name='attention_net'):
        super(TransformerEncoderLayer, self).__init__()

        self.ma_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        self.num_layers = num_layers

        for i in range(num_layers):
            self.ma_layers.append(
                MultiHeadAttention(num_input, d_qk, d_v, num_hidden, 
                                   num_heads=num_heads, dropout=dropout)
            )
            self.ffn_layers.append(
                PositionWiseFFN(num_input, num_hidden, num_hidden, 
                                leakiness=leakiness, dropout=dropout)
            )
            num_input = num_hidden

    def forward(self, x):

        for i in range(self.num_layers):
            x = self.ma_layers[i](x)
            x = self.ffn_layers[i](x)

        x = x.mean(dim=1)
            
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
            self.norm = nn.BatchNorm(num_output)
        else:
            raise ValueError('Normalization layer {} not recognized!'.format(norm_layer))
        
        self.W_q = nn.Linear(num_input, d_qk * num_heads, bias=False)
        self.W_k = nn.Linear(num_input, d_qk * num_heads, bias=False)
        self.W_v = nn.Linear(num_input, d_v * num_heads, bias=False)
        self.W_o = nn.Linear(d_v * num_heads, num_output, bias=False) 
        
        self.dropout = dropout
        if self.dropout:
            self.dropout1 = nn.Dropout()
        
        
    def forward(self, x):
        
        num_output, num_heads = self.num_output, self.num_heads
        
        residual = x
        
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
                 leakiness=0.0, dropout=True):
        super(PositionWiseFFN, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_output)
        self.norm = nn.LayerNorm(num_output)
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


