import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, num_input, num_output, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.W_q = nn.Linear(num_input, num_heads * num_output)
        self.W_k = nn.Linear(num_input, num_heads * num_output)
        self.W_v = nn.Linear(num_input, num_heads * num_output)


    def forward(self, x, edge_index, edge_attr, u=None, batch=None):

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        print(q.shape, k.shape)

        pass