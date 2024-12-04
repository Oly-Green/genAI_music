import torch
from torch import nn
from math import sqrt

class SelfAttend(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttend, self).__init__()

        self.Q = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.K = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.V = nn.Linear(in_features=input_size, out_features=hidden_size)

        self.output = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)

        attention = nn.functional.scaled_dot_product_attention(query, key, value)

        # d_k = query.size(-1)

        # with torch.cuda.amp.autocast():
        #     scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(d_k)
        #     attention_weights = torch.softmax(scores, dim=-1)
        #     attention = torch.bmm(attention_weights, value)

        # del scores, attention_weights

        out = self.output(attention)

        return out.squeeze(1)