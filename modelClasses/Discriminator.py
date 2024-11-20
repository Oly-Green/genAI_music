import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size_linear):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size, hidden_size_linear),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size_linear, hidden_size_linear),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size_linear, hidden_size_linear),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            torch.nn.Linear(hidden_size_linear, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        out = self.out(h3)
        return out.squeeze()