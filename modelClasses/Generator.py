import torch
from torch import nn
from helpers.SelfAttention import SelfAttend

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size_linear, hidden_size_attention, output_size, padding_value):
        super(Generator, self).__init__()

        self.padding_value = padding_value

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

        # self.lstm = nn.LSTM(hidden_size_linear, hidden_size_attention, batch_first=True)
        # self.out = nn.Linear(hidden_size_attention, output_size)

        # self.a1 = SelfAttend(hidden_size_linear, hidden_size_attention, hidden_size_linear)
        # self.aRelu1 = nn.ReLU()

        self.out = nn.Linear(hidden_size_linear, output_size)

    def setMask(self, mask):
        self.mask = mask

    def forward(self, x):
        masked_x = x * self.mask

        h1 = self.l1(masked_x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)

        # attn1 = self.a1(h3)
        # attn1A = self.aRelu1(attn1)
        #
        # out = self.out(attn1A)

        # lstm_output, (h_n, c_n) = self.lstm(h3)
        # out = self.out(lstm_output)

        out = self.out(h3)

        out *= self.mask
        out = out * 255
        final_output = torch.where(out < 0, -1.0, out)
        return final_output.squeeze()
