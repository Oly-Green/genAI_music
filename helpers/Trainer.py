import torch
from torch import nn, device, cuda
from torch.optim import Adam
from modelClasses.Generator import Generator
from modelClasses.Discriminator import Discriminator

class Trainer:
    def __init__(self, input_size, hidden_size_linear, hidden_size_attention, output_size, batch_size, padding_value=-1):
        self.input_size = input_size
        self.hidden_size_linear = hidden_size_linear
        self.hidden_size_attention = hidden_size_attention
        self.output_size = output_size
        self.batch_size = batch_size
        # self.d_steps = d_steps
        self.padding_value = padding_value
        self.device = device('cuda' if cuda.is_available() else 'cpu')

        self.model()

    def model(self):
        self.G = Generator(self.input_size, self.hidden_size_linear, self.hidden_size_attention, self.output_size, self.padding_value)
        self.D = Discriminator(self.input_size, self.hidden_size_linear)

        self.G.to(self.device)
        self.D.to(self.device)

        self.G_optimizer = Adam(self.G.parameters(), lr=1e-4)
        self.D_optimizer = Adam(self.G.parameters(), lr=1e-4)

        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCELoss()

    def trainDiscriminator(self, xReal, xFake, mask):
        self.D_optimizer.zero_grad()

        xReal = xReal.squeeze()
        DOutReal = self.D(xReal)
        ones = torch.ones_like(DOutReal)
        DLossReal = self.loss(DOutReal, ones)

        self.G.setMask(mask)
        DOutFake = self.D(self.G(xFake))
        zeros = torch.zeros_like(DOutReal)
        DLossFake = self.loss(DOutFake, zeros)

        DLossTotal = DLossReal + DLossFake
        DLossTotal.backward()
        self.D_optimizer.step()

        return DLossTotal, DOutReal, DOutFake

    def trainGenerator(self, xFake, mask):
        self.G_optimizer.zero_grad()
        self.G.setMask(mask)
        pred = self.D(self.G(xFake))
        ones = torch.ones_like(pred)
        GLoss = self.loss(pred, ones)
        GLoss.backward()
        self.G_optimizer.step()
        return GLoss