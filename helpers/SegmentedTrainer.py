import torch
from torch import nn, device, cuda
from torch.optim import Adam
from modelClasses.Generator import Generator
from modelClasses.Discriminator import Discriminator

class Trainer:
    def __init__(self, input_size, G_hidden_size_linear, G_hidden_size_attention, D_hidden_size, output_size, batch_size, unique_values_0, unique_values_1, unique_values_2, padding_value=-1, g_lr=1e-4, d_lr=1e-4):
        self.input_size = input_size
        self.hidden_size_linear = G_hidden_size_linear
        self.hidden_size_attention = G_hidden_size_attention
        self.D_hidden_size = D_hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.padding_value = padding_value
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.unique_values_0 = unique_values_0
        self.unique_values_1 = unique_values_1
        self.unique_values_2 = unique_values_2
        self.device = device('cuda' if cuda.is_available() else 'cpu')

        self.model()

    def model(self):

        self.G = Generator(self.input_size, self.hidden_size_linear, self.hidden_size_attention, self.output_size, self.padding_value, self.unique_values_0, self.unique_values_1, self.unique_values_2)
        self.D = Discriminator(self.input_size, self.D_hidden_size)

        self.G.to(self.device)
        self.D.to(self.device)

        self.G = self.G.apply(self.initWeights)
        self.D = self.D.apply(self.initWeights)

        self.G_optimizer = Adam(self.G.parameters(), lr=self.g_lr)
        self.D_optimizer = Adam(self.D.parameters(), lr=self.d_lr)

        self.loss = nn.BCELoss()
        # self.loss = nn.MSELoss()

    def initWeights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def trainDiscriminator(self, output_chunk, input_chunk):
        self.D_optimizer.zero_grad()

        xReal = output_chunk.squeeze()
        DOutReal = self.D(xReal)
        ones = torch.ones_like(DOutReal)
        DLossReal = self.loss(DOutReal, ones)

        DOutFake = self.D(self.G(input_chunk))
        zeros = torch.zeros_like(DOutFake)
        DLossFake = self.loss(DOutFake, zeros)

        # DLossTotal = DLossReal + DLossFake
        DLossTotal = 0.5 * (DLossReal + DLossFake)
        DLossTotal.backward()
        self.D_optimizer.step()

        return DLossTotal, DOutReal, DOutFake

    def trainGenerator(self, input_chunk):
        self.G_optimizer.zero_grad()

        noise = torch.rand_like(input_chunk)
        noised_input = input_chunk + noise
        output_chunk = self.G(noised_input)

        pred = self.D(output_chunk)
        ones = torch.ones_like(pred)
        GLoss = self.loss(pred, ones)
        GLoss.backward()
        self.G_optimizer.step()
        return GLoss