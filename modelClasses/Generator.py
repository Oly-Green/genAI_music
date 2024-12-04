import torch
from torch import device, cuda
from torch import nn
from helpers.SelfAttention import SelfAttend

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size_linear, hidden_size_attention, output_size, padding_value, unique_vals_0, unique_vals_1, unique_vals_2):
        super(Generator, self).__init__()

        self.padding_value = padding_value

        self.cudaDevice = device('cuda' if cuda.is_available() else 'cpu')

        self.unique_vals = [torch.tensor(unique_vals_0, dtype=torch.float32).to(self.cudaDevice),
                            torch.tensor(unique_vals_1, dtype=torch.float32).to(self.cudaDevice),
                            torch.tensor(unique_vals_2, dtype=torch.float32).to(self.cudaDevice)]
        self.max_uniques = [max(self.unique_vals[0]), max(self.unique_vals[1]), max(self.unique_vals[2])]
        self.max_uniques = torch.stack(self.max_uniques)

        self.mean_uniques = [self.unique_vals[0].mean(), self.unique_vals[1].mean(), self.unique_vals[2].mean()]
        self.mean_uniques = torch.stack(self.mean_uniques)

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

        self.a1 = SelfAttend(hidden_size_linear, hidden_size_attention, hidden_size_attention)
        self.aRelu1 = nn.ReLU()

        self.a2 = SelfAttend(hidden_size_attention, hidden_size_attention, hidden_size_linear)
        self.aRelu2 = nn.ReLU()

        self.out = nn.Linear(hidden_size_linear, output_size)

    # Deprecated
    def setMask(self, mask):
        self.mask = mask

    # def snapValues(self, batch, col_idx):
    #     snapped_batch = []
    #     for tens in batch:
    #         scaled_row = tens.T[col_idx] * self.max_uniques[col_idx]
    #         snapped_row = []
    #
    #         for byte in scaled_row:
    #             differences = [byte - unq_val for unq_val in self.unique_vals[col_idx]]
    #             min_idx = differences.index(min(differences))
    #             snapped_byte = self.unique_vals[col_idx][min_idx]
    #             snapped_row.append(snapped_byte)
    #         snapped_row = torch.tensor(snapped_row)
    #
    #         snapped_batch.append(snapped_row)
    #     snapped_batch = torch.stack(snapped_batch)
    #     return snapped_batch

    # Deprecated
    def snapValues(self, batch, col_idx):

        # if batch.ndimension() == 1:  # Single input (no batch)
        #     batch = batch.unsqueeze(0)

        # Scale the rows in the batch
        scaled_batch = batch[:, col_idx] # * self.max_uniques[col_idx]  # Shape: (batch_size, N)
        # scaled_batch = batch

        # Scale each column by the corresponding maximum unique value
        # scaled_batch = batch * self.max_uniques  # Broadcasting handles scaling per column

        # scaled_batch = batch.permute(*torch.arange(batch.ndim - 1, -1, -1))[col_idx] * self.max_uniques[col_idx]
        # scaled_batch = scaled_batch.permute(*torch.arange(scaled_batch.ndim - 1, -1, -1))

        # Convert unique_vals to a tensor
        unique_vals = self.unique_vals[col_idx] # Shape: (M)

        # Expand scaled_batch to (batch_size, N, 1) and unique_vals to (1, 1, M)
        # This will allow broadcasting to calculate differences for all values in the batch
        try:
            scaled_batch_expanded = scaled_batch.unsqueeze(2)  # Shape: (batch_size, N, 1)
        except IndexError:
            scaled_batch_expanded = scaled_batch.unsqueeze(1)

        unique_vals_expanded = unique_vals.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, M)

        # Calculate the absolute differences
        differences = torch.abs(scaled_batch_expanded - unique_vals_expanded)  # Shape: (batch_size, N, M)

        # Find the indices of the minimum differences along the last dimension (M)
        min_indices = torch.argmin(differences, dim=2)  # Shape: (batch_size, N)

        # Gather the snapped values using the indices
        snapped_batch = unique_vals[min_indices]  # Shape: (batch_size, N)

        return snapped_batch.squeeze()

    def quantize_to_unique_vals(self, continuous_output):
        # Create a placeholder for quantized output
        quantized_output = continuous_output.clone()
        quantized_output = quantized_output.permute(*torch.arange(quantized_output.ndim - 1, -1, -1))

        for col_idx, unique_vals in enumerate(self.unique_vals):
            # Broadcast unique_vals to match batch size
            diffs = []
            for unique_val in unique_vals:
                diffs.append(torch.abs(quantized_output[col_idx].unsqueeze(1) - unique_val.unsqueeze(0)))  # (batch_size, num_unique_vals)
            diffs = torch.stack(diffs)
            nearest_idx = torch.argmin(diffs, dim=0)  # Get indices of the closest values
            nearest_values =  torch.stack([unique_vals[idx] for idx in nearest_idx]).squeeze()
            quantized_output[col_idx] = nearest_values # Replace with closest values
        quantized_output = quantized_output.permute(*torch.arange(quantized_output.ndim - 1, -1, -1))
        return quantized_output

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)

        attn1 = self.a1(h3)
        attn1A = self.aRelu1(attn1)

        attn2 = self.a2(attn1A)
        attn2A = self.aRelu2(attn2)

        out = self.out(attn2A)

        # out = torch.flip(out, dims=[0])

        scaled_out = out * self.max_uniques

        # quantized_output = self.quantize_to_unique_vals(scaled_out)
        return scaled_out

        # discrete_out = scaled_out.clone()
        # for col_idx, unique_vals in enumerate(self.unique_vals):
        #     # Compute the nearest unique value for each element in the column
        #     diffs = torch.abs(
        #         scaled_out[:, col_idx].unsqueeze(1) - unique_vals.unsqueeze(0))  # (batch_size, num_unique_vals)
        #     nearest_idx = torch.argmin(diffs, dim=1)  # Indices of nearest values
        #     discrete_out[:, col_idx] = unique_vals[nearest_idx]  # Replace with nearest values
        #
        # return discrete_out.squeeze()

        # snapped_out = torch.stack(
        #     [self.snapValues(out, 0),
        #     self.snapValues(out, 1),
        #     self.snapValues(out, 2)]
        # )
        #
        # snapped_out = snapped_out.permute(*torch.arange(x.ndim - 1, -1, -1))

        # out = torch.where(out < 0, 0, out)

        # return out.squeeze()
