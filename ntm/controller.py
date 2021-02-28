"""LSTM Controller."""
import itertools

import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes=None):
        super(MLP, self).__init__()
        hidden_layer_sizes = hidden_layer_sizes or [int((input_size + output_size/ 2))]
        sizes = [input_size] + hidden_layer_sizes + [output_size]
        ios = list(zip(sizes, sizes[1:]))
        seqs = [[nn.Linear(i, o), nn.ReLU()] for i, o in ios]
        seq_list = list(itertools.chain.from_iterable(seqs))[:-1] # + [nn.Sigmoid()]
        self.layers = nn.Sequential(*seq_list)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x


class FFNNController(nn.Module):
    """An NTM controller based on FFNN."""
    def __init__(self, num_inputs, num_outputs, controller_layers=1, hidden_layer_sizes=None):
        super(FFNNController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes or FFNNController._generate_hidden_layer_sizes(
            num_inputs, num_outputs, controller_layers)
        self.mlp = MLP(input_size=num_inputs,
                       output_size=num_outputs,
                       hidden_layer_sizes=hidden_layer_sizes)
        print("MLP", self.mlp)

        # The hidden state is a learned parameter

        self.reset_parameters()

    def create_new_state(self, batch_size):
        return None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        #x = x.unsqueeze(0)
        outp = self.mlp(x)
        return outp, None

    @staticmethod
    def _generate_hidden_layer_sizes(num_inputs, num_outputs, controller_layers):
        return np.linspace(num_inputs, num_outputs, controller_layers).astype(int)[1:-1].tolist()

