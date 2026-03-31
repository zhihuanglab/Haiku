import torch

import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, activation="relu", final_activation=None, bias=True):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=bias)])

        for _ in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        self.activation = get_activation(activation)

        if final_activation is not None:
            self.apply_final_activation = True
            self.final_activation = get_activation(final_activation)
        else:
            self.apply_final_activation = False

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.apply_final_activation:
            x = self.final_activation(x)
        return x


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError("Activation function not implemented")