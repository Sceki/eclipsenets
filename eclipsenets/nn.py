import torch
import numpy as np
from torch import nn

class SineLayer(nn.Module):
    """Sinusoidal activation layer for SIREN networks"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                           1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Sinusoidal Representation Network"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0
    ):
        super().__init__()

        layers = []
        layers.append(SineLayer(in_features, hidden_features, 
                               is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                   is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features,
                                   is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFNN(nn.Module):
    """Feed-Forward Neural Network with flexible architecture"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Don't add activation after last layer
                layers.append(activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def weights_and_biases_heyoka(model):
    weights = {}
    biases = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights[name] = param.data.clone()
        elif 'bias' in name:
            biases[name] = param.data.clone()
    biases_torch=[]
    weights_torch=[]
    for idx in range(len(weights)):
        weights_torch.append(weights[list(weights.keys())[idx]].numpy())
        biases_torch.append(biases[list(biases.keys())[idx]].numpy())
        
    w_flat=[]
    b_flat=[]
    for i in range(len(weights_torch)):
        w_flat+=list(weights_torch[i].flatten())
        b_flat+=list(biases_torch[i].flatten())
    w_flat=np.array(w_flat)
    b_flat=np.array(b_flat)
    print(w_flat.shape)
    return np.concatenate((w_flat, b_flat))