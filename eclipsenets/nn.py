import re
from pathlib import Path
from typing import Optional, Sequence, Union

import heyoka as hk
import numpy as np
import torch
from torch import nn

from .bodies import DATA_DIR, get_body_info


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
    """Sinusoidal Representation Network.

    The network has `hidden_layers` + 1 sine layers of `hidden_features`
    neurons, followed by the output layer.
    """

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


_ACTIVATIONS = {
    "relu": (torch.relu, hk.relu),
    "tanh": (torch.tanh, hk.tanh),
    "sigmoid": (torch.sigmoid, hk.sigmoid),
}


class FFNN(nn.Module):
    """Feed-Forward Neural Network with flexible architecture"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: Union[str, nn.Module] = "relu"
    ):
        super().__init__()
        if isinstance(activation, nn.Module):
            activation = type(activation).__name__.lower()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}: use one of {list(_ACTIVATIONS)}")
        self.activation = activation

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = _ACTIVATIONS[self.activation][0]
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        return self.layers[-1](x)  # no activation after the last layer


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _layers(model):
    """The linear layers of a SIREN or FFNN, each with its heyoka activation."""
    def identity(x):
        return x

    def sine(omega_0):
        return lambda x: hk.sin(omega_0 * x)

    if isinstance(model, SIREN):
        return [(layer.linear, sine(layer.omega_0)) if isinstance(layer, SineLayer) else (layer, identity)
                for layer in model.net]
    if isinstance(model, FFNN):
        activation = _ACTIVATIONS[model.activation][1]
        return [(layer, activation) for layer in model.layers[:-1]] + [(model.layers[-1], identity)]
    raise TypeError(f"Expected a SIREN or FFNN model, got {type(model).__name__}")


def weights_and_biases_heyoka(model: nn.Module) -> np.ndarray:
    """
    Flattens the parameters of a model in the order expected by heyoka's
    `model.ffnn`: all the weights, layer by layer, followed by all the biases.

    Args:
        model: a SIREN or FFNN model.

    Returns:
        1D np.array with all the parameters, in double precision.
    """
    linears = [linear for linear, _ in _layers(model)]
    flat = [linear.weight.detach().cpu().numpy().ravel() for linear in linears]
    flat += [linear.bias.detach().cpu().numpy().ravel() for linear in linears]
    return np.concatenate(flat).astype(float)


def heyoka_model(model: nn.Module, inputs: Sequence, weights: Optional[Sequence] = None):
    """
    Builds the heyoka expression of a torch model.

    Args:
        model: a SIREN or FFNN model.
        inputs: the heyoka expressions fed to the network.
        weights: the weights and biases to use, ordered as in
            :func:`weights_and_biases_heyoka`. By default the values of the
            parameters of the model are used; passing heyoka parameters
            instead leaves the weights free to change at runtime.

    Returns:
        list of heyoka expressions, one per output of the network.
    """
    layers = _layers(model)
    if weights is None:
        weights = weights_and_biases_heyoka(model)
    return hk.model.ffnn(
        inputs=list(inputs),
        nn_hidden=[linear.out_features for linear, _ in layers[:-1]],
        n_out=layers[-1][0].out_features,
        activations=[activation for _, activation in layers],
        nn_wb=weights,
    )


def load_model(path, activation: str = "relu", omega_0: float = 30.0) -> nn.Module:
    """
    Loads a SIREN or FFNN model from a file.

    The file can either be a checkpoint written by :func:`eclipsenets.train`
    or a bare state dict, as the models in models/ are. In the second case the
    architecture is read off the shapes of the weights, while the activation
    of a feed-forward network and the frequency of a SIREN, which are not
    stored, are taken from the arguments.

    Args:
        path: path of the file.
        activation: activation function of a feed-forward network.
        omega_0: frequency of the sine layers of a SIREN.

    Returns:
        the model, in evaluation mode.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("model_config", {})
    activation = config.get("activation", activation)
    omega_0 = config.get("omega_0", omega_0)

    if any(key.startswith("layers.") for key in state):
        weights = [state[key] for key in sorted(
            (key for key in state if key.endswith(".weight")), key=lambda key: int(key.split(".")[1]))]
        model = FFNN(weights[0].shape[1], [w.shape[0] for w in weights[:-1]], weights[-1].shape[0],
                     activation=activation)
    else:
        sine_weights = [value for key, value in state.items() if key.endswith(".linear.weight")]
        final_weight = [value for key, value in state.items() if re.fullmatch(r"net\.\d+\.weight", key)]
        outermost_linear = len(final_weight) == 1
        n_sine = len(sine_weights) - (0 if outermost_linear else 1)
        out_features = (final_weight[0] if outermost_linear else sine_weights[-1]).shape[0]
        model = SIREN(sine_weights[0].shape[1], sine_weights[0].shape[0], n_sine - 1, out_features,
                      outermost_linear=outermost_linear, first_omega_0=omega_0, hidden_omega_0=omega_0)
    # keep the precision the weights were saved with
    model.to(next(iter(state.values())).dtype).load_state_dict(state)
    return model.eval()


# The activation of the released feed-forward networks is not stored in their
# files. It was identified as the one that reproduces the losses in the file
# names: the networks of 67P use tanh, the others ReLU.
_PRETRAINED_FFNN_ACTIVATION = {"bennu": "relu", "itokawa": "relu", "chu-ger": "tanh", "eros": "relu"}


def pretrained_path(body: str, kind: str = "siren", epoch: Optional[int] = None) -> Path:
    """
    Path of one of the networks of the paper, stored in models/ffnn_vs_siren.

    Args:
        body: name of the body.
        kind: "siren" or "ffnn".
        epoch: by default the fully trained network is returned; an epoch
            selects one of the intermediate checkpoints instead (epochs 0
            and 7 of the feed-forward network of 67P are available).
    """
    key = get_body_info(body).key
    stage = "2_32_" if epoch is None else f"epoch_{epoch}"
    matches = sorted((DATA_DIR / "models" / "ffnn_vs_siren").glob(f"{key}_model_{kind}_{stage}_*.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {kind} model for {body} (epoch={epoch}) in models/ffnn_vs_siren, found {len(matches)}")
    return matches[0]


def load_pretrained(body: str, kind: str = "siren", epoch: Optional[int] = None) -> nn.Module:
    """Loads one of the networks of the paper (see :func:`pretrained_path`)."""
    activation = _PRETRAINED_FFNN_ACTIVATION[get_body_info(body).key]
    return load_model(pretrained_path(body, kind, epoch), activation=activation)
