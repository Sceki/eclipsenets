"""Training of EclipseNETs on datasets of eclipse function samples."""
import pickle
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from tqdm import tqdm

from .nn import FFNN, SIREN, count_parameters


@dataclass
class TrainingConfig:
    """Architecture and hyperparameters of a training run.

    The defaults give the two architectures of the paper, both with 2,369
    parameters: a SIREN with three sine layers of 32 neurons (`hidden_layers`
    counts the sine layers after the first one) and a feed-forward network
    with three hidden layers of 32 neurons.
    """
    # Model architecture
    model_type: str = "siren"  # "siren" or "ffnn"
    input_dim: int = 6  # set from the data by train()

    # SIREN specific
    hidden_features: int = 32
    hidden_layers: int = 2
    omega_0: float = 30.0

    # FFNN specific
    hidden_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "relu"

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 100
    seed: Optional[int] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoints (none are written if checkpoint_dir is None)
    checkpoint_dir: Optional[str] = None
    save_frequency: int = 10


def build_model(config: TrainingConfig) -> nn.Module:
    """Creates the model described by a configuration."""
    if config.model_type == "siren":
        return SIREN(
            in_features=config.input_dim,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            out_features=1,
            outermost_linear=True,
            first_omega_0=config.omega_0,
            hidden_omega_0=config.omega_0
        )
    elif config.model_type == "ffnn":
        return FFNN(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=1,
            activation=config.activation
        )
    raise ValueError(f"Unknown model type: {config.model_type}")


def _to_tensors(data, device):
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    return data[:, :-1], data[:, -1]


@torch.no_grad()
def evaluate_mse(model: nn.Module, data, batch_size: int = 2**16) -> float:
    """
    Mean squared error of a model on a dataset.

    Args:
        model: the model.
        data: array of shape (N, features + 1), where the last column is the target.
        batch_size: number of samples evaluated at once.

    Returns:
        the mean squared error.
    """
    parameter = next(model.parameters())
    inputs, targets = _to_tensors(data, parameter.device)
    was_training = model.training
    model.eval()
    total = 0.0
    for start in range(0, len(inputs), batch_size):
        outputs = model(inputs[start:start + batch_size].to(parameter.dtype)).squeeze(-1)
        total += torch.sum((outputs - targets[start:start + batch_size]) ** 2).item()
    model.train(was_training)
    return total / len(inputs)


def _save_checkpoint(path, model, config, epoch, train_loss, valid_loss):
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }, path)


def train(train_data, valid_data, config: Optional[TrainingConfig] = None, progress: bool = True):
    """
    Trains an EclipseNET, with Adam and a cosine annealing of the learning rate.

    Args:
        train_data: array of shape (N, features + 1), where the last column is the target.
        valid_data: array with the same columns, used for validation.
        config: architecture and hyperparameters; the defaults are used if None.
        progress: whether to show a progress bar for each epoch.

    Returns:
        the trained model (on the CPU, in evaluation mode) and a dictionary with:
        'train_losses' and 'valid_losses' (one value per epoch), 'batch_losses'
        (one value per optimization step) and 'best_valid_loss'. If
        `config.checkpoint_dir` is set, the dictionary is saved there as
        <model_type>_history.pkl, the model with the best validation loss as
        <model_type>_best.pt, and one checkpoint every `config.save_frequency`
        epochs as <model_type>_epoch_<n>.pt: the models can be read with
        :func:`eclipsenets.load_model`.
    """
    config = TrainingConfig() if config is None else config
    if config.seed is not None:
        torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_inputs, train_targets = _to_tensors(train_data, device)
    valid_data = torch.as_tensor(valid_data, dtype=torch.float32, device=device)
    config = replace(config, input_dim=train_inputs.shape[1])

    model = build_model(config).to(device)
    print(f"Model with {count_parameters(model):,} parameters, training on {device}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)

    if config.checkpoint_dir is not None:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_losses": [], "valid_losses": [], "batch_losses": [], "best_valid_loss": float("inf")}
    n_samples = len(train_inputs)
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        # The dataset already sits on the device: slicing a permutation of it
        # is much faster than going through a DataLoader.
        permutation = torch.randperm(n_samples, device=device)
        starts = range(0, n_samples, config.batch_size)
        for start in tqdm(starts, desc=f"Epoch {epoch + 1}/{config.epochs}", disable=not progress):
            batch = permutation[start:start + config.batch_size]
            loss = criterion(model(train_inputs[batch]).squeeze(-1), train_targets[batch])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["batch_losses"].append(loss.item())
            epoch_loss += loss.item() * len(batch)
        scheduler.step()

        train_loss = epoch_loss / n_samples
        valid_loss = evaluate_mse(model, valid_data)
        history["train_losses"].append(train_loss)
        history["valid_losses"].append(valid_loss)
        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.6e}, Valid Loss: {valid_loss:.6e}")

        is_best = valid_loss < history["best_valid_loss"]
        if is_best:
            history["best_valid_loss"] = valid_loss
        if config.checkpoint_dir is not None:
            if is_best:
                _save_checkpoint(checkpoint_dir / f"{config.model_type}_best.pt",
                                 model, config, epoch, train_loss, valid_loss)
            if (epoch + 1) % config.save_frequency == 0:
                _save_checkpoint(checkpoint_dir / f"{config.model_type}_epoch_{epoch + 1}.pt",
                                 model, config, epoch, train_loss, valid_loss)

    print(f"Training complete! Best validation loss: {history['best_valid_loss']:.6e}")
    if config.checkpoint_dir is not None:
        with open(checkpoint_dir / f"{config.model_type}_history.pkl", "wb") as f:
            pickle.dump(history, f)
    return model.cpu().eval(), history
