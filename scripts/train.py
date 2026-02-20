#!/usr/bin/env python3

import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import argparse

import numpy as np
import sys
sys.path.append('../')
import eclipsenets as en
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Training hyperparameters and paths"""
    # Data paths
    mesh_path: str = "../3dmeshes/Churyumov-Gerasimenko_raw.pk"
    train_data_path: str = "../datasets/chu-ger_db_2349414_L=5002.5703125_larger_train.pk"
    valid_data_path: str = "../datasets/chu-ger_db_939678_L=5002.5703125_valid.pk"

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 100

    # Model architecture
    input_dim: int = 6  # Will be set from data
    hidden_dims: list = None
    output_dim: int = 1

    # Model type: 'ffnn' or 'siren'
    model_type: str = 'ffnn'

    # SIREN specific
    hidden_features: int = 32
    hidden_layers: int = 2
    omega_0: float = 30.0

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpoint
    checkpoint_dir: str = "models"
    save_frequency: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 32, 32]


class EclipseDataset(Dataset):
    """PyTorch Dataset for eclipse function data"""

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: numpy array of shape (N, features+1) where last column is target
        """
        self.inputs = torch.from_numpy(data[:, :-1]).float()
        self.targets = torch.from_numpy(data[:, -1]).float()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

class Trainer:
    """Training pipeline with checkpointing and validation"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load data
        self.train_dataset, self.valid_dataset = self._load_data()

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if config.device == 'cuda' else False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.device == 'cuda' else False
        )

        # Initialize model
        self.model = self._create_model().to(self.device)

        # Setup training
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            amsgrad=True
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=0
        )

        # Tracking
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"Model initialized with {self._count_parameters():,} parameters")
        print(f"Training on {config.device}")

    def _load_data(self) -> Tuple[EclipseDataset, EclipseDataset]:
        """Load and preprocess training and validation data"""
        with open(self.config.train_data_path, 'rb') as f:
            train_data = pickle.load(f)

        with open(self.config.valid_data_path, 'rb') as f:
            valid_data = pickle.load(f)

        # Update input_dim from data
        self.config.input_dim = train_data.shape[1] - 1

        print(f"Training data shape: {train_data.shape}")
        print(f"Validation data shape: {valid_data.shape}")

        # Count shadow vs light points
        shadow = (train_data[:, -1] < 0).sum()
        light = (train_data[:, -1] > 0).sum()
        print(f"Shadow: {shadow:,}, Light: {light:,}")

        return EclipseDataset(train_data), EclipseDataset(valid_data)

    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        if self.config.model_type == 'siren':
            return en.SIREN(
                in_features=self.config.input_dim,
                hidden_features=self.config.hidden_features,
                hidden_layers=self.config.hidden_layers,
                out_features=self.config.output_dim,
                outermost_linear=True,
                first_omega_0=self.config.omega_0,
                hidden_omega_0=self.config.omega_0
            )
        elif self.config.model_type == 'ffnn':
            return en.FFNN(
                input_dim=self.config.input_dim,
                hidden_dims=self.config.hidden_dims,
                output_dim=self.config.output_dim,
                activation=nn.Tanh()
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return epoch_loss / len(self.train_dataset)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0

        for inputs, targets in self.valid_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

        return total_loss / len(self.valid_dataset)

    def save_checkpoint(self, epoch: int, train_loss: float, valid_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) /             f"{self.config.model_type}_epoch_{epoch}_loss_{valid_loss:.6e}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            best_path = Path(self.config.checkpoint_dir) /                 f"{self.config.model_type}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved (valid loss: {valid_loss:.6e})")

    def train(self):
        """Full training loop"""
        print("Starting training...")

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            valid_loss = self.validate()
            self.valid_losses.append(valid_loss)

            # Learning rate step
            self.scheduler.step()

            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train Loss: {train_loss:.6e}, Valid Loss: {valid_loss:.6e}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_frequency == 0 or                valid_loss < self.best_valid_loss:
                self.save_checkpoint(epoch, train_loss, valid_loss)

        print(f"Training complete! Best validation loss: {self.best_valid_loss:.6e}")

        return {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'best_valid_loss': self.best_valid_loss
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train EclipseNET neural network')

    # Data paths
    parser.add_argument('--mesh-path', type=str, 
                       default='../3dmeshes/Churyumov-Gerasimenko_raw.pk')
    parser.add_argument('--train-data', type=str,
                       default='../datasets/chu-ger_db_2349414_L=5002.5703125_larger_train.pk')
    parser.add_argument('--valid-data', type=str,
                       default='../datasets/chu-ger_db_939678_L=5002.5703125_valid.pk')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)

    # Model architecture
    parser.add_argument('--model-type', type=str, default='ffnn',
                       choices=['ffnn', 'siren'])
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[32, 32, 32])
    parser.add_argument('--hidden-features', type=int, default=32)
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--omega-0', type=float, default=30.0)

    # Device and checkpointing
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='models')
    parser.add_argument('--save-freq', type=int, default=10)

    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()

    # Create config from args
    config = TrainingConfig(
        mesh_path=args.mesh_path,
        train_data_path=args.train_data,
        valid_data_path=args.valid_data,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        model_type=args.model_type,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        omega_0=args.omega_0,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_freq
    )

    # Initialize trainer and train
    trainer = Trainer(config)
    results = trainer.train()

    # Save training history
    history_path = Path(config.checkpoint_dir) / f"{config.model_type}_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()