#!/usr/bin/env python3
"""Trains an EclipseNET on a dataset of eclipse function samples.

By default the datasets stored in datasets/ are used, e.g.:

    python scripts/train.py --body 67P --model-type siren
    python scripts/train.py --body 67P --model-type ffnn --activation tanh
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eclipsenets as en


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train EclipseNET neural network')

    # Data: either a body, to use its datasets in datasets/, or explicit paths
    parser.add_argument('--body', type=str, default='Churyumov-Gerasimenko',
                       help='body whose datasets in datasets/ are used')
    parser.add_argument('--train-data', type=str, default=None,
                       help='training dataset (overrides --body)')
    parser.add_argument('--valid-data', type=str, default=None,
                       help='validation dataset (overrides --body)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)

    # Model architecture
    parser.add_argument('--model-type', type=str, default='siren',
                       choices=['ffnn', 'siren'])
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[32, 32, 32])
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'sigmoid'])
    parser.add_argument('--hidden-features', type=int, default=32)
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--omega-0', type=float, default=30.0)

    # Device and checkpointing
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='models/runs')
    parser.add_argument('--save-freq', type=int, default=10)

    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()

    train_path = args.train_data or en.published_dataset_path(args.body, 'train')
    valid_path = args.valid_data or en.published_dataset_path(args.body, 'valid')
    train_data = en.load_dataset(train_path)
    valid_data = en.load_dataset(valid_path)
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {valid_data.shape}")

    # Count shadow vs light points
    shadow = (train_data[:, -1] < 0).sum()
    light = (train_data[:, -1] > 0).sum()
    print(f"Shadow: {shadow:,}, Light: {light:,}")

    config = en.TrainingConfig(
        model_type=args.model_type,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        omega_0=args.omega_0,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_freq
    )
    en.train(train_data, valid_data, config)
    print(f"Checkpoints and training history saved to {config.checkpoint_dir}")


if __name__ == '__main__':
    main()
