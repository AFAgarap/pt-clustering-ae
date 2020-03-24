# Improving k-Means Clustering Performance with Disentangled Internal Representations
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""PyTorch implementation of a vanilla Autoencoder"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=kwargs["input_shape"], out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=2000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2000, out_features=kwargs["code_dim"]),
            torch.nn.Sigmoid()
        ])
        self.decoder_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=kwargs["code_dim"], out_features=2000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2000, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=kwargs["input_shape"]),
            torch.nn.Sigmoid()
        ])

    def forward(self, features):
        activations = {}
        for index, encoder_layer in enumerate(self.encoder_layers):
            if index == 0:
                activations[index] = encoder_layer(features)
            else:
                activations[index] = encoder_layer(activations[index - 1])
        code = activations[len(activations) - 1]
        activations = {}
        for index, decoder_layer in enumerate(self.decoder_layers):
            if index == 0:
                activations[index] = decoder_layer(code)
            else:
                activations[index] = decoder_layer(activations[index - 1])
        reconstruction = activations[len(activations) - 1]
        return reconstruction

    def fit(self, data_loader, epochs):
        """
        Trains the autoencoder model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        """
        train_loss = []
        for epoch in range(epochs):
            epoch_loss = epoch_train(self, data_loader)
            train_loss.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} : mean loss = {train_loss[-1]:.6f}")
        self.train_loss = train_loss


def epoch_train(model, data_loader):
    """
    Trains a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_loader : torch.utils.dataloader.DataLoader
        The data loader object that consists of the data pipeline.

    Returns
    -------
    epoch_loss : float
        The epoch loss.
    """
    epoch_loss = 0
    for batch_features, batch_labels in data_loader:
        model.optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = model.criterion(outputs, batch_features)
        train_loss.backward()
        model.optimizer.step()
        epoch_loss += train_loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss


def train_step(
    model: nn.Module, optimizer: object, features: torch.Tensor, loss_fn: object
) -> torch.Tensor:
    """
    Trains a model for a single step.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : object
        The optimizer function to use.
    features : torch.Tensor
        The features to train on.
    loss_fn : object
        The loss function to optimize.

    Returns
    -------
    train_loss : torch.Tensor
        The training loss for a single step.
    """
    optimizer.zero_grad()
    outputs = model(features)
    train_loss = loss_fn(outputs, features)
    train_loss.backward()
    optimizer.step()
    return train_loss


def train(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    loss: object,
    optimizer: object,
) -> list:
    train_loss = []
    for epoch in range(epochs):
        epoch_loss = []
        for batch_features, batch_labels in data_loader:
            step_loss = train_loss(model, optimizer, batch_features, loss)
            epoch_loss.append(step_loss.item())
        epoch_loss = torch.mean(epoch_loss)
        train_loss.append(epoch_loss)
    return train_loss
