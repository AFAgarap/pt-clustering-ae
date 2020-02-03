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
        self.enc_hidden_layer_1 = nn.Linear(kwargs["input_shape"], 500)
        self.enc_hidden_layer_2 = nn.Linear(500, 500)
        self.enc_hidden_layer_3 = nn.Linear(500, 2000)
        self.code_layer = nn.Linear(2000, kwargs["code_dim"])

        self.dec_hidden_layer_1 = nn.Linear(kwargs["code_dim"], 2000)
        self.dec_hidden_layer_2 = nn.Linear(2000, 500)
        self.dec_hidden_layer_3 = nn.Linear(500, 500)
        self.reconstruction_layer = nn.Linear(500, kwargs["input_shape"])

    def forward(self, features):
        activation = self.enc_hidden_layer_1(features)
        activation = torch.relu(activation)
        activation = self.enc_hidden_layer_2(activation)
        activation = torch.relu(activation)
        activation = self.enc_hidden_layer_3(activation)
        activation = torch.relu(activation)
        activation = self.code_layer(activation)
        code = torch.sigmoid(activation)

        activation = self.dec_hidden_layer_1(code)
        activation = torch.relu(activation)
        activation = self.dec_hidden_layer_2(activation)
        activation = torch.relu(activation)
        activation = self.dec_hidden_layer_3(activation)
        activation = torch.relu(activation)
        activation = self.reconstruction_layer(activation)
        reconstruction = torch.sigmoid(activation)
        return reconstruction
