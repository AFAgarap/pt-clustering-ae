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
