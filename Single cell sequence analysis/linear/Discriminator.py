import torch
import numpy as np
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size, classes, num_layers=3, hidden_units = 256):
        super(Classifier, self).__init__()

        
        if num_layers > 1:
            layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())
        else:
            layers = [nn.Linear(input_size, classes)]
            
        if num_layers > 1:
            layers.append(nn.Linear(hidden_units, classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.model(x)
        return feature


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.model(x)
        return feature


