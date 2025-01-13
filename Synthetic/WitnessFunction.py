import torch
import numpy as np
import torch.nn as nn



# class WitnessFunction(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(WitnessFunction, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(input_size, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, output_size),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         feature = self.model(x)
#         return feature


class WitnessFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(WitnessFunction, self).__init__()

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

