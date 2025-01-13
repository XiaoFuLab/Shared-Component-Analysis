import torch
import numpy as np
import torch.nn as nn
from scipy.linalg import sqrtm
from torch.autograd import Variable

class CanonicalComponent(nn.Module):
    def __init__(self, X, D, view_dim, num_samples, Tensor):
        super(CanonicalComponent, self).__init__()
        self.D = D
        self.ID = torch.eye(D).type(Tensor)

        ###### G part ######
        self.g = torch.Tensor( X.cpu() @ X.cpu().T/ num_samples  )
        self.G = Variable(self.g.type(Tensor))

        self.Q = nn.Parameter(torch.randn(D, view_dim))


    def id_loss(self):
        return torch.norm(self.Q @ self.G @ self.Q.T - self.ID)

    def get_Q_value(self):
        return self.Q

    def forward(self, x_j):
        # s = torch.matmul(self.z.T, x_j.T)
        s = torch.matmul(self.Q, x_j.T)
        return s