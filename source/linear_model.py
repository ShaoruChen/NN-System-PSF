import numpy as np
from torch import nn
import torch
from scipy.signal import cont2discrete


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, y):
        return self.net(y)


class LinearModel(nn.Module):
    def __init__(self, params, dt=0.05):
        super(LinearModel, self).__init__()

        self.gravity    = params[0]
        self.mass       = params[1]
        self.length     = params[2]

        self.n_state = 2
        self.n_ctrl = 1
        self.lower, self.upper = -100., 100.
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        A               = np.zeros((2, 2))
        A[0, 1]         = 1.0
        A[1, 0]         = 3.0 * self.gravity / (2 * self.length)

        B               = np.zeros((2, 1))
        B[1, 0]         = 3.0 / (self.mass * self.length * self.length)

        Ad, Bd, Cd, Dd, dt  = cont2discrete((A, B, np.eye(2), 0), dt, method='zoh')
        self.Ad             = Ad
        self.Bd             = Bd
        self.dt             = dt

        self.nn             = MLP()
        model_path          = "./data/pendulum_model_Oct23_1.pth"
        checkpoint          = torch.load(model_path, map_location=torch.device('cpu'))
        self.nn.load_state_dict(checkpoint['state_dict'])

    def forward(self, x, u):
        squeeze = x.ndimension() == 1
        if squeeze:
            x = x.unsqueeze(0)
        states  = x
        force   = u

        model   = states @ torch.Tensor.float(torch.from_numpy(self.Ad)).T + force @ torch.Tensor.float(torch.from_numpy(self.Bd)).T
        y       = torch.cat((x, u), dim=1)
        nn_output = self.nn.forward(y)

        return model + nn_output

