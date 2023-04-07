import numpy as np
import torch
import torch.nn as nn
import control

class ResNetController(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.nx, self.nu = nx, nu

        self.net_1 = nn.Sequential(
            nn.Linear(nx, 64),
            nn.ReLU(),
            nn.Linear(64, nx),
        )

        self.net_2 = nn.Sequential(
            nn.Linear(nx, 32),
            nn.ReLU(),
            nn.Linear(32,nu)
        )

    def forward(self, x):
        identity = x
        x = self.net_1(x)
        x += identity

        identity = x
        x = self.net_2(x)
        return x

class BasePolicy(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.nx, self.nu = nx, nu

        # the base policy is parameterized as u = Kx + pi(x)
        # self.nn_policy = nn.Sequential(
        #                         nn.Linear(nx, 64),
        #                         nn.LeakyReLU(0.1),
        #                         # nn.Tanh(),
        #                         nn.Linear(64, 32),
        #                         nn.LeakyReLU(0.1),
        #                         # nn.Tanh(),
        #                         nn.Linear(32, nu)
        #                         )

        self.nn_policy = ResNetController(nx, nu)

        self.K = nn.Linear(nx, nu, bias = False)
        # for param in self.K.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.K(x) + self.nn_policy(x)

    def weights_init(self, A, B, Q = None, R = None):
        # initialize the linear feedback weight K as K = dlqr(A, B, Q, R)
        if Q is None:
            Q = np.eye(self.nx)

        if R is None:
            R = np.eye(self.nu)

        F, _, _ = control.dlqr(A, B, Q, R)
        self.K.weight.data = torch.from_numpy(-F.astype('float32'))

    def to(self, device):
        self.nn_policy = self.nn_policy.to(device)
        self.K = self.K.to(device)


