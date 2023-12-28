import torch
import torch.nn as nn
from base_policy import BasePolicy

class BaseSystem(nn.Module):
    def __init__(self, A, B, nn_residual, device, sigma_w = None, augmented_nn_input=True, with_nn=True):
        super().__init__()

        # nominal linear dynamics
        self.A, self.B = A, B
        self.nx = A.size(0)
        self.nu = B.size(1)

        # neural network residual dynamics
        self.nn_model = nn_residual
        for param in self.nn_model.parameters():
            param.requires_grad = False

        base_policy = BasePolicy(self.nx, self.nu)
        base_policy.weights_init(A.detach().cpu().numpy(), B.detach().cpu().numpy())
        base_policy.to(device)

        self.base_policy = base_policy

        # norm(w, inf) <= sigma_w
        self.sigma_w = sigma_w
        self.augmented_nn_input = augmented_nn_input
        self.with_nn = with_nn

    def forward(self, x, u, w = None):
        if w is None:
            # nominal dynamics without w
            if self.augmented_nn_input:
                x_plus = x @ self.A.T + u @ self.B.T + self.nn_model(torch.cat((x, u), dim=1))
            else:
                x_plus = x @ self.A.T + u @ self.B.T + self.nn_model(x)
        else:
            if self.augmented_nn_input:
                x_plus = x@self.A.T + u@self.B.T + self.nn_model(torch.cat((x, u), dim = 1)) + w
            else:
                x_plus = x@self.A.T + u@self.B.T + self.nn_model(x) + w
        return x_plus

    def simulate_nominal_dynamics(self, horizon, x0, u_seq = None):
        # simulate the nominal nonlinear dynamics x_+ = Ax + Bu + f(x, u) given initial state x0 and control sequence u_seq
        # if u_seq is None, generate the inputs using the base policy
        # x0: (nx) tensor; u_seq: (T, nu) tensor

        device = x0.device
        nx, nu = self.nx, self.nu

        traj = x0.unsqueeze(0)
        x = x0.unsqueeze(0)

        inputs = torch.zeros((horizon, nu))

        for i in range(horizon):
            if u_seq is not None:
                u = u_seq[i, :].unsqueeze(0).to(device)
            else:
                u = self.base_policy(x).to(device)

            inputs[i,:] = u[0]
            x_plus = self.forward_nominal(x, u)
            traj = torch.cat((traj, x_plus))

            x = x_plus

        simulated_traj = {'x': traj, 'u': inputs, 'w': None}
        return simulated_traj

    def simulate_dynamics(self, horizon, x0, u_seq=None, w_seq=None):
        # simulate the nonlinear dynamics x_+ = Ax + Bu + f(x, u) + w given initial state x0, control sequence u_seq, and disturbance sequence w_seq
        # if u_seq is None, generate the inputs using the base policy
        # if w_seq is None, generate the disturbances using random samples
        # x0: (nx) tensor; u_seq: (T, nu) tensor, w_seq: (T, nx) tensor

        device = x0.device
        nx, nu = self.nx, self.nu

        traj = x0.unsqueeze(0)
        x = x0.unsqueeze(0)

        inputs = torch.zeros((horizon, nu))
        disturbances = torch.zeros((horizon, nx))

        for i in range(horizon):
            if u_seq is not None:
                u = u_seq[i, :].unsqueeze(0).to(device)
            else:
                u = self.base_policy(x).to(device)

            if w_seq is not None:
                w = w_seq[i, :].unsqueeze(0).to(device)
            else:
                w = self.sigma_w*torch.rand(1, nx).to(device)

            inputs[i,:] = u[0]
            disturbances[i,:] = w[0]

            x_plus = self.forward(x, u, w)
            traj = torch.cat((traj, x_plus))

            x = x_plus

        simulated_traj = {'x': traj, 'u': inputs, 'w': disturbances}
        return simulated_traj

    def simulate_dynamics_SLS_MPC_policy(self, x0, K, h_vec, v_vec, w_seq = None):
        # simulate the nominal nonlinear dynamics x_+ = Ax + Bu + f(x, u) + w given initial state x0 and an LTV state
        # apply u = K(x - h_vec) + v_vec
        # x0: (nx) tensor, w_seq: (T, nx) tensor

        device = x0.device

        nu = self.nu
        nx = self.nx
        horizon = K.size(0)//nu - 1

        traj = x0.unsqueeze(0).to(device)
        inputs = torch.zeros((horizon, nu)).to(device)
        disturbances = torch.zeros((horizon, nx)).to(device)

        x = x0.unsqueeze(0)

        for i in range(horizon):
            K_i = K[i*nu:(i+1)*nu, :(i+1)*nx]
            u = K_i @ (traj.flatten() - h_vec[:(i + 1) * nx]) + v_vec[i*nu:(i + 1) * nu]
            u = u.unsqueeze(0)

            if w_seq is not None:
                w = w_seq[i,:].unsqueeze(0).to(device)
            else:
                w = self.sigma_w*torch.rand(1, nx).to(device)

            inputs[i,:] = u[0]
            disturbances[i,:] = w[0]

            x_plus = self.forward(x, u, w)

            traj = torch.cat((traj, x_plus))

            x = x_plus

        simulated_traj = {'x': traj, 'u': inputs, 'w': disturbances}

        return simulated_traj



