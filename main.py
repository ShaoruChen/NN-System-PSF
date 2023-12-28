import time
import torch
import numpy as np
from tqdm import tqdm

from pympc.geometry.polyhedron import Polyhedron
from ilqr.mpc import MPC, QuadCost, GradMethods
from linear_model import LinearModel
from base_system import BaseSystem
from safety_filter import PredictiveSafetyFilter

torch.manual_seed(1234)
np.random.seed(1234)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pendulum parameters
    params_true = torch.tensor((9.81, 0.75, 2.0))
    sampling_rate = 0.05
    dx = LinearModel(params_true, sampling_rate)
    nn_model = dx.nn.net.to(device)

    Ad, Bd = dx.Ad.astype('float32'), dx.Bd.astype('float32')
    A, B = torch.from_numpy(Ad).to(device), torch.from_numpy(Bd).to(device)
    nx, nu = B.shape
    sigma_w = 0.05

    dyn_system = BaseSystem(A, B, nn_model, device, sigma_w=sigma_w, augmented_nn_input=True)

    # state and input constraints
    theta_max, theta_dot_max = np.deg2rad(180.0), 4.0
    A_x = torch.cat((torch.eye(nx), -torch.eye(nx))).to(device)
    b_x = torch.tensor([theta_max, theta_dot_max, theta_max, theta_dot_max]).to(device)

    r_u = 100.0
    A_u = torch.cat((torch.eye(nu), -torch.eye(nu)))
    b_u = torch.cat((r_u * torch.ones(nu), r_u * torch.ones(nu)))

    dx.lower, dx.upper = -r_u, r_u

    # construct polyhedral state and input constraints
    X = Polyhedron(A_x.detach().cpu().numpy(), b_x.detach().cpu().numpy())
    U = Polyhedron(A_u.detach().cpu().numpy(), b_u.detach().cpu().numpy())

    # Simulation steps and nominal MPC horizon
    T, mpc_T = 41, 15

    # Test case number (choose 1 - 4)
    test_case_num = 1

    # Soft constraint option for iLQR: None (for nominal iLQR), 'ReLU' (for soft constrained iLQR)
    soft_constraint_option = None

    theta_goal_array = torch.zeros((T,))
    if test_case_num == 1:
        x0 = torch.tensor([1.0, -2.1])
        goal_weights = torch.Tensor((1., 0.1))
        ctrl_penalty = 0.0001
        theta_goal_array[:int((T - 1) / 2)] = np.deg2rad(120.0)
        theta_goal_array[int((T - 1) / 2):T] = np.deg2rad(-50.0)
    elif test_case_num == 2:
        x0 = torch.tensor([-1.5, -1.5])
        goal_weights = torch.Tensor((1., 0.1))
        ctrl_penalty = 0.0001
        theta_goal_array[:int((T - 1) / 2)] = np.deg2rad(-150.0)
        theta_goal_array[int((T - 1) / 2):T] = np.deg2rad(40.0)
    elif test_case_num == 3:
        x0 = torch.tensor([-1.5, -2.0])
        goal_weights = torch.Tensor((1., 1e-4))
        ctrl_penalty = 5e-5
        theta_goal_array[:int((T - 1) / 2)] = np.deg2rad(-100.0)
        theta_goal_array[int((T - 1) / 2):T] = np.deg2rad(-180.0)
    elif test_case_num == 4:
        x0 = torch.tensor([1.5, 1.0])
        goal_weights = torch.Tensor((1., 1e-4))
        ctrl_penalty = 1e-4
        theta_goal_array[:int((T - 1) / 2)] = np.deg2rad(100.0)
        theta_goal_array[int((T - 1) / 2):T] = np.deg2rad(180.0)
    else:  # default
        print('invalid test case, check test_case_num.')

    x       = x0.unsqueeze(0).repeat((1, 1))
    traj    = dyn_system.simulate_dynamics(mpc_T, x0)
    u_init  = None

    q = torch.cat((goal_weights, ctrl_penalty * torch.ones(dx.n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_T, 1, 1, 1)

    state_log = np.zeros((T, 2))
    control_log = np.zeros((T, 1))
    mpc_time = 0.0

    enable_safety_filter = True
    TR_slack_record = []
    constr_slack_record = []

    # closed-loop simulation
    for t in tqdm(range(T)):
        start_time = time.time()

        goal_state = torch.Tensor((theta_goal_array[t], 0.))
        px = -torch.sqrt(goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(dx.n_ctrl)))
        p = p.unsqueeze(0).repeat(mpc_T, 1, 1)

        nominal_states, nominal_actions, nominal_objs = MPC(
            dx.n_state, dx.n_ctrl, mpc_T,
            u_init=u_init,
            u_lower=dx.lower, u_upper=dx.upper,
            lqr_iter=5,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=False,
            n_batch=1,
            linesearch_decay=dx.linesearch_decay,
            max_linesearch_iter=dx.max_linesearch_iter,
            grad_method=GradMethods.AUTO_DIFF,
            eps=1e-2,
            state_con_A=torch.from_numpy(X.A),
            state_con_b=torch.from_numpy(X.b),
            soft_const_opt=soft_constraint_option,
            soft_const_multiplier=1e5,
        )(x, QuadCost(Q, p), dx)

        end_time = time.time()
        mpc_time += (end_time - start_time)

        if enable_safety_filter:
            iLQR_ol_traj = nominal_states[:, 0, :].detach().cpu().numpy()
            iLQR_ol_inputs = nominal_actions[:, 0, :].detach().cpu().numpy()

            nominal_x = torch.from_numpy(iLQR_ol_traj).to(device)
            nominal_u = torch.from_numpy(iLQR_ol_inputs).to(device)
            nominal_traj = {'x': nominal_x, 'u': nominal_u}

            # choose the horizon for the predictive safety filter
            # note that filter_horizon is different from mpc_T
            # mpc_T is used to generate a nominal trajectory as a reference while filter_hoirzon is used in the
            # predictive safety filter to provide guarantees

            filter_horizon = 5
            state_constr = [{'A': A_x, 'b': b_x}] * (filter_horizon + 1)
            input_constr = [{'A': A_u, 'b': b_u}] * filter_horizon
            safety_filter = PredictiveSafetyFilter(dyn_system, x.flatten(), filter_horizon, state_constr=state_constr, input_constr=input_constr)

            # choose the radius of the trust regions
            x_eps = nominal_traj['x'].abs().max() * 0.1
            u_eps = nominal_traj['u'].abs().max() * 0.1

            start_time = time.time()
            u0, u_ref, is_feasible, record, record_list = safety_filter.solve_safety_filter(ref_control=None, init=nominal_traj,
                                                                                    horizon=filter_horizon, x_eps=x_eps,
                                                                                    u_eps=u_eps, enlarge_multiplier=1.1)

            TR_slack_record.append(record['sol']['TR_slack_max'])
            constr_slack_record.append(record['sol']['constr_slack_max'])

            safety_filter_runtime = time.time() - start_time

            next_action = torch.from_numpy(u0).unsqueeze(0).to(device)
        else:
            next_action = nominal_actions[0]

        u_init      = torch.cat((nominal_actions[1:], torch.zeros(1, 1, dx.n_ctrl)), dim=0)
        u_init[-2]  = u_init[-3]

        angle = x[0][0].detach()
        angle_rate = x[0][1].detach()
        state_log[t, :] = np.array([angle.numpy(), angle_rate.numpy()])
        control_log[t] = np.array([next_action[0].detach().numpy()])

        x = dx(x, next_action)+2*(torch.rand(x.size()).to(device)-0.5)*sigma_w