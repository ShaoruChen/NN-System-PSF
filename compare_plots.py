import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from pympc.geometry.polyhedron import Polyhedron

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nx, nu = 2, 1
theta_max, theta_dot_max = 180.0, np.rad2deg(4.0)
A_x = torch.cat((torch.eye(nx), -torch.eye(nx))).to(device)
b_x = torch.tensor([theta_max, theta_dot_max, theta_max, theta_dot_max]).to(device)
X = Polyhedron(A_x.detach().cpu().numpy(), b_x.detach().cpu().numpy())

plt.rcParams.update({'font.size': 23})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

(fig, axes) = plt.subplots(nrows=2, ncols=2)
ax_big = fig.add_subplot(111, frameon=False)
ax_big.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.subplots_adjust(top=0.8, bottom=0.12)

date = 'Mar23'

for test_case_num in range(1, 5):
    if test_case_num == 1:
        data_path       = "data/pendulum_"+date+"_ilqr_testcase1.npy"
        with open(data_path, 'rb') as f:
            dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_relu_testcase1.npy"
        with open(data_path, 'rb') as f:
            relu_dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_safefilter_testcase1.npy"
        with open(data_path, 'rb') as f:
            filter_dat = np.load(f)

        data_path = "data/pendulum_"+date+"_relu_safefilter_testcase1.npy"
        with open(data_path, 'rb') as f:
            relufilter_dat = np.load(f)

        ax = axes[0, 0]
    elif test_case_num == 2:
        data_path       = "data/pendulum_"+date+"_ilqr_testcase2.npy"
        with open(data_path, 'rb') as f:
            dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_relu_testcase2.npy"
        with open(data_path, 'rb') as f:
            relu_dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_safefilter_testcase2.npy"
        with open(data_path, 'rb') as f:
            filter_dat = np.load(f)

        data_path = "data/pendulum_"+date+"_relu_safefilter_testcase2.npy"
        with open(data_path, 'rb') as f:
            relufilter_dat = np.load(f)

        ax = axes[0, 1]
    elif test_case_num == 3:
        data_path       = "data/pendulum_"+date+"_ilqr_testcase3.npy"
        with open(data_path, 'rb') as f:
            dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_relu_testcase3.npy"
        with open(data_path, 'rb') as f:
            relu_dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_safefilter_testcase3.npy"
        with open(data_path, 'rb') as f:
            filter_dat = np.load(f)

        data_path = "data/pendulum_"+date+"_relu_safefilter_testcase3.npy"
        with open(data_path, 'rb') as f:
            relufilter_dat = np.load(f)

        ax = axes[1, 0]
    elif test_case_num == 4:
        data_path       = "data/pendulum_"+date+"_ilqr_testcase4.npy"
        with open(data_path, 'rb') as f:
            dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_relu_testcase4.npy"
        with open(data_path, 'rb') as f:
            relu_dat = np.load(f)

        data_path       = "data/pendulum_"+date+"_safefilter_testcase4.npy"
        with open(data_path, 'rb') as f:
            filter_dat = np.load(f)

        data_path = "data/pendulum_"+date+"_relu_safefilter_testcase4.npy"
        with open(data_path, 'rb') as f:
            relufilter_dat = np.load(f)

        ax = axes[1, 1]

    ilqr_cnt = 0
    relu_cnt = 0
    filter_cnt = 0
    relufilter_cnt = 0
    for idx in range(dat.shape[0]):
        if X.contains(np.rad2deg(dat[idx, 1:3])) is not True:
            ilqr_cnt = ilqr_cnt + 1
    for idx in range(relu_dat.shape[0]):
        if X.contains(np.rad2deg(relu_dat[idx, 1:3])) is not True:
            relu_cnt = relu_cnt + 1
    for idx in range(filter_dat.shape[0]):
        if X.contains(np.rad2deg(filter_dat[idx, 1:3])) is not True:
            filter_cnt = filter_cnt + 1
    for idx in range(relufilter_dat.shape[0]):
        if X.contains(np.rad2deg(relufilter_dat[idx, 1:3])) is not True:
            relufilter_cnt = relufilter_cnt + 1
    print('Test case', test_case_num, ': violation %:', ilqr_cnt/dat.shape[0]*100, relu_cnt/relu_dat.shape[0]*100, filter_cnt/filter_dat.shape[0]*100, relufilter_cnt/relufilter_dat.shape[0]*100)

    ax.plot(np.rad2deg(dat[0, 1]), np.rad2deg(dat[0, 2]), 'o', linewidth=1, color='lime', label='Start')
    ax.plot(np.rad2deg(dat[-1, 1]), np.rad2deg(dat[-1, 2]), 'o', linewidth=1, color='magenta', label='End')
    step_temp = 2 * theta_dot_max / dat[:, 2].shape[0]
    step_temp1 = (2 * theta_dot_max + step_temp) / dat[:, 2].shape[0]
    ax.plot(theta_max*np.ones_like(dat[:, 1]), np.arange(start=-theta_dot_max, stop=theta_dot_max+step_temp, step=step_temp1), color='k', linestyle='--', linewidth=2, label='Constraints')
    ax.plot(-theta_max*np.ones_like(dat[:, 1]), np.arange(start=-theta_dot_max, stop=theta_dot_max+step_temp, step=step_temp1), color='k', linestyle='--', linewidth=2)
    step_temp = 2*theta_max/dat[:, 1].shape[0]
    step_temp1 = (2*theta_max+step_temp)/dat[:, 1].shape[0]
    ax.plot(np.arange(start=-theta_max, stop=theta_max+step_temp, step=step_temp1), theta_dot_max*np.ones_like(dat[:, 2]), color='k', linestyle='--', linewidth=2)
    ax.plot(np.arange(start=-theta_max, stop=theta_max+step_temp, step=step_temp1), -theta_dot_max*np.ones_like(dat[:, 2]), color='k', linestyle='--', linewidth=2)

    ax.plot(np.rad2deg(dat[:, 1]), np.rad2deg(dat[:, 2]), '-', linewidth=2, color='slateblue', label='iLQR')
    ax.plot(np.rad2deg(relu_dat[:, 1]), np.rad2deg(relu_dat[:, 2]), '-', linewidth=2, color='indianred', label='Soft-constrained iLQR (SC-iLQR)')
    ax.plot(np.rad2deg(filter_dat[:, 1]), np.rad2deg(filter_dat[:, 2]), '-', linewidth=2, color='mediumseagreen', label='Safe-filtered iLQR')
    ax.plot(np.rad2deg(relufilter_dat[:, 1]), np.rad2deg(relufilter_dat[:, 2]), '-', linewidth=2, color='sandybrown', label='Safe-filtered SC-iLQR')
    ax.set_ylim([np.rad2deg(-6.9), np.rad2deg(6.9)])
    ax.set_xlim([np.rad2deg(-3.3), np.rad2deg(3.3)])
    if test_case_num == 3 or test_case_num == 4:
        ax.set_xlabel('$\\theta$ [deg]')
    if test_case_num == 1 or test_case_num == 3:
        ax.set_ylabel('$\\dot{\\theta}$ [deg/s]')
    ax.grid('major')
handles, labels = ax.get_legend_handles_labels()
ax_big.legend(handles, labels, fontsize=20, bbox_to_anchor=(0., 1.13, 1.0, 0.05), loc='center', ncol=3, mode="expand", borderaxespad=0.)
# ax_big.set_xlabel('$\\theta$ [deg]')
# ax_big.set_ylabel('$\\dot{\\theta}$ [deg/s]')
plt.show()