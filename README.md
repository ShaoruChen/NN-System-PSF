# Safety Filter Design for Neural Network Systems 
This repository implements a [predictive safety filter (PSF)](https://www.sciencedirect.com/science/article/abs/pii/S0005109821001175) for neural network systems using convex optimization. In particular, we first use [NN verification](https://github.com/Verified-Intelligence/auto_LiRPA) to extract sound local abstraction of the NN dynamics and then apply [robust linear model predictive control](https://github.com/ShaoruChen/Polytopic-SLSMPC) to filter safe control inputs. This method is presented in the paper 

[Safety Filter Design for Neural Network Systems via Convex Optimization](https://arxiv.org/pdf/2308.08086.pdf) \
Shaoru Chen*, Kong Yao Chee*, Nikolai Matni, M. Ani Hsieh, George J. Pappas (* co-first authors)\
Conference on Decision and Control, 2023


## Installation
The following installation has been verified. To run codes on GPU, please install pytorch with cuda enabled. 

```
conda create -n psf python=3.10
conda activate psf

# Example: install pytorch on macOS. 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install auto-lirpa
conda install -c conda-forge cvxpy
pip install -r requirements.txt
```

The [Mosek](https://www.mosek.com/products/academic-licenses/) optimizer is applied and can be installed by 
```
pip install Mosek
```
Other optimizers for solving the convex program can be used too. 

An example of applying the predictive safety filter on a pendulum system is given by running `python main.py`. Run `python compare_plots.py` to recover the figures in the [paper](https://arxiv.org/pdf/2308.08086.pdf).


## Problem Formulation
We consider designing a predictive safety filter for an uncertain NN dynamical system by solving the following **constrained, robust optimal control problem**:

<img src="https://github.com/ShaoruChen/web-materials/blob/main/PSF_CDC_23/psf_formulation.png" width=360, height=160> 

where $\pi(\cdot)$ denotes a reference policy, $Ax + Bu$ denotes the linear component of the dynamics, $f(x, u)$ is a **neural network** capturing the nonlinear dynamics, and $w$ models the uncertainty effects. The PSF aims to find control inputs that make the uncertain NN dynamics safe (in terms of constraint satisfaction) over a finite horizon. 

### Challenges
Solving the above PSF problem has several challenges:
1. To reduce conservatism, we need to optimize over feedback policies $\mu_t(\cdot)$ such that $u_t = \mu_t(\cdot)$ instead of open-loop control inputs $u_t$ directly. How to parameterize the feedback policy $\mu_t(\cdot)$ and solve the PSF problem in a tractable and numerically efficient manner is an open problem.
2. The size (i.e., depth and width) of the NN dynamics $f(x, u)$ impose significant numerically challenges for optimization. We want our method to scale to large NNs such that we impose little restriction on the up-stream task of learning the NN dynamics.
3. We want to obtain interpretable and reliable safety certificates that are not prone to numerical issues.

### Method
We first over-approximate the nonlinear NN dynamics $f(x, u)$ by **uncertain linear dynamics** and then solve a robust linear MPC problem. The main algorithm is summarized below:

<img src="https://github.com/ShaoruChen/web-materials/blob/main/PSF_CDC_23/psf_algorithm.png" width=400, height=400> 

We use [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) for NN dynamics over-approximation and [SLS MPC](https://github.com/ShaoruChen/Polytopic-SLSMPC) to solve the robust linear MPC problem due to its tightness. 

### Main Features
Our method has the following benefits. 
1. In the end, we only need to solve a **convex quadratic program (QP)** as the PSF. Importantly, the complexity of the QP is **independent of the size of the NN dynamics**.
2. A numerical safety certificate is given by the QP and interpretable. Empirically, we demonstrated that the safety certificate gives a useful criterion to optimize to improve the safety of the closed-loop system.

### Example
On a pendulum system example, we demonstrated that the proposed PSF significantly improved the safety of the iLQR-based reference controller. See our paper for details. 

<img src="https://github.com/ShaoruChen/web-materials/blob/main/PSF_CDC_23/psf_fig.png" width=600, height=400> 



## Third-party dependence
We use [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) to extract bounds on the NN dynamics locally. 

We use [pympc](https://github.com/TobiaMarcucci/pympc/tree/master) mainly for operations on polyhedra.

We use [mpc.pytorch](https://github.com/locuslab/mpc.pytorch/tree/master) to implement iLQR and constrained iLQR in [this folder](https://github.com/ShaoruChen/NN-System-PSF/tree/main/source/ilqr).
