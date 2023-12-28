# Safety Filter Design for Neural Network Systems 
This repository implements a [predictive safety filter](https://www.sciencedirect.com/science/article/abs/pii/S0005109821001175) for neural network systems using convex optimization. In particular, we first use [NN verification](https://github.com/Verified-Intelligence/auto_LiRPA) to extract sound local abstraction of the NN dynamics and then apply [robust linear model predictive control](https://github.com/ShaoruChen/Polytopic-SLSMPC) to filter safe control inputs. This method is presented in the paper 

Safety Filter Design for Neural Network Systems via Convex Optimization\
Shaoru Chen*, Kong Yao Chee*, Nikolai Matni, M. Ani Hsieh, George J. Pappas (* co-first authors)\
Conference on Decision and Control, 2023 (To appear)

## Installation
Create a conda environment using the following recommended options: 

```
conda create -n psf python=3.10

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


## Third-party dependence
We use [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) to extract bounds on the NN dynamics locally. 

We use [pympc](https://github.com/TobiaMarcucci/pympc/tree/master) mainly for operations on polyhedra.

We use [mpc.pytorch](https://github.com/locuslab/mpc.pytorch/tree/master) to implement iLQR and constrained iLQR in [this folder](https://github.com/ShaoruChen/NN-System-PSF/tree/main/source/ilqr).
