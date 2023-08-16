# Safety Filter Design for Neural Network Systems 
This repository implements a [predictive safety filter](https://www.sciencedirect.com/science/article/abs/pii/S0005109821001175) for neural network systems using convex optimization. In particular, we first use [NN verification](https://github.com/Verified-Intelligence/auto_LiRPA) to extract sound local abstraction of the NN dynamics and then apply [robust linear model predictive control](https://github.com/ShaoruChen/Polytopic-SLSMPC) to filter safe control inputs. This method is presented in the paper 

Safety Filter Design for Neural Network Systems via Convex Optimization\
Shaoru Chen*, Kong Yao Chee*, Nikolai Matni, M. Ani Hsieh, George J. Pappas (* co-first authors)\
Conference on Decision and Control, 2023 (To appear)

## Third-party dependence
We use [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) to extract bounds on the NN dynamics locally. Some changes have been made in order to extract the local linear bounds in [this folder](https://github.com/ShaoruChen/NN-System-PSF/tree/main/source/auto_LiRPA). 

We use [pympc](https://github.com/TobiaMarcucci/pympc/tree/master) mainly for operations on polyhedra.

We use [mpc.pytorch](https://github.com/locuslab/mpc.pytorch/tree/master) to implement iLQR and constrained iLQR in [this folder](https://github.com/ShaoruChen/NN-System-PSF/tree/main/source/ilqr).
