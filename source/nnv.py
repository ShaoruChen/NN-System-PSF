import torch
import warnings
warnings.simplefilter("always")

import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# LiRPA based analysis
def output_Lp_bounds_LiRPA(nn_model, lb, ub, method = 'backward', need_linear_bounds = False, C = None):

    center = (lb + ub)/2
    radius = (ub - lb)/2

    model = BoundedModule(nn_model, center)
    ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
    # Make the input a BoundedTensor with perturbation
    my_input = BoundedTensor(center, ptb)
    # Compute LiRPA bounds
    if 'optimized' in method or 'alpha' in method:
        model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})

    # final = model._modules[model.final_name]
    # output = model.compute_bounds(x = (my_input,), method = method, return_A = True, needed_A_dict = {model.final_name: model.input_name})
    if need_linear_bounds:
        output = model.compute_bounds(x = (my_input,), C = C, method = method, return_A = True, needed_A_dict = {model.final_name: model.input_name})
        output_lb, output_ub, linear_bounds = output[0], output[1], output[2]

        # Fixme: not sure if there are cases when model.input_name contains more than one input nodes
        bounds = linear_bounds[model.final_name][model.input_name[0]]
        return output_lb, output_ub, bounds
    else:
        output = model.compute_bounds(x=(my_input,),C = C, method=method)
        output_lb, output_ub = output[0], output[1]
        return output_lb, output_ub, None


def instantiate_bounds(A, b, x0_lb, x0_ub, option = 'min'):
    # entrywise min/max Ax + b s.t. x0_lb <= x <= x0_ub
    x0 = (x0_lb + x0_ub)/2
    r = (x0_ub - x0_lb)/2
    if option == 'min':
        bd = x0@A.T + b - r@A.T.abs()
    elif option == 'max':
        bd = x0@A.T + b + r@A.T.abs()
    else:
        raise NotImplementedError

    return bd
