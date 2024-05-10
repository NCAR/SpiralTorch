# fista_cuf.py
#
# Author: Adam Karboski <karboski@ucar.edu>
#
# Copyright © 2023 University Corporation for Atmospheric Research
# All rights reserved.

import torch
from torch.utils.cpp_extension import load

import pathlib

dir = str(pathlib.Path(__file__).parent.resolve())
print("fista_cuf directory")
print(dir)
f_cu = load('st_fista_subproblem', [dir + '/st_fista_cuf.cpp', dir + '/st_fista_cuf.cu'], verbose=True)

def solve_FISTA_subproblem_kernel(b:torch.Tensor,lam1:torch.Tensor,
                                  lb:torch.Tensor,ub:torch.Tensor)->torch.Tensor:

    """
    Solves optimization subproblem by using FISTA [1].
    For most MLE applications
    b = x_k - 1/alpha * grad(f)
        where x_k is the current state vector, 1/alpha is the step size,
        f - is the error function so grad(f) is the gradient of f with respect to x

    lam - TV penalty function is typically
        lam = 1/alpha or tau/alpha

    returns x - the solution to the subproblem

    Solving the sub problem should be done separately for each separable variable.
        for example, this function should be run once for backscatter coefficient, lidar ratio and depolarization each.
        For non-TV variables (gain, deadtime), don't use this function call.  Just run steepest descent:
        x_{k+1} = x_k - 1/alpha * grad(x)

    This function expects rectangular arrays.  Mapping non-rectangular spaces (e.g. altitude varying range data)
    needs to happen prior to calling this function.
    """

    return f_cu.st_problem(b, lam1.cpu(), lb, ub);
