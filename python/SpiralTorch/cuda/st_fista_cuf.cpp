// fista_cuf.cpp
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2023 University Corporation for Atmospheric Research
// All rights reserved.

#include "st_fista_cuf.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x)  AT_ASSERTM(!x.device().is_cuda(), #x " must be a CPU tensor")

torch::Tensor st_fista_subproblem(
    torch::Tensor b,
    torch::Tensor lam1,
    torch::Tensor lb,
    torch::Tensor ub
  )
{
    CHECK_CUDA(b);
    CHECK_CPU(lam1);
    CHECK_CUDA(lb);
    CHECK_CUDA(ub);

    return fista_launch(b, lam1, lb, ub);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("st_fista_subproblem", &st_fista_subproblem, "FISTA subproblem solver cuda kernel");
}
