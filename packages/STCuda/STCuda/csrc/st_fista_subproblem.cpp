// st_fista_subproblem.cpp
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2023-2026 University Corporation for Atmospheric Research
// All rights reserved.

#include <Python.h>

#include "st_fista_kernel.h"

extern "C" {
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

namespace STCuda
{

at::Tensor st_fista_subproblem(
    const at::Tensor& b,
    const at::Tensor& lam1,
    const at::Tensor& lb,
    const at::Tensor& ub)
{
    TORCH_CHECK(b.is_cuda(),    "b must be GPU");
    TORCH_CHECK(lb.is_cuda(),   "lb must be GPU");
    TORCH_CHECK(ub.is_cuda(),   "ub must be GPU");

    // Weighted TV: lam1 is now a (scalar-or-per-pixel) field consumed on the device by the
    // FGP kernel, so it must live on b's device/dtype. (The original scalar code read lam1
    // host-side via .item() and so moved it to CPU here -- that .cpu() now strands the field
    // on the host and the kernel dereferences a host pointer.)
    return fista_launch(b, lam1.to(b.options()), lb, ub);
}

TORCH_LIBRARY(STCuda, m) {
    m.def("st_fista_subproblem(Tensor b, Tensor lam1, Tensor lb, Tensor ub) -> Tensor");
}

TORCH_LIBRARY_IMPL(STCuda, CUDA, m) {
    m.impl("st_fista_subproblem", &st_fista_subproblem);
}

} // namespace STCuda
