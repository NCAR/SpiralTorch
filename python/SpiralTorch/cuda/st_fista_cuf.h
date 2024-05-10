// fista_cuf.h
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2023 University Corporation for Atmospheric Research
// All rights reserved.

#include <torch/extension.h>

torch::Tensor fista_launch(
    torch::Tensor b,
    torch::Tensor lam1,
    torch::Tensor lb,
    torch::Tensor ub
  );