# fides-cpp - C++ implementation of the fides Python package for trust region optimization

## About

The fides-cpp library provides a C++ implementation of the 
[fides Python package](https://github.com/fides-dev/fides), that implements an
interior trust region reflective strategy for solving boundary-constrained
optimization problems, based on the papers
[ColemanLi1994](https://doi.org/10.1007/BF01582221) and
[ColemanLi1996](http://dx.doi.org/10.1137/0806023).
 
## Features

* Boundary constrained interior trust-region optimization
* Recursive, reflective and truncated constraint management
* Full and 2D subproblem solution solvers
* BFGS, DFP and SR1 Hessian approximations

## Requirements

* C++17 compiler
* CMake
* LAPACK library
* [Blaze](https://bitbucket.org/blaze-lib/blaze)
