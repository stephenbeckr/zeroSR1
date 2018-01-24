# ZeroSR1 GroupLasso Experiment

The code is written in Python and uses a simple C-interface for solving the rank-1 proximal mapping more efficiently. The file `test_groupLasso.py` reproduces the code for the GroupLasso Experiment from the 2018 paper.

# Installation
* Go to the folder `clib` and compile `mymath.cpp` using the `Makefile` that is provided in that folder.
* Then, you can run `python test_groupLasso.py` from the folder `zeroSR1/paperExperiments/groupLasso/`.

# Problem
The optimization problem is generated and solved in `test_groupLasso.py` using several methods.

## Usage
In order to measure the error to the optimal value, set the flag `compute_optimal_value = True`, which runs (by default) FISTA with 50000 iterations and writes the optimal value to the file `data_group_lasso.npy`. Once this run finished, set `compute_optimal_value = False` and evaluate the implemented algorithms.

## Implemented Algorithms
* Forward-Backward Splitting
* FISTA
* Zero SR1 Proximal Quasi-Newton (with rank-1 prox implemented in C)
* Monotone Fast Zero SR1 Proximal Quasi-Newton
* Tseng Fast Zero SR1 Proximal Quasi-Newton
* Sparse Reconstruction by Separable Approximation

Rank-1 proximal mappings are implemented in C (see folder `clib`).

## Parameters
For the parameters, we refer to `test_groupLasso.py` and the implementations in the folder `Algorithms`.



