# zeroSR1 toolbox

The zeroSR1 toolbox implements the algorithm from 'A quasi-Newton proximal splitting method' by 
Stephen Becker and Jalal Fadili, which appeared in [NIPS 2012](http://nips.cc/). The paper is available at [arXiv 1206.1156](http://arxiv.org/abs/1206.1156).

(Update, January 2018, we have an extended paper [On Quasi-Newton Forward--Backward Splitting: Proximal Calculus and Convergence](https://arxiv.org/abs/1801.08691) by Stephen Becker, Jalal Fadili and Peter Ochs)

Briefly, the algorithm follows the standard proximal-gradient method, but allows a scaled prox. This enables us to use a limited-memory SR1 method (similar to L-BFGS).

The algorithm solves problems of the form min\_x f(x) + h(x) where f is differentiable (more precisely, with a Lipschitz gradient) and h is one of the following (see the paper):

Available "h" | Cost for input of size "n"
------------- | -------------
l1 norm | O( n log n)
non-negativity constraints | O( n log n)
l1 and non-negativity | O( n log n)
box constraints | O( n log n )
l\_infinity norm constraint | O( n log n )
[hinge loss](http://en.wikipedia.org/wiki/Hinge_loss) | O( n log n )

The algorithm compares favorably with other methods, including [L-BFGS-B](http://www.mathworks.com/matlabcentral/fileexchange/35104-lbfgsb-l-bfgs-b-mex-wrapper).

This toolbox currently implements in the following languages

* Matlab
* Octave

Further releases may target these languages:

* Python
* R
* C++

# Installation
For Matlab, there is no installation necessary. Every time you run a new Matlab session, run the `setup_zeroSR1.m` file and it will add the correct paths.

Run `tests/test_solver_simple.m` to see how to solve a typical problem

# Structure
In each folder, see the `Contents.m` file for more information
### Algorithms
This includes the zeroSR1 algorithm as well as implemenations of FISTA and other proximal-gradient methods

### Proxes
The scaled diagonal+ rank1 prox operators for various "g" functions

### SmoothFunctions
These are pre-made wrappers for the various smooth "f" functions. The files here with the `_splitting` suffix are intended for use with any method that requires forming the augmented variable "x\_aug = (x\_pos, x\_neg)". For example, this approach is used when using L-BFGS-B (which only allows box constraints, such as x\_pos >= 0,  x\_neg <= 0) to solve the LASSO problem.

### Utilities
Helper files

### Tests
Verify the algorithm and proxes are working correctly. This uses [CVX](http://cvxr.com/cvx) to verify; if this is not installed on your system, then it relies on precomputed solutions stored in a subdirectory.

### paperExperiments
Recreates the experiments in the 2018 paper

# Authors
The original authors are Stephen Becker, Jalal Fadili and Peter Ochs. Further contributions are welcome.

## Citing
This software is provided free of charge, but we request that if you use this for an academic paper, please cite the following work:

bibtex entry:

    @INPROCEEDINGS{quasiNewtonNIPS,
      author = {S. Becker and J. Fadili},
      title = {A quasi-{N}ewton proximal splitting method},
      booktitle = {Neural Information Processing Systems (NIPS)},
      year = {2012}
    }
