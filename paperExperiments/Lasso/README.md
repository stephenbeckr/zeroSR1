# zeroSR1: Lasso Experiments

This folder contains the Matlab code to run the Lasso experiments.
The version of code used here may be slightly different than the updated algorithms in the main repository.

Some third-party packages (not provided, though we list the URLs) are required if you want to compare with the other solvers mentioned in the paper.

In the code, "test 4" is Fig 6.1 (left) from our [2018 paper](https://arxiv.org/pdf/1801.08691.pdf) (similar to Fig 1.a from our [2012 paper](https://arxiv.org/pdf/1206.1156.pdf))

Similarly, "test 6" is Fig 6.1 (right) from our [2018 paper](https://arxiv.org/pdf/1801.08691.pdf) (similar to Fig 1.b from our [2012 paper](https://arxiv.org/pdf/1206.1156.pdf))

## Third party packages
If you install these, make sure to add them to the Matlab path. You can follow the example `addpath` commands that we used.

### L-BFGS-B
We wrote our own Matlab wrapper for this (using the L-BFGS-B 3.0 Fortran
code). You can download it from: https://github.com/stephenbeckr/L-BFGS-B-C

Unpack it somewhere and run `lbfgsb_C/Matlab/compile_mex.m`

### ASA
See http://users.clas.ufl.edu/hager/papers/Software/

as of 2013, they have [ver 3.0](http://users.clas.ufl.edu/hager/papers/CG/Archive/ASA_CG-3.0.tar.gz) but their older [ver 2.2](http://users.clas.ufl.edu/hager/papers/CG/Archive/ASA_CG-2.2.tar.gz) is still online.

You also need the Matlab interface; we wrote this ourself, and it can be downloaded from [Mathworks file exchange no 35814](https://www.mathworks.com/matlabcentral/fileexchange/35814-mex-interface-for-bound-constrained-optimization-via-asa) (it will also download the main C source code for you)

If you download the Matlab interface, run the `test_ASA.m` script and it will downlaoad the ASA code that it needs.

### CGIST
Get CGIST from their [website](http://tag7.web.rice.edu/CGIST.html) or [direct link to .zip file](http://tag7.web.rice.edu/CGIST_files/cgist.zip).

### FPC
Get FPC AS from [their website](http://www.caam.rice.edu/~optimization/L1/FPC_AS/request-for-downloading-fpc_as.html)

### L1General package, with PSSas and OWL
 Get the L1General2 code from [Mark Schmidt's software website](https://www.cs.ubc.ca/~schmidtm/Software/thesis.html) or [direct link to thesis.zip](https://www.cs.ubc.ca/~schmidtm/Software/thesis.zip).

Note: you need to compile mex files for this (for the lbfgs subroutine)
For compilation, try: `minFunc/mexAll.m`

We noticed that line 13 in `lbfgsC.c` declared `int nVars,nSteps,lhs_dims[2];` and for us, this threw a warning at compile-time and an error at run-time. One fix is to remove the `lhs_dims[2]` from that line and instead add a new line with: `size_t lhs_dims[2];`

## Output

Running test4 should give something like this:

![Test 4 results](test4.png?raw=true)

Running test5 should give something like this:

![Test 5 results](test5.png?raw=true)

## Authors
The authors are Stephen Becker, Jalal Fadili and Peter Ochs.

This README from Feb 1 2018. Thanks to https://stackedit.io/app for editing markup
