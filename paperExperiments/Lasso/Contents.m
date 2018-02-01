% FIG2_LASSO
%    Recreates Fig 6.1 from https://arxiv.org/pdf/1801.08691.pdf
% 
% Main Files
%   runTestsForPaper     - Script to run all the tests
%
% Helper Files
%   zeroSR1              - [xk,nit, errStruct, outOpts] = zeroSR1(f,g,proj,opts)
%   zeroSR1_noLinesearch - Solves smooth + nonsmooth/constrained optimization problems
%   fminunc_wrapper      - wrapper for objective and gradient
%   proj_box_weighted    - Projection onto box constraints
%   prox_l1_rank1        - Prox of l1 with diagonal + rank-1 metric
%   proj_Rplus_weighted  - Projection onto x>=0 with diagonal + rank-1
%   cummin               - Cumulative minimum
%
% Feb 1 2018