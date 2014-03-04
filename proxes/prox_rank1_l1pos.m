function varargout = prox_rank1_l1pos(varargin)   
% PROX_RANK1_L1POS returns the scaled proximity operator for the l1 norm 
%   with non-negativity constraints
%
%   x = prox_rank1_l1pos( x0, D, u )
%           where 
%           x = argmin_{x} h(x) + 1/2||x-x0||^2_{V}
%           and
%           V^{-1} = D + u*u'  (or diag(D) + u*u' if D is a vector)
%           "D" must be diagonal and positive. "u" can be any vector.
%
%   Here, h(x) = ||x||_1 + the indicator function of the set { x : x >= 0 }
%
%   There are also variants:
%   x = prox_rank1_l1pos( x0, D, u, lambda, linTerm, sigma, inverse)
%           returns
%           x = argmin_{x} h(lambda.*x) + 1/2||x-x0||^2_{V} + linTerm'*x
%           and
%           either V^{-1} = D + sigma*u*u' if "inverse" is true (default)
%           or     V      = D + sigma*u*u' if "inverse" is false
%           and in both cases, "sigma" is either +1 (default) or -1.
%           "lambda" should be non-zero
%
% Stephen Becker, Feb 26 2014, stephen.beckr@gmail.com
% Reference: "A quasi-Newton proximal splitting method" by S. Becker and J. Fadili
%   NIPS 2012, http://arxiv.org/abs/1206.1156
%
% See also prox_rank1_generic.m, prox_rank1_l1.m, proj_rank1_Rplus.m

prox            = @(x,t) max(0, x - t );
prox_brk_pts    = @(s) [s];
            
[varargout{1:nargout}] = prox_rank1_generic( prox, prox_brk_pts, varargin{:} );