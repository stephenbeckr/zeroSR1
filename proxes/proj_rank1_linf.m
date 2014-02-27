function varargout = proj_rank1_linf(varargin)   
% PROJ_RANK1_LINF returns the scaled proximity operator for l_infinity norm constraints
%
%   x = proj_rank1_linf( x0, D, u )
%           where 
%           x = argmin_{x} h(x) + 1/2||x-x0||^2_{V}
%           and
%           V^{-1} = D + u*u'  (or diag(D) + u*u' if D is a vector)
%           "D" must be diagonal and positive. "u" can be any vector.
%
%   Here, h(x) is the indicator function of the l_infinity ball, i.e.,
%       { x | norm(x,inf) <= 1 }
%       To scale the ball, just use the scaling parameter "lambda" (see below)
%
%   There are also variants:
%   x = proj_rank1_linf( x0, D, u, lambda, linTerm, sigma, inverse)
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
% See also prox_rank1_generic.m

prox            = @(x,t) sign(x).*min( 1, abs(x) );
prox_brk_pts    = @(s) [-ones(size(s)),ones(size(s))]; % since projection, scaling has no effect
            
[varargout{1:nargout}] = prox_rank1_generic( prox, prox_brk_pts, varargin{:} );