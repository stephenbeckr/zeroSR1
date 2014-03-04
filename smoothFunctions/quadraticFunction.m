function [f,g,h] = quadraticFunction(x,Q,c,errFcn,extraFcn, constant)
% f = quadraticFunction(x,Q,c, errFcn,extraFcn,constant)
%   returns the objective function 'f'
%   to f(x) = .5<x,Qx> - <c,x> + constant
% [f,g,h] = ...
%   return the gradient and Hessian as well
%
%   "Q" can be a matrix (and it should be Hermitian positive semi-definite)
%   or it can be a function handle to compute the matrix-vector product
%
% [fHist,errHist] = quadraticFunction()
%       will return the function history
%       (and error history as well, if errFcn was provided)
%       and reset the history to zero.
%   "fHist" is a record of f + extraFcn
%   (this is intended to be used where extraFcn is the non-smooth term "h")
%
% This function is (almost*) mathematically (not computationally) equivalent
%   to normSquaredFunction( x, A, b ) where
%   Q = A'*A and c = A'*b.
%   (*almost equivalent since there is a constant value difference in 
%    the objective function)
%
% The Lipschitz constant of the gradient is the spectral norm of Q, i.e., norm(Q)
%
% Feb 19 2013, Stephen Becker, stephen.beckr@gmail.com
%
% See also normSquaredFunction.m

persistent errHist fcnHist nCalls
if nargin == 0
   f = fcnHist(1:nCalls);
   g = errHist(1:nCalls);
   fcnHist = [];
   errHist = [];
   nCalls  = 0;
   return;
end
if isempty( fcnHist )
    [errHist,fcnHist] = deal( zeros(100,1) );
end


% fcnSimple   = @(w) w'*(Q*w)/2 - c'*w;
% gradSimple  = @(w) Q*w - c; % 
if isa(Q,'function_handle')
    Qx = Q(x);
else
    Qx = Q*x;
end
f   = (x'*Qx)/2 - c'*x;
if nargin >= 6 && ~isempty(constant)
    f   = f + constant;
end

% Record this:
nCalls = nCalls + 1;
if length( errHist ) < nCalls
    % allocate more memory
    errHist(end:2*end) = 0;
    fcnHist(end:2*end) = 0;
end
fcnHist(nCalls) = f;
if nargin >= 5 && ~isempty(extraFcn)
    % this is used when we want to record the objective function
    % for something non-smooth, and this routine is used only for the smooth
    % part. So for recording purposes, add in the nonsmooth part
    % But do NOT return it as a function value or it will mess up the
    % optimization algorithm.
    fcnHist(nCalls) = f + extraFcn(x);
end

if nargin > 2 && nargout > 1
%     g = G(x);
    g = Qx - c;
end
if nargout > 2
    if isa(Q,'function_handle')
        error('Function is only known implicitly, so cannot provide Hessian easily');
    end
    h = Q;
%     h = H(x);
end

% and if error is requested...
if nargin >= 4 && ~isempty( errFcn)
    errHist(nCalls) = errFcn(x);
end