function [f,g,h] = normSquaredFunction(x,A,At,b,c,errFcn,extraFcn, constant)
% f = normSquaredFunction(x,A,At,b,c,errFcn,extraFcn, constant)
%   returns the objective function 'f'
%   to f(x) = .5||Ax-b||_2^2 + c'*x + constant
% [f,g,h] = ...
%   return the gradient and Hessian as well
%
%   "A" can be a matrix (in which case set At=[], since it is ignored)
%   or it can be a function handle to compute the matrix-vector product
%   (in which case "At" should be a function handle to compute
%    the transposed-matrix - vector product )
%
%   By default, b=0 and c=0. Set any inputs to [] to use default values.
%
% [fHist,errHist] = normSquaredFunction()
%       will return the function history
%       (and error history as well, if errFcn was provided)
%       and reset the history to zero.
%   "fHist" is a record of f + extraFcn
%   (this is intended to be used where extraFcn is the non-smooth term "h")
%
% This function is (almost*) mathematically (not computationally) equivalent
%   to quadraticFunction( x, Q, c ) where
%   Q = A'*A and c = A'*b.
%   (*almost equivalent since there is a constant value difference in 
%    the objective function; you can use "constant" to change this)
%
% The Lipschitz constant of the gradient is 
%   the squared spectral norm of A, i.e., norm(A)^2
%
%
% March 4 2014, Stephen Becker, stephen.beckr@gmail.com
%
% See also quadraticFunction.m

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

error(nargchk(2,8,nargin,'struct'));
if nargin < 4 || isempty(b), b = 0; end
if nargin >= 5 && ~isempty(c)
    cx  = dot(c(:),x(:) );
else
    cx  = 0;
    c   = 0;
end
if nargin < 8 || isempty(constant), constant = 0; end
if isa(A,'function_handle')
    Ax = A(x);
else
    Ax = A*x;
end
res = Ax - b;
f   = .5*norm(res(:))^2 + cx + constant;

% Record this:
nCalls = nCalls + 1;
if length( errHist ) < nCalls
    % allocate more memory
    errHist(end:2*end) = 0;
    fcnHist(end:2*end) = 0;
end
fcnHist(nCalls) = f;
if nargin >= 7 && ~isempty(extraFcn)
    % this is used when we want to record the objective function
    % for something non-smooth, and this routine is used only for the smooth
    % part. So for recording purposes, add in the nonsmooth part
    % But do NOT return it as a function value or it will mess up the
    % optimization algorithm.
    fcnHist(nCalls) = f + extraFcn(x);
end

if nargout > 1
    if isa(A,'function_handle')
        if isempty( At )
            error('If "A" is given implicitly, we need "At" to compute the gradient');
        end
        g   = At( res ) + c;
    else
        g   = A'*res + c;
    end
end
if nargout > 2
    if isa(A,'function_handle')
        error('Function is only known implicitly, so cannot provide Hessian easily');
    end
    h = A'*A;
end

% and if error is requested...
if nargin >= 6 && ~isempty( errFcn)
    errHist(nCalls) = errFcn(x);
end