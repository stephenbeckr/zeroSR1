function [x,V] = solution_via_cvx(type,x0,d,u,lambda,offset,lwr,upr,pm,INV)
% x = solution_via_cvx(type,x0,d,u,lambda,offset,lwr,upr,pm)
%   returns the solution using CVX, to serve as a reference.
%   'type' can be one of 'l1', 'rplus' or 'box'
%   For 'box', specify lwr and upr bounds.
%
%   This computes the weighted prox_h^V(x0) where the function "h"
%   is specified by "type" (and perhaps scaled with lambda,
%   and/or linear term <offset,x> ), and
%
%   V^{-1} = diag(d) + pm*u*u'
%           where pm is +1 or -1 (default is +1)
%           i.e., V = diag(1./d) - pm*(u./d)*(u./d)'/( 1 + u'*diag(1./d)*u)
%           via Sherman-Morrison formula
%   or, in the form
%   x = solution_via_cvx(type,x0,d,u,lambda,offset,lwr,upr, INV)
%       if INV=false (default is true),
%       then
%       V = diag(d) + pm*u*u' (rather than this being inv(V) )
%
% In all cases, V must be positive definite
% [x,V] = solution_via_cvx(...)
%   also returns the matrix V
%
% Lambda is such that we really evaulate h(lambda.*x)
%   where lambda is a scalar or an array ofo the same size as x
%
% If CVX is not installed, or if this is called via octave
%   (new versions of CVX do not run on octave),
%   then the output is x=Inf.
%
% Stephen Becker, Feb 22 2014 stephen.beckr@gmail.com

% TODO: if CVX is not installed, read solution from a .mat file

if nargin < 10 || isempty(INV), INV = true; end
if nargin < 9 || isempty(pm), pm = +1; end
if nargin < 8 || isempty(upr), upr = []; end
if nargin < 7 || isempty(lwr), lwr = []; end
if nargin < 5 || isempty(lambda), lambda = 1; end
if nargin < 4 || isempty(u), u = 0; end
assert( pm==-1 | pm==+1 );
[R,L]   = deal(u);
n       = length(x0);
if nargin < 6 || isempty(offset), offset = zeros(n,1); end

% Vinv    = diag(d) + L*R';
% V       = inv(Vinv);
if INV % default
    Dinv    = diag(1./d);
    if all(u==0)
        V = Dinv;
    else
        V       = Dinv - pm*(Dinv*L)*(R'*Dinv)/( 1 + R'*Dinv*L );
    end
    if pm ==1
        % There is a chance that V is not positive definite if u was too large
%         .0421
        % It is possible to check all eigenvalues in O(n^2) rather than O(n^3)
        % but it's not particularly simple to implement.
        % See http://www.stat.uchicago.edu/~lekheng/courses/309f10/modified.pdf
        % Golub, 1973, "Some Modified Matrix Eigenvalue Problems"
        % http://epubs.siam.org/doi/abs/10.1137/1015032
        % but...
        %   if the diagonal term is just a scaled identity,
        %   then it is trivial
        if all( d==d(1) ) && ~all(u==0)% diagonal
            minE = 1/d(1) - norm(Dinv*L)^2/(1+R'*Dinv*L);
        else
            minE = min(eig(V));
        end
        if minE <= 0
            error('V must be positive definite');
        end
    end
else
    V       = diag(d) + pm*(u*u');
    if pm == -1
        if all( d==d(1) ) % diagonal
            minE = d(1) - norm(u)^2;
        else
            minE = min(eig(V));
        end
        if minE <= 0
            error('V must be positive definite');
        end    
    end
end
    
if exist('OCTAVE_VERSION','builtin') || ~exist('cvx_begin','file')
    x = Inf;
    return;
end

x = solveInCVX(type,x0,V,offset,lambda,lwr,upr);
% clean it up a bit:
x    = x.*( abs(x) > 1e-10 );

end % end of function


function x = solveInCVX(type,x0,V,offset,lambda,lwr,upr)
n   = length(x0);
cvx_precision best
cvx_quiet true
%     minimize lambda*norm(x,1) + 1/2*sum_square( Vsqrt*(x-x0) ) + dot(offset,x)
    % avoid Vsqrt=sqrtm(V) for more accurate answer:
    
switch lower(type)
    case 'l1'
        cvx_begin
        variable x(n,1)
        minimize norm(lambda.*x,1) + 1/2*quad_form(x-x0, V ) + dot(offset,x)
        cvx_end
    case 'l1pos'
        cvx_begin
        variable x(n,1)
        minimize norm(lambda.*x,1) + 1/2*quad_form(x-x0, V ) + dot(offset,x)
        subject to
            lambda.*x >= 0
        cvx_end
    case 'rplus'
        cvx_begin
        variable x(n,1)
        minimize 1/2*quad_form(x-x0, V ) + dot(offset,x)
        subject to 
            lambda.*x >= 0
        cvx_end
    case 'box'
        if ~all( lwr <= upr )
            error('Problem is infeasible');
        end
        % Carefully handle cases when lwr = -Inf and/or upr=+Inf
        set1 = ~isinf(lwr);
        set2 = ~isinf(upr);
        if length(lambda)==1, lambda = repmat(lambda,n,1); end
        cvx_begin
        variable x(n,1)
        minimize 1/2*quad_form(x-x0, V ) + dot(offset,x)
        subject to 
            lambda(set1).*x(set1) >= lwr(set1)
            lambda(set2).*x(set2) <= upr(set2)
        cvx_end
    case 'hinge'
        hinge = @(x) sum(max(0,1-x));
        cvx_begin
         variable x(n,1)
         minimize 1/2*quad_form(x-x0,V) + dot(offset,x) + hinge(lambda.*x)
        cvx_end
    case 'linf'
        hinge = @(x) sum(lambda.*max(0,1-x));
        cvx_begin
         variable x(n,1)
         minimize 1/2*quad_form(x-x0,V) + dot(offset,x)
         subject to
           norm(lambda.*x, inf ) <= 1
        cvx_end
    otherwise
        error('That type is not yet supported');
end
end