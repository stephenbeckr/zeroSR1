function [x,a,cnt] = prox_rank1_generic( prox, prox_brk_pts, x0, D, u, lambda, linTerm, plusminus, INVERT )
% PROX_RANK1_GENERIC returns the scaled proximity operator for a generic function h
%   (provided the generic function is separable and has a piece-wise linear prox)
% This function is intended be used as follows:
%
%   (1) Instantiate:
%        scaledProx = @(varargin) prox_rank1_generic( prox, prox_brk_pts,varargin{:})
%           where 'prox' and 'prox_brk_pts' implicitly define the function h
%           i.e., prox(x0,t) = argmin_{x} t*h(x) + 1/2||x-x0||^2
%           and 
%               prox_brk_pts(t) returns a row-vector with the break points
%               that specify where t*h(x) is piecewise linear
%               (this is if h(x) = [ h_1(x_1); ... ; h_n(x_n) ]. If instead not
%                all the h_i are identical, prox_brk_pts(t) should return
%                a matrix).
%           See the examples below because prox_brk_pts must allow a vector "t"
%            so you must define this appropriately.
%
%   (2) Call the "scaledProx" function, which has signature:
%       x = scaledProx( x0, D, u )
%           where 
%           x = argmin_{x} h(x) + 1/2||x-x0||^2_{V}
%           and
%           V^{-1} = D + u*u'  (or diag(D) + u*u' if D is a vector)
%           "D" must be diagonal and positive. "u" can be any vector.
%
%       There are also variants:
%
%       x = scaledProx( x0, D, u, lambda, linTerm, sigma, inverse)
%           returns
%           x = argmin_{x} h(lambda.*x) + 1/2||x-x0||^2_{V} + linTerm'*x
%           and
%           either V^{-1} = D + sigma*u*u' if "inverse" is true (default)
%           or     V      = D + sigma*u*u' if "inverse" is false
%           and in both cases, "sigma" is either +1 (default) or -1.
%           "lambda" should be non-zero
%
%   Examples:
%      1. if h(x) = ||x||_1 then
%           prox            = @(x,t) sign(x).*max(0, abs(x) - t );
%           prox_brk_pts    = @(t) [-t,t];
%      2. if h(x) is the indicator function of the set { x : x >= 0}, then
%           prox            = @(x,t) max(0, x);
%           prox_brk_pts    = @(t) 0; 
%      3. if h(x) is the indicator function of the set { x : lwr <= x <= upr }
%           where lwr and upr are vectors, then
%           prox            = @(x,t) max( min(upr,x), lwr );
%           prox_brk_pts    = @(t) [lwr,upr];  (Note: this is a matrix)
%      4. if h(x) is the hinge-loss h(x) = max( 1-x, 0 ), then
%           prox        = @(x,t) 1 + (x-1).*( x > 1 ) + (x + t - 1).*( x + t < 1  );
%           prox_brk_pts    = @(t)[ones(size(t)), 1-t];
%      5. if h(x) is the indicator function of the l_infinity ball, then
%           prox            = @(x,t) sign(x).*min( 1, abs(x) );
%           prox_brk_pts    = @(t) [-ones(size(t)),ones(size(t))]; 
%
%
% Stephen Becker, Feb 26 2014, stephen.beckr@gmail.com
% Reference: "A quasi-Newton proximal splitting method" by S. Becker and J. Fadili
%   NIPS 2012, http://arxiv.org/abs/1206.1156

PRINT = false; % set to "true" for debugging purposes
if PRINT
    dispp = @disp;
    printf = @fprintf;
else
    dispp = @(varargin) 1;
    printf = @(varargin) 1;
end
dispp(' ');

n   = length(x0);
if nargin < 5 || isempty(u), u = 0; end
if nargin < 6, lambda = []; end
if nargin < 7, linTerm = []; end
if nargin < 8 || isempty(plusminus), plusminus = 1; end
assert( plusminus==-1 | plusminus==+1 )
if nargin < 9 || isempty(INVERT), INVERT = true; end

if size(D,2) > 1, d = diag(D); else d = D; end % extract diagonal part
if any( d < 0 ), error('D must only have strictly positive entries'); end

if all( u==0 )
    % Just a diagonal scaling, so this code is overkill,
    % but we should be able to handle it for consistency.
    NO_U = true;
else
    NO_U = false;
    if numel(u) > length(u)
        error('u must be a vector, not a matrix');
    end
end

% Now, V > 0 (i.e., V is positive definite) iff V^{-1} exists and V^{-1} > 0
% So V^{-1} > 0 is automatically true if sigma = + 1
% If sigma = -1, then it could be indefinite or semidefinite
%
% It is possible to check all eigenvalues in O(n^2) rather than O(n^3)
% but it's not particularly simple to implement.
% See http://www.stat.uchicago.edu/~lekheng/courses/309f10/modified.pdf
% Golub, 1973, "Some Modified Matrix Eigenvalue Problems"
% http://epubs.siam.org/doi/abs/10.1137/1015032
% But in the special case when D is a scaled identity, checking is very easy:
if plusminus < 0 && all( d==d(1) )
    minE = d(1) + plusminus*norm(u)^2;
    if minE <= 0, error('The scaling matrix is not positive definite'); end
end

% this comes from the Sherman-Morrison-Woodbury formula:
if NO_U
    uinv = 0;
else
    uinv    = (u./d)/sqrt(1+u'*(u./d));
end
% In all cases, we find prox_h^V, but how we define V
%   in terms of d and u depends on "INVERT"
if INVERT
    % So V^{-1} = diag(d)     + sigma*u*u'
    % and     V = diag(1./d)  - sigma*uinv*uinv';
    Vinv = @(y) d.*y + plusminus*(u'*y)*u;
    
    %   The code below expects V = diag(dd) + sigma*uu*uu', so...
    dd          = 1./d; 
    uu          = uinv; 
    plusminus   = -plusminus;
    
    % The code also requires uu./dd and 1./dd, so define these here
%     ud      = uu./dd; 
    ud      = u/sqrt(1+u'*(u./d)); % more accurate? % 6.01e-3 error
    dInv    = 1./dd;
else
    % Here, V    = diag(d) + sigma*u*u'
    % and V^{-1} = diag(1./d) - sigma*uinv*uinv';
    Vinv = @(y) y./d - plusminus*(uinv'*y)*uinv;

    %   The code below expects V = diag(dd) + sigma*uu*uu', so...
    dd          = d; 
    uu          = u; 
    %plusminus   = plusminus;
    
    % The code also requires uu./dd and 1./dd, so define these here
    ud      = uu./dd;
    dInv    = 1./dd;
end
if NO_U, uu = 0; ud = 0; end % any value, since we won't use them...
if ~isempty(lambda)
    % We make a change of variables, e.g., x <-- lambda*.x
    % change x0 <-- lambda.*x0, linTerm <-- linTerm./lambda
    % and V <-- diag(1./lambda)*V*diag(1./lambda). Because V is defined
    % implicitly, and depends on INVERT, this is a bit of a headache.
    % We'll do some changes here, and some later in the code
    % e.g., combine linTerm and V scaling so we don't have to redefine Vinv
    if any(lambda==0), error('scaling factor lambda must be non-zero'); end
    % note that lambda < 0 should be OK
    x0 = lambda.*x0;

    % Scale V = diag(dd) + sigma*uu*uu' by V <-- diag(1./lambda)*V*diag(1./lambda)
    dd = dd./(lambda.^2);
    uu = uu./lambda;
    ud = ud.*lambda;
    dInv    = 1./dd;
end

t   = prox_brk_pts(1./dd);
if size(t,1) < n
    if size(t,1) > 1
        error('"prox_brk_pts" should return a ROW VECTOR of break points');
    end
    % otherwise, assume each component identical, so scale
    t = repmat(t,n,1);
end
if ~isempty(linTerm) && norm(linTerm)>=0
    if isempty(lambda)
        x0  = x0 - Vinv(linTerm);
    else
        % V is scaled V <-- diag(1./lambda)*V*diag(1./lambda)
        %   so Vinv is scaled the opposite.
        % linTerm is scaled linTerm <== linTerm./lambda
        x0  = x0 - lambda.*Vinv(linTerm);
    end
end

% The main heart:
X       = @(a) prox( x0 - plusminus*a*ud, dInv );

% Early return if we have only a diagonal scaling...
if NO_U
    % in this case, "alpha" is irrelevant
    x   = prox( x0, dInv );
    if ~isempty(lambda)
        % Undo the scaling of x <-- lambda.*x
        x = x./lambda;
    end
    return;
end
    
brk_pts = bsxfun(@times,plusminus*(dd./uu),  bsxfun(@minus,x0,t) );
brk_pts = unique(brk_pts(:)); % will sort and remove duplicates
brk_pts = brk_pts(~isinf(brk_pts)); % in case lwr/upr=Inf for box


% p(a) = a + dot(u, y - prox_{1/d_i}( y_i - a u_i/d_i) )
% Then p is strictly increasing. We want a root of this: p(a) = 0

% Defined above for numerical reasons...
% ud      = uu./dd;
% dInv    = 1./dd;


% Main for-loop:
% "lower bound" are "a" for which p <= 0
% "upper bound" are "a" for which p >= 0
% if a is increasing, so is p(a) (double-check for both plusminus cases )
lwrBnd       = 0;
uprBnd       = length(brk_pts) + 1;
iMax         = ceil( log2(length(brk_pts)) ) + 1;
for i = 1:iMax
    if uprBnd - lwrBnd <= 1
        dispp('Bounds are too close; breaking');
        break;
    end
    j = round(mean([lwrBnd,uprBnd]));
    printf('j is %d (bounds were [%d,%d])\n', j, lwrBnd,uprBnd );
    if j==lwrBnd
        dispp('j==lwrBnd, so increasing');
        j = j+1;
    elseif j==uprBnd
        dispp('j==uprBnd, so increasing');
        j = j-1;
    end
    
    a   = brk_pts(j);
    x   = X(a);  % the prox
    p   = a + dot(uu,x0-x);
    
    if p > 0
        uprBnd = j;
    elseif p < 0
        lwrBnd = j;
    end
    if PRINT
        % Don't rely on redefinition of printf,
        % since then we would still calculate find(~x)
        % which is slow
        printf('i=%2d, a = %6.3f, p = %8.3f, zeros ', i, a, p );
        if n < 100, printf('%d ', find(~x) ); end
        % printf('; nonzeros ');printf('%d ', find(x) );
        printf('\n');
    end
end
cnt     = i; % number of iterations we took

% Now, determine linear part, which we infer from two points.
% If lwr/upr bounds are infinite, we take special care
% e.g., we make a new "a" slightly lower/bigger, and use this
% to extract linear part.
if lwrBnd == 0
    a2 = brk_pts( uprBnd );
    a1 = a2 - 10; % arbitrary
    aBounds = [-Inf,a2];
elseif uprBnd == length(brk_pts) + 1;
    a1 = brk_pts( lwrBnd );
    a2 = a1 + 10; % arbitrary
    aBounds = [a1,Inf];
else
    % In general case, we can infer linear part from the two break points
    a1 = brk_pts( lwrBnd );
    a2 = brk_pts( uprBnd );
    aBounds = [a1,a2];
end
x1 = X(a1);
x2 = X(a2);
dx = (x2 - x1)/(a2-a1); 
% Thus for a in (a1,a2), x(a) = x1 + (a-a1)*dx
% Solve 0 = a + dot( uu, y - (x1 + (a-a1)*dx ) )
%         = a + dot(uu,y - x1 + a1*dx ) - a*dot(uu,dx)
% so:
a = dot( uu, x0 - x1 + a1*dx)/( -1 + dot(uu,dx) );
if a < aBounds(1) || a > aBounds(2), error('alpha is not in the correct range'); end
% If we were not linear, we could do a root-finding algorithm, e.g., 
% a = fzero( @(a) a+dot(uu,x0-X(a)), a );

% Now, the solution:
x = X(a);

if ~isempty(lambda)
    % Undo the scaling of x <-- lambda.*x
    x = x./lambda;
end

printf('Took %d of %d iterations, lwrBnd is %d/%d \n', i, iMax, lwrBnd,length( brk_pts ) );
