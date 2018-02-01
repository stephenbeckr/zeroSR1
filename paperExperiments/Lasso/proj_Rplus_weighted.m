function [x,lambda,cnt, sEst] = proj_Rplus_weighted( x0, D, L, scale, linTerm )
% x = proj_Rplus_weighted( x0, D, L )  or 
% x = proj_Rplus_weighted( x0, D, L, scale ) or 
% x = proj_Rplus_weighted( x0, D, L, scale, c )
%   returns the solution
%       x = argmin   1/2||x-x0||^2_{Q,2}' + <c,x>  subject to scale*x >= 0
% where
%   ||x-x0||^2_{Q,2} = < x - x0, Q*(x-x0) >
% and 
%   inv(Q) = D + L*L'  is a diagonal + rank-1 matrix
%   with D = diag(d) > 0 is psd.
%
% Only the sign of the "scale" input has an effect (if scale < 0,
%   the the constraints become x <= 0 instead of x >= 0 ).
%
% The algorithm takes O( n*log(n) ) running time.
%
% [x,lambda] = ...
%   also returns a dual vector lambda
%
% [x,lambda,iter] = ...
%   also returns the number of iterations ( iter <= ceil( log_2(n) + 1 ) ).
%
% [x,lambda,iter,s] = ...
%   also returns the scalar dual variable 's'
%
% Stephen Becker, Dec 1 2011.  srbecker@caltech.edu
% March 13, changing order of (... linTerm, scale ) to (... scale, linTerm )


VERBOSE = false;
if nargin < 3, L = []; end % this is just scalar stuff then...
if nargin < 5, linTerm = []; end
if nargin < 4 || isempty(scale), scale = 1; end

if isvector(D), d = D; %D = diag(d); 
else d = diag(D); end
if any( d < 0 ), error('Diagonal term must be positive'); end
% make sure everything is a column vector
if size(x0,2) > 1, x0 = x0.'; end
if size(L,2)  > 1, L  = L.' ; end



% -- If the user doesn't specify L, then it's a standard projection --
if isempty(L)
    if ~isempty( linTerm )
        error('Can''t handle that case yet. Shouldn''t be too difficult though...');
    else
        x   = max( x0, 0 );
        lambda  = [];
        cnt     = 1;
        sEst    = 0;
        return;
    end
end




RESCALE     = false;
if scale == 0
    error('Cannot handle lambda = 0');
elseif scale < 0
   RESCALE  = true;
   x0       = -x0;
end

R = L;
N = length(x0);

if nargin >= 4 && ~isempty( linTerm )
    % We can incorporate this (i.e. "c" in the equation above,
    %   but not the same "c" used below) into the x0 term:
    if size(linTerm,2) > 1, linTerm = linTerm.'; end
    if RESCALE, linTerm = -linTerm; end
    x0  = x0 - (d.*linTerm + L*(R'*linTerm) );
    
end


% from now on, "lambda" refers to the dual vector
%   We will find a strictly complementary solution (x,lambda) such
%   that x >= 0, lambda >= 0, and <x,lambda> = 0

% sList   = -x0./L;
% sList   = sort(sList); % +/- inf are OK.
sList   = unique( -x0./L ); % remove duplicate +/- infinities
sListInf = [ -Inf; sList; Inf ];

S       = [ sList(1)-1;   (sList + circshift(sList,-1))/2 ];
S(end)  = sList(end) + 1;
% so the element of S are right in the middle: no boundary points,
%   ensuring strict complementarity

% Thus, we have defined the active set for both x and lambda
DONE    = false;
mn      = 0; % inclusive
mx      = length(sList); % inclusive
maxIt   = ceil( log2(mx) ) + 1;

A       = -R./d;
B       = A.*L;
A       = A.*x0;

for cnt = 0:maxIt % should take logN iterations, or fewer
    
    k   = round( (mn+mx)/2 );   % pick the next entry
    s   = S(k+1);  % i.e. sList(k-1) < s < sList(k)
    
%     T   = find( x0 + s*L > 0);
    Tc  = find( x0 + s*L < 0); % we never have y + s = 0, by design of S
    
    % support of lambda is now well defined in terms of 's'.
%     a   = -R(Tc)'*( x0(Tc)./d(Tc) ); 
%     b   = -R(Tc)'*(  L(Tc)./d(Tc) );
    % alternatively, compute them this way (might be faster):
    a   = sum( A(Tc) );
    b   = sum( B(Tc) );
    sEst    = a/(1-b);
    
    % find bounds:
    lb  = sListInf( k+1 );
    ub  = sListInf( k+2 );
    
    % debugging: verify that these are indeed the correct bounds:
%     OK  = ( s > lb ) && ( s < ub );
%     if ~OK, disp('Violated bounds!'); error('Problem!'); end
    
    if sEst < lb
        str = 'v';
        % reduce the upper bound
        mx  = k;
    elseif sEst > ub
        str = '^';
        % increase the lower bound
        mn  = k;
    else
        str = '-';
        DONE = true;
    end
    if VERBOSE, fprintf('k=%2d, [%6.1f, %6.1f], sEst is %6.1f:  %s\n',k,lb,ub, sEst, str ); end
    if DONE, break; end
end
assert( cnt < maxIt, 'rank-1 prox algorithm failed to converge');

T       = find( x0 + s*L > 0);
x       = zeros(N,1);
x(T)    = x0(T)  + sEst*L(T);
if nargout > 1
    lambda  = zeros(N,1);
    lambda(Tc)  = -(x0(Tc) + sEst*L(Tc) )./d(Tc);
end

if RESCALE
    x = -x;
    if nargout > 1
        lambda  = -lambda;
    end
end
