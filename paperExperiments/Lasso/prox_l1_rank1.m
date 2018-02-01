function [x,cBest,cnt] = prox_l1_rank1( x0, D, L, lambda, linTerm )
% x = prox_l1_weighted( x0, D, u )  or 
% x = prox_l1_weighted( x0, D, u, lambda ) or 
% x = prox_l1_weighted( x0, D, u, lambda, c )
%   returns the solution
%       x = argmin  lambda*||x||_1 + 1/2||x-x0||^2_{B,2}' + <c,x>
% where
%   ||x-x0||^2_{B,2} = < x - x0, B*(x-x0) >
% and 
%   H = inv(B) = D + u*u'  is a diagonal + rank-1 matrix
%   with D = diag(d) > 0 is positive definite.
%
% The algorithm takes O( n*log(n) ) running time.
% Inputs must be real, not complex.
%
% [x,c] = ...
%   also returns c, where
%   x = shrink( x0 - c*u, d );
%
% [x,c,iter] = ...
%   also returns the number of iterations ( iter <= ceil( log_2(n) + 1 ) ).
%
% Stephen Becker, Dec 10 2010 -- April 2012.  stephen.beckr@gmail.com

% Modified May 27 2011 to handle the "c" term
% Modified Nov 24 2011 to handle the "lambda" term and be more efficient
% Modified Feb 29 2012 to be more accurate when L has many zeros
% Modified Mar 13 2012 to allow vector "lambda" term.

% Note about the code: the documentation refers to H = D + u*u'
%   The code uses the notation H = D + L*R' (L for left, R for right)
%   In general, we need R=L (so that H is positive definite),
%   but we keep the "R" notation because it makes the derivation more clear,
%   and we don't always have R=L after we remove zero terms from L.


% -------- Preprocess ---------------------

if isvector(D), d = D; % D = diag(d); 
else d = diag(D); end
if any( d < 0 ), error('Diagonal term must be positive'); end
% make sure everything is a column vector
if size(x0,2) > 1, x0 = x0.'; end
if size(L,2)  > 1, L  = L.' ; end

if nargin >= 5 && ~isempty( linTerm )
    if size(linTerm,2) > 1, linTerm = linTerm.'; end
    x0  = x0 - (d.*linTerm + L*(L'*linTerm) );
end

if nargin < 4 || isempty(lambda), lambda = 1; end
if numel(lambda)>1 
    % rescale
    if size(lambda,2) > 1, lambda = lambda.'; end 
    if size(lambda,2) > 1
        lambda = diag(lambda); 
    end
    d   = lambda.*d;
elseif lambda ~= 1
    % rescale
    d    = lambda*d;
    L    = sqrt(lambda)*L;
end
if isscalar(d), d = d*ones(size(x0)); end
N = length(x0);

% Now, we can pretend lambda=1 and linTerm=0, since they have been accounted for

shrinkVec   = @(x,d) sign(x).*max( abs(x) - d, 0 );

% If there is no low-rank term:
if nargin < 3 || isempty(L)
    x       = shrinkVec( x0, d );
    cBest   = 0;
    cnt     = 0;
    return;
end

% Account for cases when L has many zeros...
nonzeroL    = find( abs(L) > 100*eps );
if length(nonzeroL) < N
    L_HAS_ZEROS = true;
    
    % and reduce the rest of it to a smaller problem:
    old_L   = L;
    old_x0  = x0;
    old_d   = d;
    x0  = x0(nonzeroL);
    d   = d(nonzeroL);
    L   = L(nonzeroL);
else
    L_HAS_ZEROS = false;
end
R = L;
if numel(lambda)>1  % For diagonal lambda
    if L_HAS_ZEROS
        R = lambda(nonzeroL).*R;
    else
        R = lambda.*R;
    end
end

c1 = (x0+d)./L; % if x_i < 0
c2 = (x0-d)./L; % if x_i > 0
c = [c1,c2];
cList = sort(c(:));  % list of break-points.
offset = 1e0;
cList2 = [ cList(1)-offset; cList + [diff(cList)/2;offset]  ]; % look in-between stuff
cListInf = [-Inf; cList; +Inf ];
sL = sign(L);

sLc1 = sL.*c1; % precompute
sLc2 = sL.*c2;

NN = length(cList2);
% Keep track of counters:
mn  = 1;
mx  = NN;
cnt = 0;
j = round( (mn+mx)/2 );

% This loop would be nice in a mex file
%   (we want to do the "sort" in Matlab, since Matlab has a great sort function)

% -------- Main loop ---------------------
maxIt = NN+3;
while cnt < maxIt  % should never max out, but just in case of infinite loop due to coding error...
    cnt = cnt + 1;
    ci = cList2(j);
    
    % -- Step 1: estimate the support 
    dx = ( sL*ci < sLc2 ) - ( sL*ci > sLc1 );
    
    
    Tc = ~dx; 
    T  = ~~dx;
    alpha = R(T)'*dx(T);
    
    invA_vec = 1./d(Tc); % precompute for speed
    
    
    u = L(Tc);
    v = R(Tc); % since lambda nonscalar, we may have u ~= v
    
    vv  = invA_vec.*v;
    zc  = 1 + vv'*u;
%     QQ = invA - invA*u*v'*invA/zc;    % conceptually, this is what we do, but this is slow numerically
%     dxTc = QQ*(x0(Tc) - alpha*L(Tc) );
    
    % Make the above faster:
    yy   = x0(Tc) - alpha*L(Tc);
    dxTc = invA_vec.*(yy - u*(vv'*yy)/zc);
    
    dx(Tc) = dxTc;
    cEst = R'*dx;   % based on this support, this is our estimate of the shrinkage scalar

    % Test if this shrinkage scalar is permissible 
    if cEst < cListInf(j)
        % We need to decrease the value of c
        mx = j;
    elseif cEst > cListInf(j+1)
        % We need to increase the value of c
        mn = j;
    else
        % The support is acceptable!
        cBest   = cEst;
        if any( abs(dxTc) > 1 )
            disp('Weird behavior: bad subgradient');
            cBest = NaN;
        end
        break;
    end

    % Next direction:
%     j = round( (mn+mx)/2 );
    if mx > mn + 1
        j = round( (mn+mx)/2 );
    else
        % There are only two left, [mn mn+1]
        if j == mn
            j = mn+1;
        else
            j   = mn;
        end
    end
    
    
end
assert( cnt < maxIt, 'rank-1 prox algorithm failed to converge');
if isnan(cBest)
    warning('Found NaN','prox_l1_weighted:failed');
    x = NaN;
else
    % Account for cases when L has many zeros...
    if L_HAS_ZEROS
        x = shrinkVec( old_x0 - cBest*old_L, old_d ); % for
    else
        % In this case, I didn't waste the memory to copy L, x0 and d
        x = shrinkVec( x0 - (cBest)*L, d );
    end
end
