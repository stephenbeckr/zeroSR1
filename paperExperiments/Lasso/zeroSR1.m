function [xk,nit, errStruct, defaultOpts, stepsizes] = zeroSR1(fcn,grad,prox,opts)
%   [xk,nit, errStruct, outOpts] = zeroSR1(f,g,proj,opts)
%
% This uses the zero-memory SR1 method to solve:
% 
%   min_x f(x)  + h(x)
%
% where
%   'f' calculates f(x), 'g' calculates the gradient of f at x,
%   and h(x) is a non-smooth term that can be infinite-valued (a constraint),
%   so long as you present a function 'prox' that computes diagional plus
%   rank-1 projections. The 'prox' function should accept at least three inputs:
%
%   y = prox( x0 , d, v, ) 
% where
%   y = argmin_x h(x) + 1/2||x-x0||^2_B 
% where
%   B = inv(H) = inv( diag(D) + v*v' )
%
% If 'prox' isn't provided or is [], it defaults to the identity mapping, which corresponds
%   to the case when h=0.
%
% "opts" is a structure with additional options. To see their default values,
%   call this function with no input arguments.
%
%   .tol is a tolerance for relative variation
%   .nmax is max # of allowed iterations
%   .verbose can be either 0 (no output), 1 (every iteration), or n
%       If 'n' is an integer greater than 1, output will be written
%       every n iterations
%   .x0
%       starting vector
%   .N
%       size of primal domain (only necessary of x0 wasn't provided)
%
%   .SR1  if true, uses the zero-memory SR1 method
%         if false, uses gradient descent/forward-backward method
%         (or variant, such as FISTA, or BB stepsizes as in the SPG method)
%
%   .errFcn can be an arbitrary function that calculates an error metric
%       on the primal variable at every iteration.
%
%   .L, .L0, .BB
%       control the computation of the stepsize. 
%       .BB uses the Barzilai-Borwein stepsizes; when using SR1,
%           you can set 0 < opts.SR1_diagWeight < 1 to control
%           how much of the BB step is taken.
%       .L is an upper bound on the true Lipschitz constant
%       if .L is not provided, then provide .L0 as a guess
%
%   .backtrack
%       controls whether backtracking is done
%
%   .theta
%       set to 1 for gradient descent, set to within (0,2] for over-relaxed
%       forward-backward splitting, and set to [] (default) to use the
%       FISTA acceleration. This controls the momentum term. Cannot
%       be combined with the SR1 mode (in SR1 mode, theta=1).
%   .restart
%       in FISTA mode, resets the 'theta' variable every "restart" iterations
%
%   [and many more undocumented features...]
%
% Output "errStruct" contains three or four columns:
%   (1) objective function
%   (2) norm of gradient
%   (3) stepsize
%   (4) error (i.e. the output of errFcn, if provided)
%   
% Stephen Becker, Nov 24 2011 -- April 2012

if nargin == 0 || nargout >= 4
    RECORD_OPTS     = true;
    defaultOpts     = [];
else
    RECORD_OPTS     = false;
end

if nargin < 3 || isempty(prox), prox = @(x,diag,v) x; end
if nargin < 4, opts = []; end 

% see also "setOptsFcn.m"
function out = setOpts( field, default, mn, mx )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
    if nargin >= 3 && ~isempty(mn) && any(out < mn), error('Value is too small'); end
    if nargin >= 4 && ~isempty(mx) && any(out > mx), error('Value is too large'); end
    opts    = rmfield( opts, field ); % so we can do a check later
    if RECORD_OPTS
        defaultOpts.(field) = out;
    end
end


fid     = setOpts( 'fid', 1 );      % print output to the screen or a file
myDisp  = @(str) fprintf(fid,'%s\n', str );
tol     = setOpts( 'tol', 1e-6 );  
grad_tol= setOpts( 'grad_tol', tol );
nmax    = setOpts( 'nmax', 1000 );  
errFcn  = setOpts( 'errFcn', [] );
VERBOSE = setOpts( 'verbose', false );
if isinf(VERBOSE), VERBOSE = false; end
maxStag = setOpts( 'maxStag', 10 );     % used for forcing very high accuracy
backtrackType    = setOpts( 'backtrackType', 2 ); % which test to use
backtrackScalar  = setOpts( 'backtrackScalar', 0.8 );
FORCE_MIN_STEPSIZE = setOpts( 'FORCE_MIN_STEPSIZE', false ); % if BB < 1/L, then take 1/L

xk      = setOpts( 'x0', [] );
N       = setOpts( 'N', length(xk) );
if N==0 && nargin > 0, error('for now, must specify opts.N = N'); end
if isempty(xk), xk = zeros(N,1); end

% -- Options that concern the stepsize --
L               = setOpts( 'L', [] );   % Lipschitz constant, e.g. norm(A)^2
BB              = setOpts( 'BB', false );
BB_type         = setOpts( 'BB_type',[], 1,2); % type 1 or 2
BB_alt          = setOpts( 'BB_alt', false ); % alternate type of BB
BB_min          = setOpts( 'BB_min', 1e-10);
BB_max          = setOpts( 'BB_max', 1e10);

SR1             = setOpts( 'SR1', false );
SR1_diagWeight  = setOpts( 'SR1_diagWeight', 0.8 );
SR1_max         = setOpts( 'SR1_max', 1e10 ); % bound the inverse Hessian spectral norm uniformly

backtrack       = setOpts( 'backtrack', true );
% We might want to disable any line search, since the quasi-Newton methods
%   are already well-scaled.
% noLinesearch    = setOpts( 'noLinesearch', false );
% if backtrack  + noLinesearch ~= 1
%     error('''backtrack'', and ''noLinesearch'' are mutually exclusive');
% end
noLinesearch = ~backtrack;

KNOWN_L     = ~isempty(L);
if isempty(L), 
    L   = setOpts( 'L0', 1, 0 );
end % when we do linesearch, this isn't so important



% -- Options that concern choosing "theta" --
% We can set theta in [0,2]. If we choose theta > 1, we do the usual
%   forward-backward over-relaxation, which means we still take a convex
%   combination of old and new steps.
% If theta = [], then we do FISTA rule
if SR1
    fixed_theta     = setOpts( 'theta', 1, 0, 2 );
    fixed_theta = 1; % override any option given
else
    fixed_theta     = setOpts( 'theta', [], 0, 2 );
end
restart          = setOpts( 'restart', Inf ); % only applies if theta=[]



if isempty(BB_type)
    if SR1 || isempty(fixed_theta) || fixed_theta ~= 1, BB_type = setOpts('BB_type',2);
    else, BB_type   = setOpts('BB_type',1); % faster, generally
    end
end
if SR1 && BB_type == 1
    warning('With zero-memory SR1, BB_type must be set to 2. Forcing BB_type = 2 and continuing','zeroSR1:BB_warn');
    BB_type     = 2;
end
if SR1 && BB_alt
    warning('With zero-memory SR1, cannot use BB_alt. Forcing BB_alt = false and continuing','zeroSR1:BB_warn');
    BB_alt      = false;
end




DISABLE_L_WARNING = false;
% ------------ Scan options for capitalization issues, etc. -------
if nargin == 0 
    disp('Default options:');
    disp( defaultOpts );
end
if ~isempty(fieldnames(opts))
    disp('Error detected! I didn''t recognize these options:');
    disp( opts );
    error('Bad options');
end
if nargin == 0 , return; end

% ------------ Begin algorithm ------------------------------------

yk      = xk;
xk_old  = xk;
yk_old  = xk;
grad_y  = zeros(N,1);
fxold   = Inf;

cnt  = Inf;
t    = 1/L; % initial stepsize
t_old= t;
stag = 0;   % for detecting stagnation/convergence
nit  = 0;   % iteration counter
tk   = 1;   % used for updating theta


% Collect some statistics
stepsizes = zeros(nmax,1 + 2*SR1);

if ~isempty(errFcn)
    if ~isa(errFcn,'function_handle')
        error('errFcn must be a function');
    end
    errStruct   = zeros( nmax, 4 ); % f, norm(gx), step, err
else
    errStruct   = zeros( nmax, 3 ); % f, norm(gx), step
end
while (nit < nmax)
    
    nit = nit +1;
    skipSR1     = false;
    

    grad_y_old  = grad_y;
    grad_y  = grad(yk);
    p       = -grad_y; % initial search direction (later, it may be scaled)

    if nit > 1
        sk_old  = sk;   % I might use this later...
        gk_old  = gk;   % I might use this later...
    end
    if isempty( fixed_theta ) % e.g. FISTA method
        sk  = yk - yk_old;  % trying this Fri, Feb 24
    else
        sk  = xk - xk_old;
    end
    gk      = grad_y  - grad_y_old;   % this is "yk" in Nocedal/Wright
    if nit > 1 && norm(gk) < 1e-13
        warning('||gk||=0, try changing opts.L0','specialSR1:zeroChangeInGradient');
        gk = [];
        skipBB = true;
    end
    
    
    % ---------------------------------------------------------------------
    % -- Find an initial stepsize --
    % ---------------------------------------------------------------------
    if t < 1e-14
        stag    = Inf;
%         error('stepsize is too small'); 
    end
    t_old   = t;
    if BB && nit > 1 && ~skipBB  
        if BB_alt
            % In this case, we alternate between the two types of BB steps
            BB_type     = 1 + rem(nit,2);
        end
        if BB_type == 2
            t   = sk'*gk/( norm(gk)^2 ); % eq (1.7) in Dai/Fletcher. This is shorter
        elseif BB_type == 1
            t   = (norm(sk)^2)/(sk'*gk); % eq (1.6) in Dai/Fletcher. This is longer
        else
            error('bad value for BB_type: must be 1 or 2');
        end
        
        % and uniformly bound the stepsizes
        if t < BB_min && SR1
            % in this case, we'll have to skip our rank-1 step
            skipSR1 = true;
        end
        t   = min(   max( t, BB_min ), BB_max );
        
        if t <= 0 % t < 0 should not happen on convex problem!
            % See Nocedal and Wright: we can sometimes do something.
            % There are really two curvature conditions (one from SR1,
            % another in BFGS), and one is more important than the other;
            % in particular, an indefinite Hessian can be OK, since that
            % often reflects reality.
            if KNOWN_L
                myDisp('Curvature condition violated; setting stepsize to 1/L');
                t = 1/L;
            else
                myDisp('Curvature condition violated!');
                myDisp('Quitting...');
                stag    = Inf;
            end
        end
        if KNOWN_L && t <1/L && FORCE_MIN_STEPSIZE
            t = 1/L;
        end
        if SR1
            t_BB    = t;
            % we cannot take a full BB step, otherwise we exactly satisfy the secant
            %   equation, and there is no need for a rank-1 correction.
            if BB_type == 2
                t    = SR1_diagWeight*t_BB; % SR1_diagWeights is a scalar less than 1 like 0.6
            else
                error('Cannot handle this variant of BB');
            end
        end
        H0  = @(x) t*x;
        diagH   = t*ones(N,1);
        B0  = @(x) (1/t)*x;
    else 
        % Simplest: fixed step length
        if backtrack
            if cnt == 1
                % our previous stepsize was accepted without any backtracking, 
                %  so let's be aggressive
                t = 1.1*t_old;
            else
                t = t_old;
            end
        else
            t   = 1/L;
        end
        H0  = @(x) t*x; 
        B0  = @(x) (1/t)*x; 
        diagH   = t*ones(N,1);
    end
    skipBB  = false;
    t_absolute = t;
    stepsizes(nit,1) = t;
    
    
    
    % ---------------------------------------------------------------------
    % -- Quasi-Newton -- Requries: H0, and builds H
    % ---------------------------------------------------------------------
    if SR1 && nit > 1 && ~isempty(gk) && ~skipSR1
        gs = gk'*sk;
        gHg = gk'*(diagH.*gk);
%         if gs < 1e-14,
        if gs < 0
            myDisp('Serious curvature condition problem!');
%             return
            stag = Inf;
        end
        if gHg > gs % this shouldn't happen, since we scale BB step
            % Stepsize was huge, and we can't choose a rank-1 term unless we make H0 smaller
            scale   = 0.95*gs/gHg;
            diagH = scale*diagH;
%             myDisp('  warning');
        end
        H0  = @(x) diagH.*x;
        B0  = @(x) x./diagH;

        
        vk  = sk - H0(gk);
        v   = B0(vk);
        if vk'*gk  <= 0
            myDisp('Warning: violated curvature conditions');
            % This should only happen if we took an exact B-B step, which we don't.
            vk  = [];
            H   = H0;
            B   = B0;
        else
            vk  = vk/sqrt( vk'*gk );
            if vk'*vk > SR1_max
                vk  = sqrt(SR1_max)/norm(vk)*vk;
            end
            H   = @(x) H0(x) + vk*(vk'*x);
            % The (inverse) secant equation is B*sk = gk(=y), or Hy=s

            % update B (inv(H)) via Sherman-Morrison formula. We need this
            %   for the line search step.
            % B = diagB  - (diagB*vk*vk'*diagB)/( 1 + vk'*diagB*vk );
            %   = diagB  - v*v'  where  v   = diagB.*vk/sqrt( ... );
            v   = B0(vk);
            v   = v/sqrt(1 + vk'*v);
            B   = @(x) B0(x) - v*(v'*x); % O(2n) operations
        end
        stepsizes(nit,2)    = vk'*vk;
        stepsizes(nit,3)    = v'*v;
    else
        H = H0;
        B = B0;
        vk= [];
    end
    
    
    
    
    % ---------------------------------------------------------------------
    % -- Make the proximal update -----------------------------------------
    % ---------------------------------------------------------------------
    xk_old  = xk;
    p       = H(p); 
    
    % Note: the 1/L test has no meaning unless p was proportional
    %   to the gradient. Feb 29.
%     CHECK_MIN_STEPSIZE = KNOWN_L && ~SR1 && ~diag_secantCondition && ~DISABLE_L_WARNING;
    CHECK_MIN_STEPSIZE = KNOWN_L && ~DISABLE_L_WARNING;
        % for SR1, allowing CHECK_MIN_STEPSIZE helps A LOT!!!
%     t_min  = 1/L/t; % scaling by t
    t_min   = 1/L; % we use "B" now in the line search
    t_scaled      = 1;
    
    xk      = prox( yk + p, diagH, vk );
    norm_grad = norm( xk - yk );
    if any(isnan(xk)) 
        stag = Inf; % will cause it to break
        xk   = xk_old;
        myDisp('Prox algorithm failed, probably due to numerical cancellations');
    end
   
    
    % ---------------------------------------------------------------------
    % --- Linesearch ------------------------------------------------------
    % --- Update x, linearizing around y, using search direction from proximal update
    % ---------------------------------------------------------------------
    % Basic update:
    if noLinesearch || isinf(stag)
        % Do nothing: leave scaling as it was.
        t_scaled    = 1;
        % xk is already updated
        cnt         = 0;
    elseif backtrack
        
        % This requires "B", the inverse to H
        
        direction   = (xk-yk)/t_scaled;
        % hence, xk = yk + t_scaled*direction
        
        cntMax = 50;
        q   = fcn(yk);
        for cnt = 1:cntMax

            L_local   = 1/t_scaled;
            
            if CHECK_MIN_STEPSIZE && ( t_scaled < t_min )
                fprintf(fid,'\twarning: linesearch found stepsize (%.1e) below 1/L (%.1e) at step %d, iter %d\n',...
                    t_scaled,t_min,cnt, nit);
                break;
            end
            
            % Linesearch tests to make sure that our stepsize allows
            %   a quadratic upper bound like the Lipschitz upper bound.
            %   Now, set as an option.
            TEST    = backtrackType;
            switch TEST
                case 1
                    % Test if we have a quadratic upper bound to the function
                    F = fcn(xk);
                    Q = q + (xk - yk)'*grad_y + L_local/2*(xk-yk)'*B(xk-yk);
                case 2  % better numerical stability
                    % we need grad_x = A'*(Ax-b),
                    % and compute < x-y, grad_x - grad_y >
                    % but this is equivalent to
                    % < Ax - Ay, Ax - Ay >, = ||Ax-Ay||^2
                    F = (xk-yk)'*( grad(xk) - grad_y );
                    Q   = L_local/2*(xk-yk)'*B(xk-yk); 
                    if Q < 0, myDisp('Q should never be zero! entering debug mode');
                        keyboard;
                    end
            end
            
            if F < Q
                % stepsize was accepted
                break;
            else
                % In this case, if we are too low,
                %   check whether the other TEST would have found it
                if CHECK_MIN_STEPSIZE && ( t_scaled < t_min )
                    switch 3 - TEST
                        case 1
                            F = fcn(xk);
                            Q = q + (xk - yk)'*grad_y + L_local/2*(xk-yk)'*B(xk-yk);
                        case 2
                            F = (xk-yk)'*( grad(xk) - grad_y );
                            Q   = L_local/2*(xk-yk)'*B(xk-yk);
                            if Q < 0, myDisp('Q should never be zero! entering debug mode');
                                keyboard;
                            end
                    end
                    if F < Q
                        % This means the other method said the step was OK
                        % So, we will accept it, and not decrease t_scaled in the future
                        t_scaled   = 1/L_local;
                        break;
                    else
                        fprintf(fid,'\tIter %d, stepsize below 1/L for both linesearchs\n', nit );
                        break;
                    end
                end
                % reduce the stepsize 
                t_scaled = t_scaled*backtrackScalar; % default: 0.8
                
            end
            % "direction" already incorporates t_absolute
            xk      = yk + t_scaled*direction;

        end
        if cnt == cntMax
            if nit == 1
                myDisp('Try setting opts.L0 to a large number');
            end
            myDisp('Backtracking maxed out');
            stag    = Inf;
        end
    end
    % --- End of backtracking loop ----
    t   = t_scaled*t_absolute; % this is rather meaningless in the SR1 case
    
    
    % ---------------------------------------------------------------------
    if any(isnan(xk)) || norm(xk) > 1e10
        xk      = xk_old;
        stag    = Inf;
        myDisp('Prox algorithm failed (NaN), probably due to numerical cancellations');
    end
    
    %--- Compute the new "constant" according to Beck and teboulle
    if ~isempty( fixed_theta )
        theta   = fixed_theta;  % e.g. gradient descent, or over-relaxation
        yk = theta*xk + (1-theta)*xk_old; % convex combination
    else
        theta = (tk-1)/tk; % from FISTA paper
        yk_old = yk;
        yk = xk + theta*(xk - xk_old); % non-convex combination
    end
    % -- restart --
    rs = mod(nit,restart);
    if ~isnan(rs) && ~rs
        tk = 1; % implies that theta goes back to zero
    end
    if nit > 1
        tk = (0.5*(1 + sqrt(1 + 4*(t_old/t_absolute)*tk^2)));  % TFOCS version
    else
        tk = (0.5*(1 + sqrt(1 + 4*tk^2)));  % FISTA update
    end
    
    
    % -- record function values --
    fx  = fcn(xk);
    df  = abs(fx - fxold)/abs(fxold);
    
    
    fxold = fx;
%     norm_grad   = norm(grad_y);
    % We should show norm( gradient_mapping ) instead, since that is more instructive...
    % May 14, I am doing that, using norm_grad = norm(xk-yk) above...
    %   Also adding test for norm_grad < grad_tol, where grad_tol = tol by default
    if backtrack
        t_print = t_scaled;
    else
        t_print = t_absolute;
    end
    printf('Iter: %5d, f: %.3e, df: %.2e, ||grad||: %.2e, step %.2e, backtrack %d\n',...
        nit,fx,df, norm_grad, t_print, cnt);
    
    errStruct(nit,1)    = fx;
    errStruct(nit,2)    = norm_grad;
    errStruct(nit,3)    = t_scaled;
    if ~isempty(errFcn)
        errStruct(nit,4)    = errFcn( xk );
        printf('\b, err %.2e\n', errStruct(nit,4) );
    end
    
    if (df < tol) || ( t_scaled < 1e-10 ) || (isnan(fx) ) || norm_grad < grad_tol
        stag = stag + 1;
    end
    if stag > maxStag
        if VERBOSE, myDisp('Quitting (e.g. reached tolerence)...'); end
        break;
    end
    
end

if nit == nmax && VERBOSE, myDisp('Maxed out iteration limit'); end
if nit < nmax
    errStruct = errStruct( 1:nit, : );
    stepsizes = stepsizes( 1:nit, : );
    printf('Iter: %5d, f: %.3e, df: %.2e, ||grad||: %.2e, step %.2e, backtrack %d\n',...
        nit,fx,df, norm_grad, t_scaled, cnt);
    if ~isempty(errFcn)
        printf('\b, err %.2e\n', errStruct(nit,4) );
    end
end

% ---------------------------------------------------------------------
% Nested functions:
% ---------------------------------------------------------------------
function printf(varargin)
 if VERBOSE
    if VERBOSE > 1 
        if ~rem(nit,VERBOSE)
          fprintf(fid,varargin{:}); 
        end
    else
      fprintf(fid,varargin{:}); 
    end
 end
end


end  % end of main routine
