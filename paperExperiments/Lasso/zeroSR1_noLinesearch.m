function [xk,nit, errStruct, defaultOpts, stepsizes] = zeroSR1_noLinesearch(fcn,grad,prox,opts)
% ZEROSR1_NOLINESEARCH Solves smooth + nonsmooth/constrained optimization problems
% [xk,nit, errStruct, outOpts] = zeroSR1_noLinesearch(f,g,proj,opts)
%
% This uses the zero-memory SR1 method (quasi-Newton) to solve:
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
%   .BB  
%       use the Barzilai-Borwein scalar stepsize
%
%   .errFcn can be an arbitrary function that calculates an error metric
%       on the primal variable at every iteration.
%
%
% Output "errStruct" contains three or four columns:
%   (1) objective function
%   (2) norm of gradient
%   (3) stepsize
%   (4) error (i.e. the output of errFcn, if provided)
%   
% Stephen Becker and Jalal Fadili, Nov 24 2011 -- Dec 2012
% Copied from zeroSR1.m Dec 11 2012
%   (zeroSR1.m is the "full version of the code with more bells and 
%    whistles, and also allows Nesterov acceleration and over-relaxation.
%   This version is designed to have more human readable source-code. )
% See also zeroSR1.m



% -----------------------------------------------------------------
% ------------ Boring initializations -----------------------------
% ------------ for understanding the algorithm, skip ahead --------
% ------------ to where it says "Begin algorithm"------------------
% -----------------------------------------------------------------

if nargin == 0 || nargout >= 4
    RECORD_OPTS     = true;
    defaultOpts     = [];
else
    RECORD_OPTS     = false;
end

if nargin < 3 || isempty(prox), prox = @(x,diag,v) x; end
if nargin < 4, opts = []; end 

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
maxStag = setOpts( 'maxStag', 10 ); % force very high accuracy
xk      = setOpts( 'x0', [] );
N       = setOpts( 'N', length(xk) );
if N==0 && nargin > 0, error('for now, must specify opts.N = N'); end
if isempty(xk), xk = zeros(N,1); end

% -- Options that concern the stepsize --
L               = setOpts( 'L', 1, 0 );   % Lipschitz constant, e.g. norm(A)^2
SR1             = setOpts( 'SR1', false );
SR1_diagWeight  = setOpts( 'SR1_diagWeight', 0.8 );
BB              = setOpts( 'BB', SR1 );

if SR1, BB_type = setOpts('BB_type',2);
else, BB_type   = setOpts('BB_type',1); % faster, generally
end
if SR1 && BB_type == 1
    warning('With zero-memory SR1, BB_type must be set to 2. Forcing BB_type = 2 and continuing','zeroSR1:BB_warn');
    BB_type     = 2;
end
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

% ------------ Initializations and such ---------------------------
xk_old  = xk;
gradient  = zeros(N,1);
fxold   = Inf;
t       = 1/L; % initial stepsize
stepsizes = zeros(nmax,1 + SR1); % records some statisics
if ~isempty(errFcn)
    if ~isa(errFcn,'function_handle')
        error('errFcn must be a function');
    end
    errStruct   = zeros( nmax, 4 ); % f, norm(gx), step, err
else
    errStruct   = zeros( nmax, 3 ); % f, norm(gx), step
end
skipBB = false;
stag   = 0;

% -----------------------------------------------------------------
% ------------ Begin algorithm ------------------------------------
% -----------------------------------------------------------------
for nit = 1:nmax

    gradient_old    = gradient;
    gradient        = grad(xk);

    % "sk" and "gk" are the vectors that will give us quasi-Newton
    %   information (and also used in BB step, since that can be
    %   seen as a quasi-Newton method)
    sk      = xk        - xk_old;
    gk      = gradient  - gradient_old;   % this is "yk" in Nocedal/Wright
    if nit > 1 && norm(gk) < 1e-13
        warning('gradient isn''t changing , try changing opts.L','specialSR1:zeroChangeInGradient');
        gk = [];
        skipBB = true;
    end
    
    
    % ---------------------------------------------------------------------
    % -- Find an initial stepsize --
    % ---------------------------------------------------------------------
    t_old   = t;
    if BB && nit > 1 && ~skipBB  
        switch BB_type
            case 1
                t   = (norm(sk)^2)/(sk'*gk); % eq (1.6) in Dai/Fletcher. This is longer
            case 2
                t   = sk'*gk/( norm(gk)^2 ); % eq (1.7) in Dai/Fletcher. This is shorter
        end
        if t < 1e-14 % t < 0 should not happen on convex problem!
            myDisp('Curvature condition violated!');
            stag    = Inf;
        end
        if SR1
            % we cannot take a full BB step, otherwise we exactly satisfy the secant
            %   equation, and there is no need for a rank-1 correction.
            t    = SR1_diagWeight*t; % SR1_diagWeights is a scalar less than 1 like 0.6
        end
        H0      = @(x) t*x;
        diagH   = t*ones(N,1);
    else 
        t       = 1/L;
        H0      = @(x) t*x;         % diagonal portion of inverse Hessian
        diagH   = t*ones(N,1);
    end
    skipBB  = false;
    stepsizes(nit,1) = t;
    
    
    
    % ---------------------------------------------------------------------
    % -- Quasi-Newton -- Requries: H0, and builds H
    % ---------------------------------------------------------------------
    if SR1 && nit > 1 && ~isempty(gk) 
        gs = gk'*sk;
        gHg = gk'*(diagH.*gk);
        if gs < 0,  myDisp('Serious curvature condition problem!'); stag = Inf;  end
        H0  = @(x) diagH.*x;
        vk  = sk - H0(gk);
        if vk'*gk  <= 0
            myDisp('Warning: violated curvature conditions');
            % This should only happen if we took an exact B-B step, which we don't.
            vk  = [];
            H   = H0;
        else
            vk  = vk/sqrt( vk'*gk );
            % And at last, our rank-1 approximation of the inverse Hessian.
            H   = @(x) H0(x) + vk*(vk'*x);
            % The (inverse) secant equation is B*sk = gk(=y), or Hy=s
            % N.B. We can make a rank-1 approx. of the Hessian too; see the full
            % version of the code.
        end
        stepsizes(nit,2)    = vk'*vk;
    else
        H = H0;
        vk= [];
    end
    
    
    % ---------------------------------------------------------------------
    % -- Make the proximal update -----------------------------------------
    % ---------------------------------------------------------------------
    p       = H(-gradient);  % Scaled descent direction. H includes the stepsize
    xk_old  = xk;
    xk      = prox( xk_old + p, diagH, vk ); % proximal step
    norm_grad = norm( xk - xk_old );
    if any(isnan(xk)) || norm(xk) > 1e10
        stag = Inf; % will cause it to break
        xk   = xk_old;
        myDisp('Prox algorithm failed, probably due to numerical cancellations');
    end
    
    % ---------------------------------------------------------------------
    % -- The rest of the code is boring. The algorithmic stuff is done. ---
    % ---------------------------------------------------------------------
    % -- record function values --
    % ---------------------------------------------------------------------
    fx  = fcn(xk);
    df  = abs(fx - fxold)/abs(fxold);
    fxold = fx;
    
    printf('Iter: %5d, f: %.3e, df: %.2e, ||grad||: %.2e, step %.2e\n',...
        nit,fx,df, norm_grad, t);
    
    errStruct(nit,1)    = fx;
    errStruct(nit,2)    = norm_grad;
    errStruct(nit,3)    = t;
    if ~isempty(errFcn)
        errStruct(nit,4)    = errFcn( xk );
        printf('\b, err %.2e\n', errStruct(nit,4) );
    end
    
    if (df < tol) || ( t < 1e-10 ) || (isnan(fx) ) || norm_grad < grad_tol
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
    printf('Iter: %5d, f: %.3e, df: %.2e, ||grad||: %.2e, step %.2e\n',...
        nit,fx,df, norm_grad, t);
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
