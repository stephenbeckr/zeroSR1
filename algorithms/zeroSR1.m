function [xk,nit, errStruct, defaultOpts, stepsizes] = zeroSR1(fcn,grad,h,prox,opts)
% ZEROSR1 Solves smooth + nonsmooth/constrained optimization problems
% [xk,nit, errStruct, outOpts] = zeroSR1(f,grad_f,h,prox_h,opts)
%
% This uses the zero-memory SR1 method (quasi-Newton) to solve:
% 
%   min_x f(x)  + h(x)
%
% where
%   'f' calculates f(x), 'grad_f' calculates the gradient of f at x,
%   and h(x) is a non-smooth term that can be infinite-valued (a constraint),
%   so long as you present a function 'prox' that computes diagional plus
%   rank-1 projections. The 'prox' function should accept at least three inputs:
%
%   if 'grad_f' is empty, then we assume the 'f' function is actually
%   computing both f and grad_f (e.g., just f if nargout=1, and
%   f and grad_f if nargout=2). This method is often preferable
%   since you can re-use computation
%
%   'h' is the non-smooth function, and prox_h is a function with
%   3 or 4 inputs that returns:
%       y = prox_h( x0 , d, v, ) 
% where
%       y = argmin_x h(x) + 1/2||x-x0||^2_B 
% and
%       B = inv(H) = inv( diag(D) + v*v' )
% or, for the case with 4 arguments, y = prox_h( x0, d, v, sigma )
%   then B = inv( diag(D) + sigma*v*v' ) where sigma should be +1 or -1
%   The 4 argument case only matters when opts.SR1=true and opts.BB_type=1
%   or opts.SR1=true, opts.BB_type=1 and opts.SR1_diagWeight > 1
%
% If 'prox_h' isn't provided or is [], it defaults to the identity mapping, which corresponds
%   to the case when h=0.
%
% 'prox_h' is mean to be given by something like prox_rank1_l1
% e.g., 
%   prox        = @(x0,d,v) prox_rank1_l1( x0, d, v, lambda );
%   or, for 4 arguments,
%   prox        = @(x0,d,v,varargin) prox_rank1_l1( x0, d, v, lambda, [], varargin{:} );
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
%       size of primal domain (only necessary if x0 wasn't provided)
%
%   .SR1  if true, uses the zero-memory SR1 method (default)
%         if false, uses gradient descent/forward-backward method
%         (or variant, such as BB stepsizes as in the SPG method)
%   .SR1_diagWeight is a scalar > 0 that controls the weight of the 
%           BB stepsize, and is usually between 0 and 1.
%           If set to exactly 1, then the rank 1 term is exactly zero
%   .BB  
%       use the Barzilai-Borwein scalar stepsize (by default, true)
%       .BB_type = 1 uses the longer of the B-B steps
%       .BB_type = 2 uses the shorter of the steps
%       with 0SR1, BB_type=1 is not possible
%       BB_type=2 is used, and is scaled by 0 < opts.SR1_diagWeight < 1
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
%   Feb 28 2014, unnesting all functions to make compatible with octave.
%
% See also proximalGradient.m



% -----------------------------------------------------------------
% ------------ Boring initializations -----------------------------
% ------------ for understanding the algorithm, skip ahead --------
% ------------ to where it says "Begin algorithm"------------------
% -----------------------------------------------------------------

if nargin == 0 || nargout >= 4
    RECORD_OPTS     = true;
%     defaultOpts     = [];
else
    RECORD_OPTS     = false;
end

if nargin < 3 || isempty(h)
    if nargin >= 4 && ~isempty(prox)
        warning('zeroSR1:h_not_provided','Found prox_h but not h itself. Setting h=0, prox=I');
        prox = @(x,varargin) x;
    end
    h = @(x) 0;
end
if nargin < 4 || isempty(prox), prox = @(x,varargin) x; end
if nargin < 5, opts = []; end 

setOptsSubFcn(); % zero out any persistent variables
setOpts     = @(varargin) setOptsSubFcn( RECORD_OPTS, opts, varargin{:} );
% Usage: setOpts( field, default, mn, mx, emptyOK (default:false) );

fid     = setOpts('fid', 1 );      % print output to the screen or a file
myDisp  = @(str)  fprintf(fid,'%s\n', str );
tol     = setOpts( 'tol', 1e-6 );  
grad_tol= setOpts( 'grad_tol', tol );
nmax    = setOpts( 'nmax', 1000 );  
errFcn  = setOpts( 'errFcn', [] );
VERBOSE = setOpts( 'verbose', false );
if isinf(VERBOSE), VERBOSE = false; end
maxStag = setOpts( 'maxStag', 10 ); % force very high accuracy
xk      = setOpts( 'x0', [], [], [], true );
N       = setOpts( 'N', length(xk) );
if N==0 && nargin > 0, error('for now, must specify opts.N = N'); end
if isempty(xk), xk = zeros(N,1); end
damped  = setOpts('damped',false); % 1=no damping, .01 = very tiny step

% -- Options that concern the stepsize --
SR1             = setOpts( 'SR1', true );
BFGS            = setOpts( 'BFGS', false );
if SR1 && BFGS
    error('zeroSR1:conflictingArgs','Cannot set SR1 and BFGS to both be true');
end
BB              = setOpts( 'BB', SR1 || BFGS );
if isfield(opts,'L') && isempty(opts.L) && ~BB
    warning('zeroSR1:noGoodStepsize','Without Lipschitz constant nor BB stepsize nor line search, bad things will happen');
end
L               = setOpts( 'L', 1, 0 );   % Lipschitz constant, e.g. norm(A)^2

SIGMA           = +1; % used for SR1 feature
% Default BB stepsize. type "1" is longer and usually faster
BB_type = setOpts('BB_type',2*(SR1||BFGS) + 1*(~(SR1||BFGS)) );
if (SR1||BFGS) && BB_type == 1
%     warning('zeroSR1:badBB_parameter','With zero-memory SR1, BB_type must be set to 2. Forcing BB_type = 2 and continuing');
%     BB_type     = 2;

    warning('zeroSR1:experimental','With zero-memory SR1, BB_type=1 is an untested feature');
    SIGMA       = -1;
end
if SR1
    defaultWeight = 0.8*(BB_type==2) + 1.0*(BB_type==1);
else
    defaultWeight = 1;
end
SR1_diagWeight  = setOpts( 'SR1_diagWeight', defaultWeight );
if SR1 && BB_type == 2 && SR1_diagWeight > 1
    SIGMA       = -1;
end

% ------------ Scan options for capitalization issues, etc. -------
[defaultOpts,opts] = setOpts();
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
% gradient  = zeros(N,1);
getGradient     = @(varargin) getGradientFcn(fcn,grad, varargin{:});
fxold   = Inf;
t       = 1/L; % initial stepsize
stepsizes = zeros(nmax,1 + (SR1||BFGS)); % records some statisics
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


gradient        = getGradient(xk);
gradient_old    = gradient;
f_xk            = [];

% -----------------------------------------------------------------
% ------------ Begin algorithm ------------------------------------
% -----------------------------------------------------------------
for nit = 1:nmax

    % Do this at end now, so we can get fcn value for free
%     gradient_old    = gradient;
%     gradient        = grad(xk);

    % "sk" and "yk" are the vectors that will give us quasi-Newton
    %   information (and also used in BB step, since that can be
    %   seen as a quasi-Newton method)
    sk      = xk        - xk_old;
    yk      = gradient  - gradient_old;   % Following notation in Nocedal/Wright
    if nit > 1 && norm(yk) < 1e-13
        warning('zeroSR1:zeroChangeInGradient','gradient isn''t changing , try changing opts.L');
        yk = [];
        skipBB = true;
    end
    
    
    % ---------------------------------------------------------------------
    % -- Find an initial stepsize --
    % ---------------------------------------------------------------------
%     t_old   = t;
    if BB && nit > 1 && ~skipBB  
        switch BB_type
            case 1
                t   = (norm(sk)^2)/(sk'*yk); % eq (1.6) in Dai/Fletcher. This is longer
            case 2
                t   = sk'*yk/( norm(yk)^2 ); % eq (1.7) in Dai/Fletcher. This is shorter
        end
        if t < 1e-14 % t < 0 should not happen on convex problem!
            myDisp('Curvature condition violated!');
            stag    = Inf;
        end
        if SR1 || BFGS
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
    if SR1 && nit > 1 && ~isempty(yk) 
        gs = yk'*sk;
%         gHg = yk'*(diagH.*yk); % not needed any more
        if gs < 0
            myDisp('Serious curvature condition problem!');
            stag = Inf;  
        end
        H0  = @(x) diagH.*x;
        vk  = sk - H0(yk);
        vkyk    = vk'*yk;
        SIGMA_LOCAL = sign( vkyk );
        %if SIGMA*vkyk  <= 0
        if SIGMA_LOCAL*vkyk  <= 0
            myDisp('Warning: violated curvature conditions');
            % This should only happen if we took an exact B-B step, which we don't.
            vk  = [];
            H   = H0;
            stepsizes(nit,2)    = 0;
        else
            vk  = vk/sqrt( SIGMA_LOCAL*vkyk );
            % And at last, our rank-1 approximation of the inverse Hessian.
            H   = @(x) H0(x) + SIGMA_LOCAL*(vk*(vk'*x));
            % The (inverse) secant equation is B*sk = yk(=y), or Hy=s
            % N.B. We can make a rank-1 approx. of the Hessian too; see the full
            % version of the code.
            
            stepsizes(nit,2)    = vk'*vk;
        end
    elseif BFGS && nit > 1 && ~isempty(yk) 
        gs = yk'*sk;
        rho= 1/gs;
        if gs < 0
            myDisp('Serious curvature condition problem!');
            stag = Inf;  
        end
        H0  = @(x) diagH.*x;
        
        tauBB   = sk'*yk/( norm(yk)^2);
        uk      = sk/2 + H0(sk)/(2*tauBB) - H0(yk);
        % if H0 is tauBB*I (e.g., gamma=1), then vk = sk - H0(yk).
        
        
        stepsizes(nit,2)    = uk'*uk;
        
        vk      = [sk-uk, sk+uk]*sqrt(rho/2); % rank 2!
        SIGMA_LOCAL = [-1,1];
        
        H   = @(x) H0(x) + vk*( diag(SIGMA_LOCAL)*(vk'*x) );
        
        %fprintf('DEBUG: %.2e\n', norm( H(yk) - sk )  );
        
    else
        SIGMA_LOCAL     = SIGMA;
        H = H0;
        vk= [];
    end
    
    
    % ---------------------------------------------------------------------
    % -- Make the proximal update -----------------------------------------
    % ---------------------------------------------------------------------
    p       = H(-gradient);  % Scaled descent direction. H includes the stepsize
    xk_old  = xk;
    if ~isequal(SIGMA_LOCAL,1)
        if damped
            xk      = xk + damped*(prox( xk_old + p, diagH, vk, SIGMA_LOCAL )-xk);
        else
            xk      = prox( xk_old + p, diagH, vk, SIGMA_LOCAL );
        end
    else
        if damped
            xk      = xk + damped*(prox( xk_old + p, diagH, vk )-xk);
        else
            xk      = prox( xk_old + p, diagH, vk ); % proximal step
        end
        
    end
    
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
    gradient_old    = gradient;
    [gradient,f_xk]  = getGradient(xk); % can be cheaper if user provided a nice fcn
    fx  = f_xk + h(xk);
%     fx  = fcn(xk) + h(xk);
    df  = abs(fx - fxold)/abs(fxold);
    fxold = fx;
    
    if (df < tol) || ( t < 1e-10 ) || (isnan(fx) ) || norm_grad < grad_tol
        stag = stag + 1;
    end
    
    if VERBOSE && (~rem(nit,VERBOSE) || stag>maxStag )
        fprintf(fid,'Iter: %5d, f: % 7.3e, df: %.2e, ||grad||: %.2e, step %.2e\n',...
            nit,fx,df, norm_grad, t);
    end
    
    errStruct(nit,1)    = fx;
    errStruct(nit,2)    = norm_grad;
    errStruct(nit,3)    = t;
    if ~isempty(errFcn)
        errStruct(nit,4)    = errFcn( xk );
        if VERBOSE && (~rem(nit,VERBOSE) || stag>maxStag )
            fprintf(fid,'\b, err %.2e\n', errStruct(nit,4) );
        end
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
end

end  % end of main routine

function [gradientValue,fcnValue] = getGradientFcn( fcn, gradient, x, str )
% The user can either specify fcn and gradient separately,
%   or they can specify them both in a single function (also called fcn)
% This latter option is triggered whenever gradient=[]
if nargin < 4, str = []; end
if isempty(gradient)
    [fcnValue,gradientValue]    = fcn(x);
else
    gradientValue               = gradient(x);
    if nargout > 1 
        if strcmpi(str,'fcn_optional')
            fcnValue = [];
        else
            fcnValue  = fcn(x);
        end
    end
end
end

function varargout = setOptsSubFcn(RECORD_OPTS, opts, field, default, mn, mx, emptyOK )
    persistent defaultOpts
    persistent updatedOpts
    if nargin <= 2
        % non-standard usage
        varargout{1} = defaultOpts;
        varargout{2} = updatedOpts;
        defaultOpts = [];
        updatedOpts = [];
        return;
    end
    if isempty( updatedOpts ), updatedOpts = opts; end
        
    % if emptyOK is false, then values of opts.field=[] are not allowed and
    %   are instead set to the default value
    if nargin < 7 || isempty(emptyOK), emptyOK = false; end
    if ~isfield( opts, field ) || (isempty(opts.(field)) && ~emptyOK )
        opts.(field)    = default;
    end
    out = opts.(field);
    varargout{1} = out;
    if nargin >= 5 && ~isempty(mn) && any(out < mn), error('Value is too small'); end
    if nargin >= 6 && ~isempty(mx) && any(out > mx), error('Value is too large'); end
    if isfield( updatedOpts, field )
        updatedOpts    = rmfield( updatedOpts, field ); % so we can do a check later
    end
    if RECORD_OPTS
        defaultOpts.(field) = out;
    end
end
