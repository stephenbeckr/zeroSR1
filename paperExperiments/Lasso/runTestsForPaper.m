%{
For the paper, we ran tests 4 and 5
test 4 is Fig 6.1 (left)  from https://arxiv.org/pdf/1801.08691.pdf
 (similar to Fig 1.a from https://arxiv.org/pdf/1206.1156.pdf )

test 5 is Fig 6.1 (right) from https://arxiv.org/pdf/1801.08691.pdf
 (similar to Fig 1.b from https://arxiv.org/pdf/1206.1156.pdf )

We compare with 3rd party codes, but we don't redistribute their code, so
we have documented where we got their code from and you are free to install
their code and compare.

  -- Stephen Becker, Feb 2018
%}
%% L-BFGS-B
%{
We wrote our own Matlab wrapper for this (using the L-BFGS-B 3.0 Fortran
code). You can download it from: https://github.com/stephenbeckr/L-BFGS-B-C

Unpack it somewhere and run lbfgsb_C/Matlab/compile_mex.m
%}
addpath ~/Repos/lbfgsb_C/Matlab
%% ASA
%{
http://users.clas.ufl.edu/hager/papers/Software/
as of 2013, they have v 3.0
http://users.clas.ufl.edu/hager/papers/CG/Archive/ASA_CG-3.0.tar.gz
but old code is still online at:
http://users.clas.ufl.edu/hager/papers/CG/Archive/ASA_CG-2.2.tar.gz
(old link was http://www.math.ufl.edu/~hager/papers/CG/Archive/ASA_CG-2.2.tar.gz, that's bad now)
You also need the Matlab interface; we wrote this ourself, and it can be 
  downloaded from Mathworks (it will also download the main C source code for you)
https://www.mathworks.com/matlabcentral/fileexchange/35814-mex-interface-for-bound-constrained-optimization-via-asa

If you download the Matlab interface, run the test_ASA.m script
 and it will downlaoad the ASA code that it needs.

%}
addpath('~/Documents/MATLAB/packages/ASA_CG_matlabWrapper');
%% CGIST
%{
Get CGIST from:
  http://tag7.web.rice.edu/CGIST.html
or http://tag7.web.rice.edu/CGIST_files/cgist.zip
%}
addpath('~/Documents/MATLAB/packages/cgist');
%% FPC
%{
Get FPC AS from:
http://www.caam.rice.edu/~optimization/L1/FPC_AS/request-for-downloading-fpc_as.html
%}
addpath('~/Documents/MATLAB/packages/FPC_AS_v1.21/src');
%% L1General and PSSas
%{
 Get the L1General2 code from
   https://www.cs.ubc.ca/~schmidtm/Software/thesis.html
or https://www.cs.ubc.ca/~schmidtm/Software/thesis.zip

Note: you need to compile mex files for this (for the lbfgs subroutine)
For compilation, try: SchmidtThesis/minFunc/mexAll.m 

2018, line 13 in lbfgsC.c, " int nVars,nSteps,lhs_dims[2];"
 With Matlab R2017b, this causes problems. Remove the lhs_dims[2] and add
 a new line with: "  size_t lhs_dims[2];"

%}
addpath ~/Documents/MATLAB/packages/SchmidtThesis/L1General2/
addpath ~/Documents/MATLAB/packages/SchmidtThesis/misc/
addpath ~/Documents/MATLAB/packages/SchmidtThesis/minFunc/ 

%% Setup a problem

randn('state',234213); rand('state',2342343);

% --- fcn setup ---
TEST = 4;
% TEST = 5;
switch TEST
    case 4
        % compressed sensing...
        N       = 3000; % any larger than 5000 and it takes a while to get the norm(A)
        lambda  = .1;
        A       = randn(N/2,N);
        b       = randn(size(A,1),1);
        Q       = A'*A; c = A'*b;
        
    case 5
        % See Fletcher's paper
        n = 13; 
        N = n^3; 
        fprintf('N is %d\n', N );
        lambda  = 1;
        
        I   = eye(n);
        BDG = -( diag( ones(n-1,1), 1 ) + diag( ones(n-1,1), -1 ) );
        T   = 6*I + BDG;
        
        W   = kron( I, T ) + kron( BDG, I );
        Q   = kron( I, W ) + kron( BDG, eye(n^2) );
        
        sigma = 20; a1 = 0.4; a2 = 0.7; a3 = 0.5;
        pdeSol = @(x,y,z) x.*(x-1).*y.*(y-1).*z.*(z-1).*exp( ...
            -.5*sigma^2*( (x-a1).^2 + (y-a2).^2 + (z-a3).^3 ) );
        % Find rhs c = Q*u, where u is the solution above
        h = 1/(n+1);
        grd = h:h:(1-h); % interior points
        [X,Y,Z] = meshgrid(grd);
        u_pde = pdeSol( X, Y, Z );
        c = Q*vec(u_pde);
        fprintf('||c||_inf is %g\n', norm(c,Inf) );
        
        A   = chol(Q); % has small condition number, e.g. 8, and is upper bi-diagonal
        b   = (A')\c;
end

%% More setup

% --- Plotting and such ---

NAMES       = {};
OBJECTIVES  = {};
TIMES       = {};
% -------------------------
if size(A,1) < size(Q,1)
    if issparse(Q), normQ = normest(A*A'); else normQ   = norm(A*A'); end
else
    if issparse(Q), normQ = normest(Q); else normQ   = norm(Q); end
end
lambdaVect = lambda*ones(N,1);
fcn         = @(w) w'*(Q*w)/2 - c'*w + lambda*norm(w,1);

% NOTE: the non-standard form (not |Ax-b|, rather <x,Qx> )
fcnSimple   = @(w) w'*(Q*w)/2 - c'*w;
gradSimple  = @(w) Q*w - c; % doesn't include non-smooth portion
% for L-BFGS-B, we will add to gradSimple, since we have made new smooth terms
    
% for SR1
prox    = @(x0,d,l) prox_l1_rank1( x0, d, l, lambda );

% Setup operators for L-BFGS-B
pos     = @(w) w(1:N,:);
neg     = @(w) w(N+1:2*N,:);
dbl     = @(gg) [gg;-gg];
lambdaVect2     = [lambdaVect;lambdaVect];
fcn2    = @(w) fcnSimple( pos(w) - neg(w) ) + lambdaVect2'*w;
grad2   = @(w) dbl(gradSimple(pos(w)-neg(w))) + lambdaVect2;
    

%% SR1
disp('Solving via SR1 with l1 constraint ...');
% fcn and grad are defined above now...

opts = struct('N',N,'verbose',50,'nmax',4000,'tol',1e-14);
% opts.x0     = .1*ones(N,1); % use this for SR1 versions
% opts.nmax = 5;
opts.BB     = true;
% opts.theta  = []; opts.restart=6; % use [] for FISTA
opts.theta  = 1; opts.SR1 = true; 
opts.SR1_diagWeight=0.8;

opts.L      = normQ;

opts.backtrack = false;

tic
% The code I used for the 2012 tests
% [xk,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);

% Dec '12, try our simplified code:
opts = rmfield(opts,{'theta','backtrack'});
[xk,nit, errStruct,optsOut] = zeroSR1_noLinesearch(fcn,gradSimple,prox,opts);

tm = toc;
NAMES{end+1} = '0-mem SR1';
OBJECTIVES{end+1} = errStruct(:,1);
TIMES{end+1} = tm;   
%% and run our code, but choose FISTA...

opts.BB     = true;
opts.theta  = []; opts.restart=1000; % use [] for FISTA
% opts.theta  = 1; 
opts.SR1 = false; 

opts.backtrack = true;

tic
[xk,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
tm = toc;
NAMES{end+1} = 'FISTA w/ BB'; % with linesearch
OBJECTIVES{end+1} = errStruct(:,1);
TIMES{end+1} = tm;   
%% and run our code, but choose BB...

opts.BB     = true;
opts.theta  = 1;
opts.SR1 = false; 

opts.backtrack = true;

tic
[xk,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
tm = toc;
NAMES{end+1} = 'SPG/SpaRSA'; % with linesearch
OBJECTIVES{end+1} = errStruct(:,1);
TIMES{end+1} = tm; 
%% Run L-BFGS-B
if ~exist('lbfgsb','file')
    disp('Cannot find L-BFGS-B on your path, so skipping this test');
else
%{
Solve min L(x) + lambda*||x||_1 by formulating as:
  min_{z,y} L(z-y) + ones(2N,1)'*[z,y]
s.t.
z,y >= 0. i.e. "x" is z - y
  

if we switch to simple x >= 0 formulation, then it solves it in 2 steps!!

%}
    disp('Solving via L-BFGS-B...');

    tic
    fun     = @(x)fminunc_wrapper( x, fcn2, grad2);
    opts    = struct( 'factr', 1e4, 'pgtol', 1e-12, 'm', 10, 'maxIts', 20000, 'maxTotalIts',1e6 );
    opts.printEvery     = 100;
    opts.factr          = 1e1; % more accurate soln
    if N > 200
        opts.factr = 1e-2;
        opts.pgtol = 1e-14;
    end
    % opts.factr          = 1e7; % default
    [x2, ~, info] = lbfgsb(fun, zeros(2*N,1), inf(2*N,1), opts );
    x   = pos(x2) - neg(x2);
    tm = toc;
    
    NAMES{end+1} = 'L-BFGS-B';
    OBJECTIVES{end+1} = info.err(:,1);
    TIMES{end+1} = tm;
end
%% Run ASA
if ~exist('asa_wrapper','file')
    disp('Cannot find ASA on your path, so skipping this test');
else
    
    % param = struct('A',A,'b',b);
    % param = struct('A',A,'b',b,'lambda',lambda); % No, I am not using this format...
    % an alternative way:
    
    % param = struct('Q',Q,'c',-c,'lambda',lambda);
    param = struct('Q',[Q,-Q;-Q,Q],'c',-[c;-c]+lambdaVect2,'offset',0);
    param.maxits = 1e6;
    
    % if isfield( MAXITS, 'ASA' ) && ~isempty( MAXITS.ASA )
    %     param.maxits = min( param.maxits, MAXITS.ASA );
    % end
    
    % add some options (these are optional). See driver1.c for examples,
    %   and see asa_user.h for all possible values
    [opts,CGopts] = deal(struct('PrintParms',false));
    opts.PrintParms = 0;
    opts.PrintFinal = 1;
    opts.PrintLevel = 0;
    opts.StopFac = 1e-9;
    
    % zero-out the counters
    asa_quadratic_fcnGrad();
    
    lo = zeros(2*N,1);
    hi = inf(2*N,1);
    % x0 = ones(2*N,1);
    x0 = zeros(2*N,1);
    % run the function
    disp('starting...');
    tic
    [x2,status,statistics] = asa_wrapper( x0, lo, hi,'asa_quadratic_fcn',...
        'asa_quadratic_grad', 'asa_quadratic_fcnGrad', opts, CGopts, param);
    tm = toc;
    x   = pos(x2) - neg(x2);
    % View the function values
    [fcnHistory] = asa_quadratic_fcnGrad();
    
    NAMES{end+1} = 'ASA';
    OBJECTIVES{end+1} = fcnHistory;
    TIMES{end+1} = tm;
end
%% Run PSSas and OWN (stuff from L1General toolbox)
if ~exist('L1General2_PSSas','file') || ~exist('L1General2_OWL','file')
    disp('Cannot find PSSas or OWL and L1General on your path, so skipping this test');
else
    
    gOptions = [];
    gOptions.maxIter = 4000;
    gOptions.verbose = 1; % Set to 0 to turn off output
    gOptions.corrections = 10; % for L-BFGS
    gOptions.optTol  = 1e-14;
    gOptions.progTol  = 1e-15;
    
    % funObj     = @(x)fminunc_wrapper( x, fcn, gradSimple);
    funObj     = @(x)fminunc_wrapper( x, fcnSimple, gradSimple,[]);
    %   This works well for error, but not for objective fcn value,
    %   since this is only the smooth portion. So we need to add in
    %   a non-smooth term that gets added just to the history.
    extraFcn   = @(x) lambda*norm(x,1);
    funObj     = @(x)fminunc_wrapper( x, fcnSimple, gradSimple,[],[],extraFcn);
    
    
    w_init = zeros(N,1);
    
    fprintf('\nProjected Scaled Sub-Gradient (Active-Set variant)\n');
    options = gOptions;
    
    fminunc_wrapper();
    tic
    [wk,objectiveValues] = L1General2_PSSas(funObj,w_init,lambdaVect,options);
    tm = toc;
    if isempty( objectiveValues )  % it stopped on first iter
        % do it again, with larger starting guess
        w_init = ones(N,1);
        fminunc_wrapper();
        tic
        [wk,objectiveValues] = L1General2_PSSas(funObj,w_init,lambdaVect,options);
        tm = toc;
    end
    [fcnHistory,errHistory] =fminunc_wrapper();
    NAMES{end+1} = 'PSSas';
    OBJECTIVES{end+1} = fcnHistory;
    TIMES{end+1} = tm;
    
    
    % And re-run for the OWL code
    fminunc_wrapper();
    tic
    wk = L1General2_OWL(funObj,w_init,lambdaVect,options);
    tm = toc;
    [fcnHistory,errHistory] =fminunc_wrapper();
    NAMES{end+1} = 'OWL';
    OBJECTIVES{end+1} = fcnHistory;
    TIMES{end+1} = tm;

end
%% run cgist
if ~exist('cgist','file')
    disp('Cannot find cgist on your path, so skipping this test');
else
    % solves ||Ax-f||^2 + lambda*|x|_1
    % So, from <x,Qx>/2 -c'*x format, we have
    %
    regularizer = 'l1';
    opts = [];
    opts.tol = 1e-8;
    opts.record_objective = true;
    opts.record_iterates = false; % big!
    opts.errFcn = [];
    tic
    [xk, multCount, subgradientNorm, out] = cgist(A,[],b,lambda,regularizer,opts);
    tm = toc;
    % need to subtract norm(b)^2/2 to get objective fcn to line up
    out.objectives = out.objectives - norm(b)^2/2;
    
    NAMES{end+1} = 'CGIST';
    OBJECTIVES{end+1} = out.objectives;
    TIMES{end+1} = tm;
end
%% run FPC-AS
if ~exist('FPC_AS','file')
    disp('Cannot find FPC_AS on your path, so skipping this test');
else
    % v 1.1, 10/2008 Zaiwen Wen
    %
    % For some reason, need to give it some negatives... (-x vs +x)
    
    opts = [];
    opts.gtol = 1e-9;   % a termination option of FPC_AS; see manual
    opts.mxitr = 6e3;
    opts.sub_mxitr = 80; % # of sub-space iterations (max)
    opts.lbfgs_m = 5; % storage
    opts.record = 0; % -1,0,1
    opts.PrintOptions = 0;
    % opts.scale_A = 1;
    M = [];
    % M = 10*eye(N);
    sc = 1;
    tic
    [x, out] = FPC_AS(N,-A/sc,b/sc,lambda/sqrt(sc),M,opts);
    tm = toc;
    out.fcnHist = out.fcnHist - norm(b)^2/2;
    NAMES{end+1} = 'FPC-AS';
    OBJECTIVES{end+1} = out.fcnHist;
    TIMES{end+1} = tm;

end
%% PLOT EVERYTHING
figure(1); clf;

obj_best = Inf;
for k = 1:length(OBJECTIVES)
    obj_best = min(obj_best, min( OBJECTIVES{k}) );
end
    
for k = 1:length(NAMES)
    tGrid = linspace(0,TIMES{k},length(OBJECTIVES{k}));
    h=semilogy( tGrid, cummin( OBJECTIVES{k} - obj_best)  );
    
    set(h,'linewidth',2);
    
    hold all
end
legend(NAMES)
xlabel('time in seconds','fontsize',18);
ylabel('objective value error','fontsize',18);
set(gca,'fontsize',18)
 switch TEST
     case 4
         title('Fig 6.1 (left) from https://arxiv.org/pdf/1801.08691.pdf');
         xlim([0,110]);
         ylim([1e-8,1e4]);
     case 5
         title('Fig 6.1 (right) from https://arxiv.org/pdf/1801.08691.pdf');
         xlim([0,2.5]);
         ylim([1e-8,1e9]);
 end
