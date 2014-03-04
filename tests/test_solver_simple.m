%{
Solve a few simple problems to make sure it works
Solutions are saved in reference_solutions/

For now, we only have 1 test problem

Stephen Becker, March 1 2014
%}

PROBLEM = 1;

switch PROBLEM
    
    case 1
        N   = 12;
        A   = hilb(N);
        b   = ones(N,1);
        lambda = 1e-1;
        
        Q   = A'*A;
        c   = A'*b;
        normQ   = norm(Q);
    
        problemName=sprintf('simple_%03d', PROBLEM );
        % Call this script, which returns variable xRef
        getReferenceSolution;
        nrmXref = norm(xRef);
        errFcn  = @(x) norm( x - xRef )/nrmXref;
        
        % prox        = @(x0,d,u) prox_rank1_l1( x0, d, u, lambda );
        % or, allow 4 arguments, e.g., sigma
        prox        = @(x0,d,u,varargin) prox_rank1_l1( x0, d, u, lambda, [], varargin{:} );
        h           = @(x) lambda*norm(x,1);
        
        % NOTE: the non-standard form (not |Ax-b|, rather <x,Qx> )
        % The "simple" means we do NOT include the lambda term
%         fcnSimple   = @(w) w'*(Q*w)/2 - c'*w;
%         gradSimple  = @(w) Q*w - c; % doesn't include non-smooth portion
%         % for L-BFGS-B, we will add to gradSimple, since we have made new smooth terms
%         fcn         = @(w) fcnSimple(w) + h(w);
        
        % This does all the work for you
%         fcnGrad     = @(x) quadraticFunction(x,Q,c);
        
        % Or this form, which doesn't require Q to be formed
        %   it should be a bit more numerically stable too
        fcnGrad     = @(x) normSquaredFunction(x,A,[],b);

end
    
%% Solve with zeroSR1
opts = struct('N',N,'verbose',25,'nmax',4000,'tol',1e-13);
opts.L      = normQ; % optional
opts.errFcn = errFcn;

%  -- Default values usually fine --
% opts.BB     = true;
% opts.SR1_diagWeight=0.8;

tic
[xk,nit, errStruct,optsOut] = zeroSR1(fcnGrad,[],h,prox,opts);
% -- You can also call it this way, but can be slower --
% [xk,nit, errStruct,optsOut] = zeroSR1(fcnSimple,gradSimple,h,prox,opts);
tm = toc;
solverStr = 'zeroSR1';
fprintf('Final error for %15s is %.2e, took %.2f seconds\n', solverStr, errFcn(xk), tm );
figure(1); clf;
semilogy(errStruct(:,4) );
hold all
emphasizeRecent

%% and same solver but with pure BB, no 0SR1
opts.SR1 = false;
opts.BB_type    = 2;
opts.BB         = true;
[xk,nit, errStruct,optsOut] = zeroSR1(fcnGrad,[],h,prox,opts);
tm2 = toc;
solverStr = 'BB, no linesearch, i.e., basically SPG/SpaRSA';
fprintf('Final error for %15s is %.2e, took %.2f seconds\n', solverStr, errFcn(xk), tm2 );
semilogy(errStruct(:,4) );
hold all
emphasizeRecent
legend('zeroSR1','standard proximal gradient');