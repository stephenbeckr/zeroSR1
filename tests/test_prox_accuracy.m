%{
Tests the accuracy of several prox operators
This will run a random assortment of tests
    The reference solution is computed with "solution_via_cvx"
    which uses CVX (http://cvxr.com/).
    If you don't have CVX installed, then it won't work

    In the future, one could make a predefined test set and then
    precompute the answers so that CVX is not necessary...

Stephen Becker, Feb 26 2014  stephen.beckr@gmail.com
%}

% run setup_zeroSR1.m if you haven't already
clear; clc;

nTests  = 100;
n       = 1e2; % dimension of the problem

myQuadForm = @(x,V) x'*(V*x);
for test = 1:nTests
    % Make a random problem
    d = rand(n,1);
    u = 10*randn(n,1);
    y = randn(n,1);
    offset  = randn(n,1);
    lwr     = randn(n,1); % used for the box constraints
    upr     = lwr + 2*rand(n,1);
    lwr(randi(n)) = -Inf;
    upr(randi(n)) = Inf;
    lambda  = randn(n,1);
    
    % And sometimes turn off these features
    if randn(1) > 0, lambda = []; end
    if randn(1) > 0, offset = []; end
    if randn(1) > 0, d(2:end) = d(1); end
    if randn(1) > 0, u = 0; end % i.e., normal prox!
    sigma = 1;
    
    % Pick a solver at random
    solverTypes = {'l1','l1pos','Rplus','box','hinge','linf'};
    type = solverTypes{ randi(length(solverTypes)) };
    
    INVERT  = sign( randn(1) )+1; % sometimes specify V, sometimes specify inv(V)
    
    if isempty(lambda), lambda = 1; end
    if isempty(offset), offset = zeros(n,1); end
    INFEASIBLE  = 1e14;
    EPS         = 1e-13; % feasiblity tolerance
    switch lower(type)
        case 'l1'
            [x_cvx,V]   = solution_via_cvx('l1',y,d,u,lambda,offset,[],[],sigma,INVERT);
            obj = @(x) norm(lambda.*x,1) + 1/2*myQuadForm(x-y,V) + dot(offset,x);
            % If we use prox_rank1_generic
            prox            = @(x,t) sign(x).*max(0, abs(x) - t );
            prox_brk_pts    = @(s) [-s,s];
            % or, use
            % scaledProx = @prox_rank1_l1;
        case 'l1pos'
            [x_cvx,V]   = solution_via_cvx('l1pos',y,d,u,lambda,offset,[],[],sigma,INVERT);
            obj = @(x) norm(lambda.*x,1) + 1/2*myQuadForm(x-y,V) + dot(offset,x) + INFEASIBLE*any( lambda.*x < -EPS );
            % If we use prox_rank1_generic
            prox            = @(x,t) max(0, x - t );
            prox_brk_pts    = @(s) [s];
        case 'rplus'
            [x_cvx,V]   = solution_via_cvx('Rplus',y,d,u,lambda,offset,[],[],sigma,INVERT);
            obj = @(x) 1/2*myQuadForm(x-y,V) + dot(offset,x) + INFEASIBLE*any( lambda.*x < -EPS );
            prox            = @(x,t) max(0, x);
            prox_brk_pts    = @(s) 0; % since projection, scaling has no effect
            % scaledProx = @proj_rank1_Rplus;
        case 'box'
            [x_cvx,V]   = solution_via_cvx('box',y,d,u,lambda,offset,lwr,upr,sigma,INVERT);
            obj = @(x) 1/2*myQuadForm(x-y,V) + dot(offset,x) + ...
                INFEASIBLE*any( lambda.*x < lwr-EPS | lambda.*x > upr+EPS );
            prox            = @(x,t) max( min(upr,x), lwr );
            prox_brk_pts    = @(s) [lwr,upr]; % since projection, scaling has no effect
            % scaledProx = @(varargin)proj_rank1_box(lwr,upr,varargin{:});
        case 'hinge'
            hinge = @(x) sum(max(0,1-lambda.*x));
            [x_cvx,V]   = solution_via_cvx('hinge',y,d,u,lambda,offset,[],[],sigma,INVERT);
            obj = @(x) hinge(x) + 1/2*myQuadForm(x-y,V) + dot(offset,x);
            prox    = @(x,t) 1 + (x-1).*( x > 1 ) + (x + t - 1).*( x + t < 1  );
            prox_brk_pts    = @(s)[ones(size(s)), 1-s];
            % scaledProx = @prox_rank1_hinge;
        case 'linf'
            [x_cvx,V]   = solution_via_cvx('linf',y,d,u,lambda,offset,[],[],sigma,INVERT);
            obj = @(x) INFEASIBLE*(norm(lambda.*x,Inf)>1+EPS) + 1/2*myQuadForm(x-y,V) + dot(offset,x);
            prox            = @(x,t) sign(x).*min( 1, abs(x) );
            prox_brk_pts    = @(s) [-ones(size(s)),ones(size(s))]; % since projection, scaling has no effect
            % scaledProx = @proj_rank1_linf;
    end
    if all(lambda==1), lambda = []; end % turn off the feature
    if all(offset==0), offset = []; end % turn off the feature
    
    scaledProx = @(varargin) prox_rank1_generic( prox, prox_brk_pts, varargin{:});
    x = scaledProx( y, d, u, lambda, offset, sigma, INVERT);
    
    if any(isinf( x_cvx ))
        % This means either CVX is not installed or this
        % is running in Octave.
        fprintf('Test %d/%d. Solver type %s. CVX solution not available\n', ...
            test, nTests, type );
        if obj(x) > INFEASIBLE
            fprintf(2,'\tSolution is not feasible! Maybe due to roundoff?\n');
            break;
        end
        if isnan(x)
            fprintf(2,'\tError detected!\n');
            break;
        end
    else
    
        fprintf('Test %d/%d. Solver type %s. Error is %.2e\n', ...
            test,nTests, type, norm( x - x_cvx )/max(1e-5,norm(x_cvx)) );
        fprintf('\tObjective is %.2e, for cvx is %.2e, obj(x) - obj(x_cvx) is %.2e\n', ...
            obj(x), obj(x_cvx), obj(x)-obj(x_cvx) );
        TOLERANCE1 = 1e-3;
        TOLERANCE2 = 1e-6;
        if any(isnan(x_cvx))
            if any(isnan(x))
                fprintf(2,'\tBoth solutions are NaN. Hmmm...\n');
            else
                fprintf(2,'\tCVX returned NaN, our solver did not.\n');
                if obj(x) > INFEASIBLE/2
                    fprintf(2,'\tSolution is not feasible! Maybe due to roundoff?\n');
                    break;
                end
            end
        else
            if obj(x_cvx) > INFEASIBLE/2
                fprintf(2,'\tCVX solution is not feasible!\n');
            end
            if obj(x) > INFEASIBLE/2
                fprintf(2,'\tOur solution is not feasible! Maybe due to roundoff?\n');
                break;
            end
            if (obj(x)-obj(x_cvx))/max(1,abs(obj(x_cvx)))   < TOLERANCE2
                fprintf(2,'\tGOOD\n');
            elseif (obj(x)-obj(x_cvx))/max(1,abs(obj(x_cvx)))   < TOLERANCE1
                fprintf(2,'\tMARGINAL -- Loss of accuracy\n');
            else
                fprintf(2,'\tBAD\n');
                break;
            end
        end
    end
end