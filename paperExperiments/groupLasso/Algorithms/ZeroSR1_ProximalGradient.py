import numpy as np
from numpy import zeros, sqrt, sign
import time as clock

def zeroSR1_pg(model, oracle, options, tol, maxiter, check):
    """

    Line-Search Proximal Quasi-Newton algorithm for solving:

        min_{x} h(x);   h(x):= g(x) + f(x)

    Update step:
    
        z^{k}   = argmin_{x} g(x) + <grad f(x^{k}),x-x^{k}> 
                                  + 0.5<(x-x^{k}),B^{k}(x-x^{k})>             
                = prox_{g}^{B^{k}}(x^{k} - H^{k}grad f(x^{k}))                
        x^{k+1} = LineSearch(x^{k} + eta_k*(z^{k} - x^{k}); eta_k)
        
        where LineSearch(x^{k} + eta_k*(z^{k} - x^{k}); eta_k) finds eta_k
        such that
            f(x^{k+1}) <= f(x^{k}) + gamma*eta_k*Delta_k
        with 
            Delta_k = <grad f(x^{k}),z^{k}-x^{k}> + 1/(2*alpha)|z^{k}-x^{k}|_{B^{k}}^2
        by scaling the line search variable eta by delta.


    Properties:
    -----------
    f       continuously differentiable with L-Lipschitz continuous gradient
    g       convex, simple
    alpha   in (0,2/L) or use backtracking
    gamma   in (0,1) Armijo-like parameter
    delta   scaling of the line search variable

    Assumption:                                                                 
    -----------
                                                                                
        y^{k} = B^{k}*s^{k}     (secant equation for f)
                                Holds exactly, when f is quadratic.             
                                                                                
    where:                                                                      
                                                                                
        y^{k}    := grad f(x^{k}) - grad f(x^{k-1})                         
        s^{k}    := x^{k} - x^{k-1}                                             
        B_0      := L*Id                                                        
        B^{k}    := B_0 - \sigma_k*u^{k}*u^{k}'  (Hessian approximation)        
        d^{k}    := B_0*s^k - y^k   [stored in u_k later] 
        u^{k}    := d^{k}/sqrt(<d^{k},s^{k}>)                                   
        \sigma_k := \sign(<d^{k},s^{k}>)                                        
                    \in {-1,0,1}: +1: B^{k} is pos. def. (curvature cond.)      
                                  -1: B^{k} is neg. def. (curvature cond.)      
                                   0: do not use metric                         
                    For quadratic function f: B^{k} is positive semi-definite   
        H^{k}    := (B^{k})^{-1}  (inverse Hessian approximation)               
                  = B_0^{-1} + \sigma u^{k}*u^{k}' /(L*(L-\sigma_k*|u^{k}|^2))  
                    (Using Sherman--Morrison formula)                           
                                                                                
        prox_{g}^{B^{k}} is a diagonal minus \sigma*Rank1 proximal mapping and  
                      requires specialized implementations.     


    Parameter:
    ----------
    model             model data of the optimization problem
    
    oracle:
    .'grad_f'         computes the gradient of the objective grad f(x^{k})
    .'prox_g'         computes the proximal mapping of g
    .'fun_g'          computes the value of g
    .'fun_f'          computes the value of f
    .'residual'       used for breaking condition or resor plots
    
    options (required):
    .'stepsize'       stepsize alpha = 1/L (required)
    .'gamma'          Armijo-like parameter in (0,1)
    .'init'           initialization
    .'eta0'           initialization of line search parameter
    .'delta'          scaling of the line search variable
    .'lineSearchMaxiter'  maximal number of line search trial steps
    
    options (optional):
    .'storeResidual'  flag to store all residual values
    .'storeTime'      flag to store the time of each iteration
    .'storePoints'    flag to store all iterates
    .'storeObjective' flag to store all objective values
    .'storeBeta'      flag to store alle beta values
        
    tol               tolerance threshold for the residual
    maxiter           maximal number of iterations
    check             provide information after 'check' iterations
        
    Return:
    -------
    output
    .'sol'            solution of the problems
    .'seq_res'        sequence of residual values (if activated)
    .'seq_time'       sequence of time points (if activated)
    .'seq_x'          sequence of iterates (if activated)
    .'seq_obj'        sequence of objective values (if activated)
    .'seq_beta'       sequence of beta values (extrapolation parameters)
    .'breakvalue'     code for the type of breaking condition
                      1: maximal number of iterations exceeded
                      2: breaking condition reached (residual below tol)
                      3: not enough backtracking iterations

    """
    
    # store options
    if 'storeResidual'  not in options:
        options['storeResidual']  = False;
    if 'storeTime'      not in options:
        options['storeTime']      = False;
    if 'storePoints'    not in options:
        options['storePoints']    = False;
    if 'storeObjective' not in options:
        options['storeObjective'] = False;
    if 'storeBeta' not in options:
        options['storeBeta'] = False;
    
    # load oracle
    fun_f    = oracle['fun_f'];
    fun_g    = oracle['fun_g'];
    grad_f   = oracle['grad_f'];
    prox_g   = oracle['prox_g'];
    residual = oracle['residual'];

    # load parameter
    Lip               = 1/options['stepsize'];
    tau0              = options['stepsize'];
    gamma             = options['gamma'];
    eta0              = options['eta0'];
    delta             = options['delta'];
    lineSearchMaxiter = options['lineSearchMaxiter'];
    tau_scaling       = 0.8;#95;
    

    # initialization
    x_kp1 = options['init'];
    x_k   = x_kp1.copy();
    z_k   = zeros(x_k.shape);
    s_k   = zeros(x_k.shape);
    y_k   = zeros(x_k.shape);
    u_k   = zeros(x_k.shape);
    one   = np.ones(x_k.shape);
    grad_k   = zeros(x_k.shape);
    grad_km1 = zeros(x_k.shape);
    eta   = eta0;
    sigma_k = 0;
    f_kp1 = fun_f(x_kp1, model, options);
    h_kp1 = f_kp1 + fun_g(x_kp1, model, options);
    res0  = residual(x_kp1, 1.0, model, options);

    # taping
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1);
        seq_res[0] = 1;
    if options['storeTime'] == True:
        seq_time = zeros(maxiter+1);
        seq_time[0] = 0;
    if options['storePoints'] == True:
	seq_x = zeros((model['N'],maxiter+1));        
	seq_x[:,0] = x_kp1;
    if options['storeObjective'] == True:
	seq_obj = zeros(maxiter+1);        
        seq_obj[0] = h_kp1;
    if options['storeBeta'] == True:
	seq_beta = zeros(maxiter);        
    time = 0;

    # solve 
    breakvalue = 1;
    for iter in range(1,maxiter+1):
        
        stime = clock.time();
        
        # update variables
        x_km1    = x_k.copy();
        x_k      = x_kp1.copy();
        grad_km1 = grad_k.copy();
        h_k      = h_kp1.copy();
        f_k      = f_kp1.copy();

        # compute gradient
        grad_k = grad_f(x_k, model, options);
 
        # build rank 1 metric B^{k} = L*id - sigma_k*u^{k}*u^{k}' (Hessian approximation)
        sigma_k = 0;
        tau = tau0;
        if iter>1:
            s_k = x_k - x_km1;
            y_k = grad_k - grad_km1;
            
            # step size selection (tau_BB2 if possible)
            if True: ## use BB2 step size
                nrm_yk = np.dot(y_k.T, y_k);
                if (nrm_yk > 1e-8):
                    tau = tau_scaling*np.dot(s_k.T, y_k)/nrm_yk;

            H0 = tau;

            u_k = s_k - H0*y_k;
            dts = u_k.T.dot(y_k);
            if abs(dts) >= 1e-8:
            #if abs(dts) >= 1e-8*sqrt(sum(y_k**2))*sqrt(sum(u_k**2)):
                sigma_k = sign(dts);
                u_k = u_k/sqrt(abs(dts));
            
            if sigma_k < 0:
                sigma_k = 0;
                breakvalue = 5;

        # forward step (x^{k+1} = x^{k} - (B^{k})^{-1}*\nabla f(x^{k})
        z_k = x_k - tau*grad_k - sigma_k*(u_k.dot(u_k.T.dot(tau*grad_k)))/(1.0/tau-sum(u_k**2));

        # backward step (w.r.t. to the metric B^{k})
        z_k = prox_g(z_k, one/tau, u_k, -1.0*sigma_k, model, options);

        # compute Delta
        Delta = 0;
        if lineSearchMaxiter > 0:
            dx = z_k - x_k;
            Delta = sum(grad_k*dx) + 0.5/tau*sum(dx**2);
        else:
            eta = 1;
            x_kp1 = z_k;

        # line search
        for iterls in range(0,lineSearchMaxiter):

            # trial point
            x_kp1 = x_k + eta*(z_k - x_k);

            # compute new objective value
            h_kp1 = fun_f(x_kp1, model, options) + fun_g(x_kp1, model, options);

            # no backtracking
            if lineSearchMaxiter <= 1:
                break;

            # check backtracking breaking condition
            if (h_kp1 < h_k + eta*gamma*Delta + 1e-8):
                if iterls == 0:
                    #eta = eta/delta;
                    eta = eta0;#eta/delta;
                break;
            else:
                eta = eta*delta;
                if (iterls+1 == lineSearchMaxiter):
                    breakvalue = 3;

        # check breaking condition
        res = residual(x_kp1, res0, model, options);
        if res < tol:
            breakvalue = 2;

        # tape residual
        time = time + (clock.time() - stime);
        if options['storeResidual'] == True:
            seq_res[iter] = res;
        if options['storeTime'] == True:
            seq_time[iter] = time;
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;
        if options['storeObjective'] == True:
            seq_obj[iter] = h_kp1;
        if options['storeBeta'] == True:
            Md = Lip*s_k - y_k
            if iter>1:
                beta = np.dot(Md.T, z_k - x_k)/np.dot(Md.T, s_k);
            else:
                beta = 0;
            seq_beta[iter-1] = beta;

        # print info
        if (iter % check == 0):
            print 'iter: %d, time: %5f, Lip: %f, eta: %f, res: %f' % (iter, time, Lip, eta, res);
        
    
        # handle breaking condition
        if breakvalue == 2:
            print('Tolerance value reached!!!');
            break;
        elif breakvalue == 3:
            print('Not enough backtracking iterations!!!');
            breakvalue = 1;
            #break;
        elif breakvalue == 4:
            print('Metric is not positive definite!');
            break;
        elif breakvalue == 5:
            print('Metric is not positive definite!');


    # return results
    output = {
        'sol': x_kp1,
        'breakvalue': breakvalue
    }

    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
    if options['storeTime'] == True:
        output['seq_time'] = seq_time;
    if options['storePoints'] == True:
        output['seq_x'] = seq_x;
    if options['storeObjective'] == True:
        output['seq_obj'] = seq_obj;
    if options['storeBeta'] == True:
        output['seq_beta'] = seq_beta;

    return output;

