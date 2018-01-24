import numpy as np
from numpy import zeros, sqrt, sign
import time as clock

def tseng_zeroSR1_pg(model, oracle, options, tol, maxiter, check):
    """

    Tseng-like Proximal Quasi-Newton algorithm for solving:

        min_{x} h(x);   h(x):= g(x) + f(x)

    Update step: See Section 3.3.2 in 

    P. Ochs and T. Pock: "Adaptive Fista" ArXiv:1711.04343 [Math], November 12, 2017. 


    Properties:
    -----------
    f       convex quadratic function with L-Lipschitz continuous gradient
    g       simple
    alpha   in (0,1/L)

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
    
    options (optional):
    .'storeResidual'  flag to store all residual values
    .'storeTime'      flag to store the time of each iteration
    .'storePoints'    flag to store all iterates
    .'storeObjective' flag to store all objective values
        
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
    .'breakvalue'     code for the type of breaking condition
                      1: maximal number of iterations exceeded
                      2: breaking condition reached (residual below tol)

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
    
    # load oracle
    fun_f    = oracle['fun_f'];
    fun_g    = oracle['fun_g'];
    grad_f   = oracle['grad_f'];
    prox_g   = oracle['prox_g'];
    residual = oracle['residual'];

    # load parameter
    alpha    = options['stepsize'];

    # initialization
    x_kp1 = options['init'];
    x_k   = x_kp1.copy();
    v_kp1 = zeros(x_k.shape);
    z_kp1 = zeros(x_k.shape);
    s_k   = zeros(x_k.shape);
    y_k   = zeros(x_k.shape);
    u_k   = zeros(x_k.shape);
    one   = np.ones(x_k.shape);
    grad_k   = zeros(x_k.shape);
    grad_km1 = zeros(x_k.shape);
    theta_kp1 = 1.0;
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
    time = 0;

    # solve 
    breakvalue = 1;
    for iter in range(1,maxiter+1):
        
        stime = clock.time();
        
        # update variables
        x_k      = x_kp1.copy();
        z_k      = z_kp1.copy();
        h_k      = h_kp1.copy();
        f_k      = f_kp1.copy();
        theta_k = theta_kp1;

        theta_kp1 = 0.5*(sqrt(theta_k**4 + 4.0*theta_k**2)-theta_k**2);

        # extrapolation
        pre_y_k = (1.0-theta_k)*x_k + theta_k*z_k;
        
        # compute FISTA step
        alpha_theta = alpha/theta_k;
        grad_k = grad_f(pre_y_k, model, options);
        z_kp1 = z_k - alpha_theta*grad_k;
        z_kp1 = prox_g(z_kp1, one/alpha_theta, 0.0, 0, model, options);

        # post-combination
        v_kp1 = (1-theta_k)*x_k + theta_k*z_kp1;
        
        # compute gradient
        grad_k = grad_f(x_k, model, options);
        grad_z_k = grad_f(z_k, model, options);
 
        # build rank 1 metric B^{k} = L*id - sigma_k*u^{k}*u^{k}' (Hessian approximation)
        sigma_k = 0;
        if iter>1:
            s_k = x_k - z_k;
            y_k = grad_k - grad_z_k;

            u_k = s_k/alpha - y_k;
            dts = u_k.T.dot(s_k);
            if abs(dts) >= 1e-8:
            #if abs(dts) >= 1e-8*sqrt(sum(y_k**2))*sqrt(sum(u_k**2)):
                sigma_k = sign(dts);
                u_k = u_k/sqrt(abs(dts));
            
            if sigma_k < 0:
                sigma_k = 0;
                breakvalue = 5;

        # forward step (x^{k+1} = x^{k} - (B^{k})^{-1}*\nabla f(x^{k})
        x_kp1 = x_k - alpha*grad_k - sigma_k*(u_k.dot(u_k.T.dot(alpha*grad_k)))/(1.0/alpha-sum(u_k**2));

        # backward step (w.r.t. to the metric B^{k})
        x_kp1 = prox_g(x_kp1, one/alpha, u_k, -1.0*sigma_k, model, options);
        
        # compute new objective value        
        h_xkp1 = fun_g(x_kp1, model, options) + fun_f(x_kp1, model, options);

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

        # print info
        if (iter % check == 0):
            print 'iter: %d, time: %5f, alpha: %f, res: %f' % (iter, time, alpha, res);
        
    
        # handle breaking condition
        if breakvalue == 2:
            print('Tolerance value reached!!!');
            break;
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

    return output;

