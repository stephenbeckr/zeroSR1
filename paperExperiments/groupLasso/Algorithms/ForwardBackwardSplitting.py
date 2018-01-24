import numpy as np
from numpy import zeros
import time as clock

# Forward--backward splitting algorithm
def fbs(model, oracle, options, tol, maxiter, check):
    """

    Optimization problem:
    ---------------------

        min_{x} h(x);   h(x) := g(x) + f(x)

    where:

    f       continuously differentiable with L-Lipschitz continuous gradient
    g       convex, simple

    Algorithm 1:
    ------------
    
        // Initialization 
        x^{0}   = x_init
        // Update for k=0,...,maxiter
        x^{k+1} = prox_{alpha*g}(x^{k} - alpha*grad f(x^{k}))
    
    where:

    alpha   in (0,2/L) or use backtracking


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
    .'stepsize'       stepsize alpha
    .'init'           initialization
    
    options (optional):
    .'storeResidual'  flag to store all residual values
    .'storeTime'      flag to store the time of each iteration
    .'storePoints'    flag to store all iterates
    .'storeObjective' flag to store all objective values
    .'backtrackingMaxiter'  if > 1, then backtracking is performed, which 
                            requires 'backtrackingFactor', otherwise default
                            values are set and fixed step size is used througout
    .'backtrackingFactor'   scaling of the step size when backtracking step
                            is successful or not; value in (0,1)
    
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
    .'breakvalue'     code for the type of breaking event
                      1: maximal number of iterations exceeded
                      2: breaking condition reached (residual below tol)
                      3: not enough backtracking/line sreach iterations
        
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
    
    # backtracking options
    backtrackingMaxiter = 1;   
    backtrackingFactor  = 1.0;
    if 'backtrackingMaxiter' in options:
        backtrackingMaxiter = options['backtrackingMaxiter'];
        backtrackingFactor  = options['backtrackingFactor'];

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
    f_kp1 = fun_f(x_kp1, model, options);
    h_kp1 = f_kp1 + fun_g(x_kp1, model, options);
    res0 = residual(x_kp1, 1.0, model, options);

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
        x_k = x_kp1.copy();
        h_k = h_kp1.copy();
        f_k = f_kp1.copy();

        # compute gradient
        grad_k = grad_f(x_k, model, options);

        for iterbt in range(0,backtrackingMaxiter):

            # forward step
            x_kp1 = x_k - alpha*grad_k;

            # backward step
            x_kp1 = prox_g(x_kp1, alpha, model, options);

            # compute new value of smooth part of objective
            f_kp1 = fun_f(x_kp1, model, options);

            # no backtracking
            if backtrackingMaxiter == 1:
                break;

            # check backtracking breaking condition
            dx = x_kp1 - x_k;
            Delta = sum(grad_k*dx) + 0.5/alpha*sum(dx**2);
            if (f_kp1 < f_k + Delta + 1e-8):
                if iterbt == 0:
                    alpha = alpha/backtrackingFactor;
                break;
            else:
                alpha = alpha*backtrackingFactor;
                if (iterbt+1 == backtrackingMaxiter):
                    breakvalue = 3;
            
        # compute new objective value
        h_kp1 = f_kp1 + fun_g(x_kp1, model, options);

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
        elif breakvalue == 3:
            print('Not enough backtracking iterations!!!');
            break;


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

