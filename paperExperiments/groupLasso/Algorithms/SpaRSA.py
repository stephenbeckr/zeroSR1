import numpy as np
from numpy import zeros
import time as clock

def spg(model, oracle, options, tol, maxiter, check):
    """

    Sparse Reconstruction by Separable Approximation

        min_{x} h(x);   h(x):= g(x) + f(x)

    Update step:
    
        Parameter:
        0 < Lip_min < Lip_max 
        eta > 1
        sigma \in (0,1)

        Choose Lip \in [Lip_min, Lip_max] as BB step-size projected onto
        the given interval, i.e. Lip = <s,y>/<s,s> where s = x^{k} - x^{k-1}
        and y = \nabla f(x^{k}) - \nabla f(x^{k-1}).

        Backtracking w.r.t. Lip of (set Lip = eta*Lip)
            x^{k+1} = prox_{1/Lip*g}(x^{k} - 1/Lip*grad f(x^{k}))
        until 
            h(x^{k+1}) <= \max_{i=0,...,M} h(x^{k-i}) - 
                            0.5*sigma*Lip ||x^{k+1}-x^{k}||^2
    
    
    Properties:
    -----------
    f       continuously differentiable with L-Lipschitz continuous gradient
    g       convex, simple

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
    .'init'           initialization
    
    options (optional):
    .'storeResidual'  flag to store all residual values
    .'storeTime'      flag to store the time of each iteration
    .'storePoints'    flag to store all iterates
    .'storeObjective' flag to store all objective values
    .'Lipschitz_min'  : Lip_min (default: 1e-4)
    .'Lipschitz_max'  : Lip_max (default: 1e10)
    .'backtrackingMaxiter' 
                      if > 1, then backtracking is performed, which 
                      requires 'backtrackingFactor', otherwise default
                      values are set and fixed step size is used througout
                      default: 20
    .'backtrackingFactor' : eta
                      scaling of the step size when backtracking step
                      is successful or not; value eta>1
                      default: 1.1
    .'backtrackingAcceptFactor' : sigma
                      scaling of the sufficient descent term 
    .'backtrackingHistory' : M
                      how many old objective values are stored
                      default: 0
                
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
                      3: not enough backtracking iterations

    Reference:
    ----------
    S.J. Wright, R.D. Nowak, and M.A.T. Figueiredo: "Sparse Reconstruction by 
    Separable Approximation." IEEE Transactions on Signal Processing 57, 
    No. 7:2479--93. 2009.

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
    
    # step size options
    Lip_min = 1e-4;
    Lip_max = 1e10;
    if 'Lipschitz_min' in options:
        Lip_min = options['Lipschitz_min'];
    if 'Lipschitz_max' in options:
        Lip_max = options['Lipschitz_max'];

    # backtracking options
    backtrackingMaxiter = 30;   
    backtrackingFactor = 1.5;
    M = 0;
    sigma = 1e-4;
    if 'backtrackingMaxiter' in options:
        backtrackingMaxiter = options['backtrackingMaxiter'];
    if 'backtrackingFactor' in options:
        backtrackingFactor = options['backtrackingFactor'];
    if 'backtrackingAcceptFactor' in options:
        sigma  = options['backtrackingAcceptFactor'];
    if 'backtrackingHistory' in options:
        M = options['backtrackingHistory'];

    # load oracle
    fun_f    = oracle['fun_f'];
    fun_g    = oracle['fun_g'];
    grad_f   = oracle['grad_f'];
    prox_g   = oracle['prox_g'];
    residual = oracle['residual'];

    # initialization
    Lip = 1.0;      # dummy value here
    x_kp1 = options['init'];
    x_k   = x_kp1.copy();
    f_kp1 = fun_f(x_kp1, model, options);
    h_kp1 = f_kp1 + fun_g(x_kp1, model, options);
    grad_k = grad_f(x_k, model, options);
    res0 = residual(x_kp1, 1.0, model, options);
    hist_h = -1e10*np.ones(M+1);

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
        x_km1 = x_k.copy();
        x_k = x_kp1.copy();
        h_k = h_kp1.copy();
        f_k = f_kp1.copy();
        grad_km1 = grad_k.copy();
        hist_h[iter%(M+1)] = h_kp1;
        max_h = np.amax(hist_h);

        # compute gradient
        grad_k = grad_f(x_k, model, options);

        # compute Barzilai--Borwein step length
        if iter>0:
            s_k = x_k - x_km1;
            y_k = grad_k - grad_km1;
            nrm = np.dot(s_k.T, s_k);
            if nrm>0:
                Lip = np.maximum(Lip_min, np.minimum(Lip_max, \
                            np.dot(s_k.T, y_k)/np.dot(s_k.T, s_k) ));
            else:
                Lip = Lip_max;


        for iterbt in range(0,backtrackingMaxiter):

            # forward step
            x_kp1 = x_k - 1.0/Lip*grad_k;

            # backward step
            x_kp1 = prox_g(x_kp1, 1.0/Lip, model, options);

            # compute new value of smooth part of objective
            h_kp1 = fun_f(x_kp1, model, options) + fun_g(x_kp1, model, options);

            # no backtracking
            if backtrackingMaxiter == 1:
                break;

            # check backtracking breaking condition
            Delta = -0.5*sigma*Lip*sum((x_kp1 - x_k)**2);
            if (h_kp1 < max_h + Delta + 1e-8):
                break;
            else:
                Lip = Lip*backtrackingFactor;
                if (iterbt+1 == backtrackingMaxiter):
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

        # print info
        if (iter % check == 0):
            print 'iter: %d, time: %5f, Lip: %f, res: %f' % (iter, time, Lip, res);
        
    
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

