from mymath import *


################################################################################
### Lasso problem ##############################################################
#                                                                              #
# Optimization problem:                                                        #
#                                                                              #
#     min_x  0.5*|Ax-b|_2^2 + mu*|x|_B                                         #
#                                                                              #
# where                                                                        #
#     |x|_B = sum_{I\in B} |x_{i\in I}|_2                                      # 
#                                                                              #
# Model:                                                                       #
#     A       M x N matrix                                                     #
#     b       M vector                                                         #
#     mu      positive parameter                                               #
#     N       dimension of the optimization variable                           #
#                                                                              #
################################################################################


def Model():
    """
    Define the model data of the problem to be solved in this project.

    Returns:
    --------
    struct
    .'A'      M x N matrix
    .'b'      M vector
    .'M'      Matrix dimension 1
    .'N'      Matrix dimension 2
    .'mu'     positive regularization weight
    .'B'      defines the group structure (list of start and end-indizes of 
              coordinates in the same group.

    """

    M = 1600;
    N = 2500;
    mu = 1.0;

    A = rand(M,N);
    b = rand(M,1);

    # define groups of coordinates
    maxK = 12;  # maximal group size
    k = 0;
    B = [0];
    while (k < N):
        K = pmax(1,np.asscalar(np.ceil(maxK*rand(1)))).astype(int);
        k = np.minimum(N,k + K);
        B.append(k);
        
    print "l21-block sparsity: ", B    

    return {'M': M, 'N': N, 'mu': mu, 'A': A, 'b': b, 'B': B};


# generate problem
np.random.seed(2305978);
model = Model();

compute_optimal_value = False; # Should be run first to get nice plots / switch off then


################################################################################
### Define problem specific oracles ############################################


### zero order oracles
def lasso_objective(x, model, options):
    A = model['A'];
    b = model['b'];
    mu = model['mu'];
    B = model['B'];
    reg = 0.0;
    for k in range(0,len(B)-1):
        reg = reg + sqrt(sum(x[B[k]:B[k+1]]**2));
    return 0.5*sum((A.dot(x) - b)**2) + mu*reg;
def lasso_objectiveSmooth(x, model, options):
    A = model['A'];
    b = model['b'];
    return 0.5*sum((A.dot(x) - b)**2);
def lasso_objectiveNonSmooth(x, model, options):
    mu = model['mu'];
    B = model['B'];
    reg = 0.0;
    for k in range(0,len(B)-1):
        reg = reg + sqrt(sum(x[B[k]:B[k+1]]**2));
    return mu*reg;

### first order oracles
def lasso_gradient(x, model, options): 
    A = model['A'];
    b = model['b'];
    return A.T.dot(A.dot(x) - b);

### prox operators
def lasso_prox(x, tau, model, options):
    mu  = model['mu'];
    B   = model['B'];
    return prox_groupl2l1(x, 1.0/(tau*mu), {'B':B});

### rank1 prox operators
def lasso_prox_rank1_c(x, D, u, sigma, model, options):
    mu  = model['mu'];
    B   = model['B'];

    if options.has_key('prox_a_init'):
        params = {'B': B, 'prox_a_init':options['prox_a_init']};
    else:
        params = {'B':B, 'prox_a_init':np.array(0.0)};

    if mu == 0:
        return x;
    elif sigma==0:
        return prox_groupl2l1(x, D/mu, {'B':B});
    else:
        x = cprox_rk1_groupl2l1(x,D/mu,u/sqrt(mu),sigma,params);
        options['prox_a_init'] = params['prox_a_init'];
        return x;

### residuals for breaking conditions
if compute_optimal_value: 
    def lasso_residual(x, res0, model, options):
        return lasso_objective(x, model, options)/res0;
else:    # if numeric solution is available
    # load solution
    h_sol = np.load('data_group_lasso.npy');
    def lasso_residual(x, res0, model, options):
        return (lasso_objective(x, model, options) - h_sol);



################################################################################
### Run Algorithms #############################################################
from Algorithms.ForwardBackwardSplitting import *;
from Algorithms.FISTA import *;
from Algorithms.ZeroSR1_ProximalGradient import *;
from Algorithms.MFZeroSR1_ProximalGradient import *;
from Algorithms.TsengZerosSR1_ProximalGradient import *;
from Algorithms.SpaRSA import *;

# general parameter
maxiter = 1200;
check = 200;
tol = -1;

# auxilliary variables
A = model['A'];
Lip = np.linalg.norm(A.T.dot(A))+1e-3;

# initialization
x0 = zeros((model['N'],1));

# taping
xs = [];
rs = [];
ts = [];
cols = [];
legs = [];
nams = [];

# turn algorithms to be run on or off
run_fbs          = 1;    # Forward--backward splitting
run_fista        = 1;    # FISTA
run_zeroSR1c     = 1;    # Zero SR1 Proximal Quasi-Newton (with rank-1 prox in c)
run_mfzeroSR1    = 1;    # Monotone Fast Zero SR1 Proximal Quasi-Newton
run_tseng_zeroSR1= 1;    # Tseng Fast Zero SR1 Proximal Quasi-Newton
run_SpaRSA       = 1;    # Sparse Reconstruction by Separable Approximation

if compute_optimal_value: # optimal solution is computed using FISTA
    maxiter = 50000
    check = 1000;
    run_fbs          = 0; 
    run_fista        = 1; 
    run_mfista       = 0; 
    run_zeroSR1c     = 0; 
    run_mfzeroSR1    = 0; 
    run_tseng_zeroSR1= 0; 
    run_SpaRSA       = 0; 
    

################################################################################
if run_fbs: 
    
    print('');
    print('********************************************************************************');
    print('*** Forward--Backward Splitting ***');
    print('***********************************');

    options = {
        'init':           x0,
        'stepsize':       1.0/Lip,
        'storeResidual':  True,
        'storeTime':      True
    }
    oracle = {
        'fun_f':    lasso_objectiveSmooth,
        'fun_g':    lasso_objectiveNonSmooth,
        'grad_f':   lasso_gradient,
        'prox_g':   lasso_prox,
        'residual': lasso_residual
    }
    
    output = fbs(model, oracle, options, tol, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0,1,1));
    legs.append('FBS');
    nams.append('FBS');



################################################################################
if run_fista: 
    
    print('');
    print('********************************************************************************');
    print('*** FISTA ***');
    print('*************');

    options = {
        'init':           x0,
        'stepsize':       1.0/Lip,
        'storeResidual':  True,
        'storeTime':      True
    }
    oracle = {
        'fun_f':    lasso_objectiveSmooth,
        'fun_g':    lasso_objectiveNonSmooth,
        'grad_f':   lasso_gradient,
        'prox_g':   lasso_prox,
        'residual': lasso_residual
    }
    
    output = fista(model, oracle, options, tol, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0.95,0,1));
    legs.append('FISTA');
    nams.append('FISTA');

    if compute_optimal_value:
        h_sol = lasso_objective(xs[-1], model, options);
        np.save('data_group_lasso.npy', h_sol);
    

################################################################################
if run_zeroSR1c: 
    
    print('');
    print('********************************************************************************');
    print('*** Zero SR1 Proximal Quasi-Newton Method ***');
    print('*********************************************');

    options = {
        'init':              x0,
        'stepsize':          2.0/Lip,
        'gamma':             1e-4,
        'delta':             0.7,
        'eta0':              1.0,
        'lineSearchMaxiter': 20,
        'storeResidual':     True,
        'storeTime':         True
    }
    oracle = {
        'fun_f':    lasso_objectiveSmooth,
        'fun_g':    lasso_objectiveNonSmooth,
        'grad_f':   lasso_gradient,
        'prox_g':   lasso_prox_rank1_c,
        'residual': lasso_residual
    }
    
    output = zeroSR1_pg(model, oracle, options, tol, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.2,0.8,0.5,1));
    legs.append('zeroSR1c-LS');
    nams.append('zeroSR1c_LS');



################################################################################
if run_mfzeroSR1: 
    
    print('');
    print('********************************************************************************');
    print('*** Monotone Fast Zero SR1 Proximal Quasi-Newton Method ***');
    print('***********************************************************');

    options = {
        'init':              x0,
        'stepsize':          1.0/Lip,
        'storeResidual':     True,
        'storeTime':         True
    }
    oracle = {
        'fun_f':    lasso_objectiveSmooth,
        'fun_g':    lasso_objectiveNonSmooth,
        'grad_f':   lasso_gradient,
        'prox_g':   lasso_prox_rank1_c,
        'residual': lasso_residual
    }
    
    output = mfzeroSR1_pg(model, oracle, options, tol, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.1,0.8,0.8,1));
    legs.append('mfzeroSR1');
    nams.append('mfzeroSR1');

################################################################################
if run_tseng_zeroSR1: 
    
    print('');
    print('********************************************************************************');
    print('*** Tseng Fast Zero SR1 Proximal Quasi-Newton Method ***');
    print('********************************************************');

    options = {
        'init':              x0,
        'stepsize':          1.0/Lip,
        'storeResidual':     True,
        'storeTime':         True
    }
    oracle = {
        'fun_f':    lasso_objectiveSmooth,
        'fun_g':    lasso_objectiveNonSmooth,
        'grad_f':   lasso_gradient,
        'prox_g':   lasso_prox_rank1_c,
        'residual': lasso_residual
    }
    
    output = tseng_zeroSR1_pg(model, oracle, options, tol, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.5,0.1,0.8,1));
    legs.append('tsengzeroSR1');
    nams.append('tsengzeroSR1');


################################################################################
if run_SpaRSA: 
    
    print('');
    print('********************************************************************************');
    print('*** Sparse Reconstruction by Separable Approximation ***');
    print('********************************************************');

    options = {
        'init':           x0,
        'backtrackingHistory': 0,
        'storeResidual':  True,
        'storeTime':      True
    }
    oracle = {
        'fun_f':    lasso_objectiveSmooth,
        'fun_g':    lasso_objectiveNonSmooth,
        'grad_f':   lasso_gradient,
        'prox_g':   lasso_prox,
        'residual': lasso_residual
    }
    
    output = spg(model, oracle, options, tol, maxiter, check);
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.3,0.5,1,1));
    legs.append('SPG/SpaRSA');
    nams.append('SPG-SpaRSA');




################################################################################
### evaluation #################################################################
nalgs = len(rs);


#print "Solution:\n", xs[-1];
#print "Group Structure\n", model['B']
print "Sparisty: ", (sum(abs(xs[-1])<1e-8)*100.0)/model['N'];

# print final residual
print('');
for i in range(0,nalgs):
    print 'alg: %s, time: %f, res: %f' % (legs[i], ts[i][-1], rs[i][-1])

if compute_optimal_value == False:
    for i in range(0,nalgs):
        if ts[i].size > 0:
            file = open("GroupLasso_conv_"+nams[i]+"_time.dat", "w");
            for j in range(0,maxiter,1):
                file.write("%f %.12f\n" %(ts[i][j], rs[i][j])); 
            file.close();
        file = open("GroupLasso_conv_"+nams[i]+"_iter.dat", "w");
        for j in range(0,maxiter,1):
            file.write("%d %.12f\n" %(j, rs[i][j])); 
        file.close();


# plotting
fig1 = plt.figure();

for i in range(0,nalgs):
    plt.plot(ts[i][1:-1], rs[i][1:-1], '-', color=cols[i], linewidth=2);
    #plt.plot(rs[i][1:-1], '-', color=cols[i], linewidth=2);

plt.legend(legs);
plt.yscale('log');
#plt.xscale('log');

plt.xlabel('time')
plt.ylabel('residual');
plt.title('GroupLasso')

plt.show();
#plt.draw();











