import numpy as np
from numpy.random import rand
from numpy.random import normal as randn
from numpy import abs,sum,max,sign,sqrt,maximum
from numpy import zeros,ones

import matplotlib.pyplot as plt

### use this for writing the c_interface ######################################
from numpy.ctypeslib import ndpointer
import ctypes
import os
libfile = os.path.join(os.path.dirname(__file__),
                       'clib',
                       'mymath.so');
lib = ctypes.cdll.LoadLibrary(libfile);
###############################################################################


### auxilliary functions
def pmax(A,B):
    return np.maximum(A,B);
def pmin(A,B):
    return np.minimum(A,B);



################################################################################
### Diagonally Scaled Proximal Opertators ######################################
################################################################################
#                                                                              #
# These proximal operators are functions prox_{g}^D that assign to a point x0  #
# a solution of the following optimization problem:                            #
#                                                                              #
#       min_x g(x) + 0.5|x-x0|_{D}^2                                           #
#                                                                              #
# where                                                                        #
#                                                                              #
#       |x|_D   := <x,Dx>                                                      #
#       D       := diag(d) is a diagonal matrix, d_i>0                         #
# 	x0	proximal center (point in R^N)                                 #
#       g       proper closed function.                                        #
#                                                                              #
# The signature of any of the following implementations of the proximal        #
# mapping is                                                                   #
#                                                                              #
# 	prox_[abbreviation for g](x0,d,params={}).                             #
#                                                                              #
# params can be used to pass additional arguments in a dictionary.             #
#       (igonore them, if not needed).                                         #
#                                                                              #
#                                                                              #
# If not stated explicitly, the proximal operator can be applied also with     #
# a scalar d.                                                                  #
#                                                                              #
################################################################################

def prox_zero(x0, d, params={}):
    """
    Proximal mapping for the zero function = identity mapping.
    """
    return x0;


def prox_sql2(x0, d, params={}):
    """
    Proximal mapping for the function

        g(x) = 0.5*|x|_2^2

    """
    return x0/(1.0 + 1.0/d);


def prox_l1(x0, d, params={}):
    """ 
    Proximal mapping for the function

        g(x) = |x|_1

    The solution is the soft-shrinkage thresholding.
    """
    return maximum(0.0, abs(x0) - 1.0/d)*sign(x0);

def prox_groupl2l1(x0, d, params={}):
    """ 
    Proximal mapping for the function

        g(x) = |x|_B 
    
    where
        
        B       [0,K_1,K_2,...,N] is a list of coordinates belonging to the 
                same group. It contains len(B)-1 groups. The i-the group 
                (i=0,1,...,len(B)-1) contains the indizes {B[i], ..., B[i+1]-1}.
        |x|_B   := sum_{i=0}^{len(B)-1} |x_{B[i], ..., B[i+1]-1}|_2
        d       WARNING: The implementation requires that the coordinates
                of d belonging to the same group are equal!

    The solution is the group soft-shrinkage thresholding.
    """
    x = x0.copy();
    x_sq = (d*x)**2;
    B = params['B'];
    for k in range(0,len(B)-1):
        dnrm = sqrt(sum( x_sq[B[k]:B[k+1]] ));
        if (dnrm <= 1.0):
            x[B[k]:B[k+1]] = 0.0;
        else:
            x[B[k]:B[k+1]] = x[B[k]:B[k+1]] - x[B[k]:B[k+1]]/dnrm;
    return x;


def prox_l0(x0, d, params={}):
    """ 
    Proximal mapping for the function

        g(x) = |x|_0 = |{x_i != 0}|

    The solution is a hard shrinkage.
    """
    x = x0.copy();
    idx = (x*x)<=2.0/d;
    x[idx] = 0;
    return x;


################################################################################
### Projection Operators #######################################################
################################################################################
#                                                                              #
# Projection opertors are proximal operators for indicator functions of sets.  #
# It computes a closest point from x0 in a set C. In other words, the output   #
# solves the following:                                                        #
#                                                                              #
#       min_x \ind_C(x) + 0.5|x-x0|^2                                          #
#                                                                              #
# where                                                                        #
#                                                                              #
#       \ind_C  is the indicator function of the set C                         #
# 	x0	proximal center (point in R^N)                                 #
#                                                                              #
################################################################################

def proj_simplex(x0, d=1.0, params={}):
    """ 
    Projects the point x0 onto the unit simplex.
    """
    N = x0.size;

    mu = -1e10;
    for i in range(0,N):
        a = 0;
        b = 0;
        for j in range(0,N):
            if (x0[j,0] > mu):
                a = a + x0[j,0];
                b = b + 1;
        mu = (a-1)/b;
   
    x0 = x0 - mu;
    x0[x0<=0] = 0;
    return x0;


def proj_box(x0, d=1.0, params={'a':0.0, 'b':1.0}):
    """ 
    Projects the point x0 onto a box of size [a,b]^N.
    """
    a = params['a'];
    b = params['b'];
    
    return pmax(a, pmin(b, x0));



################################################################################
### Id +/- rank 1 Scaled Proximal Opertators ###################################
################################################################################
#                                                                              #
# These proximal operators are functions prox_{g}^{M} that assign to a point   #
# x0 a solution of the following optimization problem:                         #
#                                                                              #
#       min_x g(x) + 0.5|x-x0|_{M}^2,   M := D + sigma*u*u^t                   #
#                                                                              #
# where                                                                        #
#                                                                              #
#       |x|_D   := <x,Dx>                                                      #
#       D       := diag(d) is a diagonal matrix, d_i>0                         #
#       u       in R^N yield the rank1 perturbation                            #
#       sigma   \in {-1,1} sign of the rank1 perturbation                      #
# 	x0	proximal center (point in R^N)                                 #
#       g       proper closed function.                                        #
#                                                                              #
# The signature of any of the following implementations of the proximal        #
# mapping is                                                                   #
#                                                                              #
# 	prox_rk1_[abbreviation for g](x0,d,u,sigma,params={}).                 #
#                                                                              #
# If not stated explicitly, the proximal operator can be applied also with     #
# a scalar d.                                                                  #
#                                                                              #
################################################################################


def prox_rk1_generic_PLC(x0,d,u,sigma,bpts,prox_fun,params={}):
    """
    Generic proximal mapping for piecewise linear continuous (PLC) solutions
    of the standard proximal mapping. This function solves the following:

        argmin_x g(x) + 0.5|x-x0|_{D + sigma*u*u^t}^2

    where 
        D=diag(d_1, ..., d_n) is a diagonal matrix with positive entries d_i
        u                     is a vector in \R^N, such that D+sigma*u*u^t is
                              positive definite (it will not be checked here!)
        sigma                 \in {+1,-1}
        g(x)                  Generic function; Implementation of 
                                    prox_{g}^D 
                              is required -> prox_fun.
        bpts                  A list of breakpoints
        prox_fun              Standard proximal mapping (w.r.t. diagonal metric)
                              for the function g.
    """
    # list of breakpoints
    bpts = np.unique(bpts);
    nbpts = len(bpts);

    # Now, we search for the interval between two (adjacent) breakpoints where
    # the zero of $p(a) := a - <u, x(a) - x0> = 0$, where x(a) is the prox 
    # evaluated at a.
    # The algorithmic strategy is binary search / bisectioning, which can be 
    # done, since p(a) is monotonically increasing.
    idx_la = 0;             # index of left interval border
    idx_ra = nbpts-1;       # index of right interval border
    p_bpts = zeros(nbpts);  # values of p at the (computed) breakpoints

    # compute final root of p(a) and return x(a)
    def get_x_of_root(la,p_la,ra,p_ra):
        slope = (p_ra - p_la)/(ra - la);
        a = la - p_la/slope;
        x = prox_fun(x0 - sigma*a*u/d, d, params);
        # sanity check
        err = abs(a - np.dot(u.T, x-x0));
        if err > 1e-8:
            print "ATTENTION!!! Rank1-prox could not be solved accurate\
                  enough!";
            print "Distance from zero is: ", err;
        return x;

    # TODO: handle empty list of breakpoints! # never happened so far!

    # check left border
    la = bpts[idx_la];
    x_la = prox_fun(x0 - sigma*la*u/d, d, params);
    p_bpts[idx_la] = la - np.dot(u.T, x_la-x0);
    if p_bpts[idx_la] > 0:
        # The zero of p(a) is in (-\infty,bpts(idx_la)].
        ra = la;
        p_ra = p_bpts[idx_la];
        la = la - 10.0;
        x_la = prox_fun(x0 - sigma*la*u/d, d, params);
        p_la = la - np.dot(u.T, x_la-x0);
        return get_x_of_root(la,p_la,ra,p_ra);

    # check right border
    ra = bpts[idx_ra];
    x_ra = prox_fun(x0 - sigma*ra*u/d, d, params);
    p_bpts[idx_ra] = ra - np.dot(u.T, x_ra-x0);
    if p_bpts[idx_ra] < 0:
        # The zero of p(a) is in [bpts(idx_ra),+\infty).
        la = ra;
        p_la = p_bpts[idx_ra];
        ra = ra + 10.0;
        x_ra = prox_fun(x0 - sigma*ra*u/d, d, params);
        p_ra = ra - np.dot(u.T, x_ra-x0);
        return get_x_of_root(la,p_la,ra,p_ra);

    # find interval with zero of p(a)
    maxiter = int(np.ceil(np.log2(nbpts))+1);
    for i in range(1,maxiter):
        j = int(np.round((idx_ra+idx_la+1)/2));
        
        a_j = bpts[j];
        x_j = prox_fun(x0 - sigma*a_j*u/d, d, params);
        p_bpts[j] = a_j - np.dot(u.T, x_j-x0);

        if p_bpts[j] < 0:
            idx_la = j;
        else:
            idx_ra = j; 
        
        if idx_ra - idx_la == 1:
            break;

    return get_x_of_root(bpts[idx_la],p_bpts[idx_la],bpts[idx_ra],p_bpts[idx_ra]);



def prox_rk1_l1(x0,d,u,sigma,params={}):
    """
    Proximal mapping w.r.t. the diagonal +/- rank1 metric for the function
    
        g(x) = |x|_1.

    The solution pf prox_g^D is a piecewise linear function with breakpoints
    sigma*[(-1-x0*d)/u,(1-x0*d)/u]. It is solved using 'prox_rk1_generic_PLC'.

    """
    bpts = sigma*np.vstack([(-1.0+x0*d)/u,(1.0+x0*d)/u]);
    return prox_rk1_generic_PLC(x0,d,u,sigma,bpts,prox_l1,params);
























def prox_rk1_generic_PS(x0,d,u,sigma,bpts,prox_fun,params={}):
    """
    Generic proximal mapping for piecewise smooth (PS) (once continuously 
    differentiable) solutions of the standard proximal mapping. This function 
    solves the following:

        argmin_x g(x) + 0.5|x-x0|_{D + sigma*u*u^t}^2

    where 
        D=diag(d_1, ..., d_n) is a diagonal matrix with positive entries d_i
        u                     is a vector in \R^N, such that D+sigma*u*u^t is
                              positive definite (it will not be checked here!)
        sigma                 \in {+1,-1}
        g(x)                  Generic function; Implementation of 
                                    prox_{g}^D 
                              is required -> prox_fun.
        bpts                  A list of breakpoints
        prox_fun              Standard proximal mapping (w.r.t. diagonal metric)
                              for the function g.
    """
    # list of breakpoints
    bpts = np.unique(bpts);
    nbpts = len(bpts);

    # Now, we search for the interval between two (adjacent) breakpoints where
    # the zero of $p(a) := a - <u, x(a) - x0> = 0$, where x(a) is the prox 
    # evaluated at a.
    # The algorithmic strategy is binary search / bisectioning, which can be 
    # done, since p(a) is monotonically increasing.
    idx_la = 0;             # index of left interval border
    idx_ra = nbpts-1;       # index of right interval border
    p_bpts = zeros(nbpts);  # values of p at the (computed) breakpoints

    # compute final root of p(a) and return x(a)
    def get_x_of_root(la,ra,params):

        if params.has_key('a_init'):
            a = params['a_init'];
        else:
            a = 0.0;
        # make a feasible
        a = pmin(ra, pmax(la, a));
        da_fun = params['da_fun'];    # function to compute first derivative
        tau = 1.0;
        for iter in range(0,20):
            # compute derivatives
            x_tilde = x0 - sigma*a*u/d;
            x = prox_fun(x_tilde, d, params); 
            p_a = a - np.dot(u.T, x-x0);
            dp_da = da_fun(a,x0,x_tilde,d,sigma,u,params);
            # update step
            a = a - tau*p_a/dp_da;
            tau = tau*0.95;
            # compute objective value 
            x = prox_fun(x0 - sigma*a*u/d, d, params);
            p_a = a - np.dot(u.T, x-x0);
            # check breaking condition
            if ( abs(p_a) < 1e-8 ):
                break;
        params['a_init'] = a;
        # sanity check
        err = abs(a - np.dot(u.T, x-x0));
        if err > 1e-8:
            print "ATTENTION!!! Rank1-prox could not be solved accurate enough!";
            print "Distance from zero is: ", err;
        return x;

    if (nbpts == 0):
        return get_x_of_root(-1e10,1e10,params);

    # check left border
    la = bpts[idx_la];
    x_la = prox_fun(x0 - sigma*la*u/d, d, params);
    p_bpts[idx_la] = la - np.dot(u.T, x_la-x0);
    if p_bpts[idx_la] > 0:
        # The zero of p(a) is in (-\infty,bpts(idx_la)].
        ra = la;
        la = -1e10; # approx. of -\infty
        return get_x_of_root(la,ra,params);

    # check right border
    ra = bpts[idx_ra];
    x_ra = prox_fun(x0 - sigma*ra*u/d, d, params);
    p_bpts[idx_ra] = ra - np.dot(u.T, x_ra-x0);
    if p_bpts[idx_ra] < 0:
        # The zero of p(a) is in [bpts(idx_ra),+\infty).
        la = ra;
        ra = 1e10;
        return get_x_of_root(la,ra,params);

    # find interval with zero of p(a)
    maxiter = int(np.ceil(np.log2(nbpts))+1);
    for i in range(1,maxiter):
        j = int(np.round((idx_ra+idx_la+1)/2));
        
        a_j = bpts[j];
        x_j = prox_fun(x0 - sigma*a_j*u/d, d, params);
        p_bpts[j] = a_j - np.dot(u.T, x_j-x0);

        if p_bpts[j] < 0:
            idx_la = j;
        else:
            idx_ra = j; 
        
        if idx_ra - idx_la == 1:
            break;

    return get_x_of_root(bpts[idx_la],bpts[idx_ra],params);



def prox_rk1_groupl2l1(x0,d,u,sigma,params={}):
    """
    Proximal mapping w.r.t. the diagonal +/- rank1 metric for the function
    
        g(x) = |x|_B 
    
    where
        
        B       [0,K_1,K_2,...,N] is a list of coordinates belonging to the 
                same group. It contains len(B)-1 groups. The i-the group 
                (i=0,1,...,len(B)-1) contains the indizes {B[i], ..., B[i+1]-1}.
        |x|_B   := sum_{i=0}^{len(B)-1} |x_{B[i], ..., B[i+1]-1}|_2
        d       WARNING: The implementation requires that the coordinates
                of d belonging to the same group are equal!

    The solution pf prox_g^D is a piecewise smooth function with breakpoints 
    at roots a of |x_b - a*u_b/d_b| - 1/d_b, where the index b refers to the 
    block of coordinates b. Roots needs to be found for each of the blocks.
    It is solved using 'prox_rk1_generic_PS'.

    """
    B = params['B'];

    L = len(B)-1;
    bpts = [];
    for i in range(0,L):
        u_b = u[B[i]:B[i+1]];
        x0_b = x0[B[i]:B[i+1]];
        d_b = d[B[i]];
        # solve AA*alpha**2 + BB*alpha + CC = 0
        AA = sum(u_b**2); 
        BB = -2.0*sigma*d_b*sum(x0_b*u_b);
        CC = d_b**2*sum(x0_b**2) - 1.0;
        disc = BB**2 - 4.0*AA*CC;
        if (disc < 0):
            continue;
        disc = sqrt(disc);
        ap = 0.5*(-BB+disc)/AA;
        am = 0.5*(-BB-disc)/AA;
        bpts.append(np.asscalar(am));
        bpts.append(np.asscalar(ap));
    
    # find root of l(a)
    def da_fun(a,x0,x_tilde,d,sigma,u,params):
        B = params['B'];
        da = 0.0;
        for i in range(0,len(B)-1):
            u_b = u[B[i]:B[i+1]];
            d_b = d[B[i]];        # constant within the block
            x_b = x_tilde[B[i]:B[i+1]];
            nrm_b = sqrt(sum(x_b**2));
            # check if condition is active
            if (nrm_b > 1.0/d_b):
                # compute derivative for this block
                x_b_nrm = x_b/nrm_b;
                da_b = u_b/d_b;
                # derivative of prox
                da_b = (1.0 - 1.0/(d_b*nrm_b))*da_b \
                       + (x_b_nrm*(x_b_nrm.T.dot(da_b)))/d_b/nrm_b;
                # add active blocks
                da = da + np.dot(u_b.T, da_b);
        # derivative of l(x_tilde)
        da = 1.0 + sigma*da;

        return da;

    params['da_fun'] = da_fun;

    return prox_rk1_generic_PS(x0,d,u,sigma,bpts,prox_groupl2l1,params);



# register c function in python
c_prox_rk1_groupl2l1 = lib.prox_rk1_groupl2l1;
c_prox_rk1_groupl2l1.restype = None;
c_prox_rk1_groupl2l1.argtypes = [ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                 ctypes.c_int,
                                 ctypes.c_double,
                                 ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),
                                 ctypes.c_int,
                                 ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')];

def cprox_rk1_groupl2l1(x0,d,u,sigma,params={'prox_a_init':0.0}):
    """
    Proximal mapping w.r.t. the diagonal +/- rank1 metric for the function
    
        g(x) = |x|_B 
    
    where
        
        B       [0,K_1,K_2,...,N] is a list of coordinates belonging to the 
                same group. It contains len(B)-1 groups. The i-the group 
                (i=0,1,...,len(B)-1) contains the indizes {B[i], ..., B[i+1]-1}.
        |x|_B   := sum_{i=0}^{len(B)-1} |x_{B[i], ..., B[i+1]-1}|_2
        d       WARNING: The implementation requires that the coordinates
                of d belonging to the same group are equal!

    The solution pf prox_g^D is a piecewise smooth function with breakpoints 
    at roots a of |x_b - a*u_b/d_b| - 1/d_b, where the index b refers to the 
    block of coordinates b. Roots needs to be found for each of the blocks.
    It is solved using 'prox_rk1_generic_PS'.

    """
    B = params['B'];
    a = params['prox_a_init'];

    x = x0.copy();
    c_prox_rk1_groupl2l1(x, x0, d, u, len(x0), sigma,\
                         np.asarray(B).astype(np.int32), len(B), a);

    params['prox_a_init'] = a;
    return x;


