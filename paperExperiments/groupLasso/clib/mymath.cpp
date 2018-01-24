
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "mymath.h"
#include <vector>
#include <algorithm>





class Prox_rk1_generic
{
  public: 
    Prox_rk1_generic(
      double* _x,         // solution of the proximal mapping
      double* _x0,        // proximal center (const)
      double* _d,         // diagonal of the diagonal part of the metric
      double* _u,         // rank1 part of the metrix
      double _sigma,      // sign of the rank1 part of the metric
      int _N)             // dimension of the problem
    {
      x = _x;
      x0 = _x0;
      d = _d;
      u = _u;
      sigma = _sigma;
      N = _N;
      // allocate memory for an auxiliary variable
      x_tilde = new double[N];  
      oneside_shift = 1e10;
    }
    ~Prox_rk1_generic()
    {
      delete[] x_tilde;
    }

    void solve()
    {
      // get breakpoints
      std::vector<double> bpts;
      get_breakpoints(bpts);

      // sort list of breakpoints
      sort( bpts.begin(), bpts.end() );
      bpts.erase( unique( bpts.begin(), bpts.end() ), bpts.end() );
      int nbpts = bpts.size();

      // Now, we search for the interval between two (adjacent) breakpoints 
      // that contains the root of $p(a) := a - <u, x(a) - x0> = 0$, where x(a) 
      // is the prox evaluated at a.
      // The algorithmic strategy is binary search / bisectioning, which can be 
      // done, since p(a) is monotonically increasing.
      int idx_la = 0;             // index of left interval border
      int idx_ra = nbpts-1;       // index of right interval border
      double la, ra;              // left and right interval borders

      if (nbpts == 0)
      {
        //std::cout << "Find root in (-infty,infty)" << std::endl;
        find_root(-oneside_shift,oneside_shift);
        return;
      }

      // check left border
      if (value(bpts[idx_la]) > 0)
      {
        // The zero of p(a) is in (-\infty,bpts(idx_la)].
        //std::cout << "Find root in (-infty," << bpts[idx_la] << ")" << std::endl;
        find_root(bpts[idx_la]-oneside_shift,bpts[idx_la]);
        return;
      }

      // check right border
      if (value(bpts[idx_ra]) < 0)
      {
        // The zero of p(a) is in [bpts(idx_ra),+\infty)
        //std::cout << "Find root in (" << bpts[idx_ra] << ",infty)" << std::endl;
        find_root(bpts[idx_ra],bpts[idx_ra]+oneside_shift);
        return;
      }

      // find interval with zero of p(a)
      int maxiter = (int)(ceil(log(nbpts)/log(2.0))+1);
      int j;
      for (int i=0; i<maxiter; ++i)
      {
        //std::cout << "Find root in (" << bpts[idx_la] << "," << bpts[idx_ra] << ")" << std::endl;
        j = (idx_ra+idx_la+1.0)/2.0;
        
        if (value(bpts[j]) < 0)
        {
          idx_la = j;
        }
        else
        {
          idx_ra = j;
        }

        if (idx_ra - idx_la <= 1)
        {
            break;
        }
      }
      //std::cout << "Find root in (" << bpts[idx_la] << "," << bpts[idx_ra] << ")" << std::endl;
      find_root(bpts[idx_la],bpts[idx_ra]);
      return;

    };

    virtual void find_root(double la, double ra) = 0;
    virtual void prox_diag(double* x_tilde) = 0;
    virtual void get_breakpoints(std::vector<double>& bpts) = 0;

    // computes a - dot(u.T, x(a)-x0)
    double value (double a)
    {
      for (int i=0; i<N; ++i)
      {
        x_tilde[i] = x0[i] - sigma*a*u[i]/d[i];
      }
      prox_diag(x_tilde);

      for (int i=0; i<N; ++i)
        a -= u[i]*(x[i]-x0[i]);
      return a;
    }



  protected:

    // The arguments of the function are stored as member variables to 
    // have a quick and flexible access to all the variables

    double* x;         // solution of the proximal mapping
    double* x0;        // proximal center (const)
    double* x_tilde;   // a shifted version of the proximal center 
    double* d;         // diagonal of the diagonal part of the metric
    double* u;         // rank1 part of the metrix
    double sigma;      // sign of the rank1 part of the metric
    int N;             // dimension of the problem

    // for one-sided intervals, \infty is approximated by this number
    double oneside_shift;


};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////





class Prox_rk1_generic_PLC : public Prox_rk1_generic
{
  public:
    Prox_rk1_generic_PLC(
      double* _x,         // solution of the proximal mapping
      double* _x0,        // proximal center (const)
      double* _d,         // diagonal of the diagonal part of the metric
      double* _u,         // rank1 part of the metrix
      double _sigma,      // sign of the rank1 part of the metric
      int _N)             // dimension of the problem
    : Prox_rk1_generic(_x,_x0,_d,_u,_sigma,_N) 
    {
      oneside_shift = 10.0;
    };
  

  // find root of a linear function in [la, ra] 
  void find_root(double la, double ra)
  {
    double p_la = value(la);
    double p_ra = value(ra);
    double slope = (p_ra - p_la)/(ra - la);
    double a = la - p_la/slope;
      
    // compute solution of proximal mapping
    for (int i=0; i<N; ++i)
    {
      x_tilde[i] = x0[i] - sigma*a*u[i]/d[i];
    }
    prox_diag(x_tilde);

    // sanity check
    double err = value(a); // Warning: This also modifies the output!
    if (fabs(err) > 1e-8)
    {
      std::cout << "WARNING! Rank1 prox could not be solved accurately. Error: "
                << err << std::endl;
    }

  }
  
  virtual void prox_diag(double* x_tilde) = 0;
  virtual void get_breakpoints(std::vector<double>& bpts) = 0;
  
};






////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////






class Prox_rk1_generic_PS : public Prox_rk1_generic
{
  public:
    Prox_rk1_generic_PS(
      double* _x,         // solution of the proximal mapping
      double* _x0,        // proximal center (const)
      double* _d,         // diagonal of the diagonal part of the metric
      double* _u,         // rank1 part of the metrix
      double _sigma,      // sign of the rank1 part of the metric
      int _N)             // dimension of the problem
    : Prox_rk1_generic(_x,_x0,_d,_u,_sigma,_N) 
    {
      use_a_init = false;
    };
  
  
  
  void find_root(double la, double ra)
  {
    
    // initialization
    double a = 0.0;
    if (use_a_init)
    {
      a = a_init;
    }
    a = fmax(la, fmin(ra, a));
    double tau = 1.0;
    double pa, dp_da;
    for (int iter=0; iter<20; ++iter)
    {
      pa = value(a);
      if (fabs(pa) < 1e-8) break;  // breaking condition
      dp_da = derivative(a);
      a = a - tau*pa/dp_da;
      tau = tau*0.95;
    }
    // sanity check
    double err = value(a); // Warning: This also modifies the output!
    if (fabs(err) > 1e-8)
    {
      std::cout << "WARNING! Rank1 prox could not be solved accurately. Error: "
                << err << std::endl;
    }
    if (use_a_init)
    {
      a_init = a;
    }

  }
  virtual double derivative(double a) = 0;
  
  virtual void prox_diag(double* x_tilde) = 0;
  virtual void get_breakpoints(std::vector<double>& bpts) = 0;
  
  // use this to do warm start in find_root
  bool use_a_init;
  double a_init;



};



class Prox_rk1_groupl2l1 : public Prox_rk1_generic_PS
{
  public:
    Prox_rk1_groupl2l1(
      double* _x,         // solution of the proximal mapping
      double* _x0,        // proximal center (const)
      double* _d,         // diagonal of the diagonal part of the metric
      double* _u,         // rank1 part of the metrix
      double _sigma,      // sign of the rank1 part of the metric
      int _N)             // dimension of the problem
    : Prox_rk1_generic_PS(_x,_x0,_d,_u,_sigma,_N) { };


  double derivative(double a)
  {
    int j,k;
    double da = 0.0;
    double da_b;
    double d_b;
    double nrm;
    double nrm_db_inv;
    double dot_xu_b;

    for (k=0; k<lenB-1; ++k)
    {
      d_b = d[B[k]];
      nrm = 0.0;
      dot_xu_b = 0.0;
      for (j=B[k]; j<B[k+1]; ++j)
      {
        nrm += x_tilde[j]*x_tilde[j];
        dot_xu_b += x_tilde[j]*u[j];
      }
      nrm = sqrt(nrm);                  // = |x_b|
      nrm_db_inv = 1.0/(nrm*d_b);       // = 1.0/(d_b*|x_b|)
      dot_xu_b =  dot_xu_b*nrm_db_inv;  // = <x_b/|x_b|,u_b/d_b>

      if (nrm > 1.0/d_b)
      {
        da_b = 0.0;
        for (j=B[k]; j<B[k+1]; ++j)
        {
          // Compute the j-th coordinate of 
          //    (1-1/(d_b*|x_b|))*u_b/d_b
          //    + x_b/|x_b|*<x_b/|x_b|,u_b/d_b>/(d_b*|x_b|)
          // 
          da_b = (1.0-nrm_db_inv)*u[j]/d_b              
               + (x_tilde[j]*dot_xu_b)*nrm_db_inv/nrm;
          
          da += u[j]*da_b;  // = <u_b,da_b> 
        }
      }
    }
    da = 1.0 + sigma*da; // = 1.0 + sigma*<u,da>
  
    return da;
  }

  void prox_diag(double* x_tilde)
  {

    double tmp;
    for (int k=0; k<lenB-1; ++k)
    {
      double nrm = 0.0;
      for (int j=B[k]; j<B[k+1]; ++j)
      {
        tmp = x_tilde[j]*d[j];
        nrm += tmp*tmp;
      }
      if (nrm <= 1.0)
      {
        for (int j=B[k]; j<B[k+1]; ++j)
        {
          x[j] = 0.0;
        }
      }
      else
      {
        nrm = 1.0/sqrt(nrm);
        for (int j=B[k]; j<B[k+1]; ++j)
        {
          x[j] = x_tilde[j] - x_tilde[j]*nrm;
        }
      }
    }

  }


  void get_breakpoints(std::vector<double>& bpts)
  {
    // find breakpoints
    bpts.reserve(2*lenB);
    double AA, BB, CC; 
    double d_b;
    double dis;
    for (int i=0; i+1<lenB; ++i)
    {
      // solve AA*alpha**2 + BB*alpha + CC = 0
      // AA = sum(u_b**2); 
      // BB = -2.0*sigma*d_b*sum(x0_b*u_b);
      // CC = d_b**2*sum(x0_b**2) - 1.0;
      AA = BB = CC = 0.0;
      d_b = d[B[i]];
      for (int j=B[i]; j<B[i+1]; ++j)
      {
        AA += u[j]*u[j];
        BB += x0[j]*u[j];
        CC += x0[j]*x0[j];
      }
      BB = -2.0*sigma*d_b*BB;
      CC = d_b*d_b*CC-1.0;
      dis = BB*BB - 4.0*AA*CC;
      if (dis < 0.0)
        continue;
      dis=sqrt(dis);
      bpts.push_back(0.5*(-BB-dis)/AA);
      bpts.push_back(0.5*(-BB+dis)/AA);
    }
    
    //std::cout << "Breakpoints: [";
    //for (int i=0; i<bpts.size(); ++i)
    //{
    //  std::cout << bpts[i];
    //  if (i+1< bpts.size())
    //    std::cout << ", ";
    //}
    //std::cout << "]" << std::endl;
  }



  // prox specific parameters
  int* B;        
  int lenB;

};





/*
 *  Proximal mapping w.r.t. the diagonal +/- rank1 metric for the function
 *  
 *      g(x) = |x|_B 
 *  
 *  where
 *      
 *      B       [0,K_1,K_2,...,N] is a list of coordinates belonging to the 
 *              same group. It contains len(B)-1 groups. The i-the group 
 *              (i=0,1,...,len(B)-1) contains the indizes {B[i], ..., B[i+1]-1}.
 *      |x|_B   := sum_{i=0}^{len(B)-1} |x_{B[i], ..., B[i+1]-1}|_2
 *      d       WARNING: The implementation requires that the coordinates
 *              of d belonging to the same group are equal!
 *
 *  The solution pf prox_g^D is a piecewise smooth function with breakpoints 
 *  at roots a of |x_b - a*u_b/d_b| - 1/d_b, where the index b refers to the 
 *  block of coordinates b. Roots needs to be found for each of the blocks.
 *  It is solved using 'prox_rk1_generic_PS'.
 * 
 *  x    : double*      result of proximal problem
 *  x0   : double*      proximal center
 *  d    : double*      diagonal part of metric
 *  u    : double*      rank1 part of the metric
 *  N    : int          problem dimension
 *  sigma: double       sign of the rank1 part of the metric
 *  B    : int*         List of coordinate groups
 *  lenB : int          length of list of coordinate groups
 *
 */
void prox_rk1_groupl2l1(double* x, 
                        double* x0, 
                        double* d,
                        double* u,
                        int N,
                        double sigma,
                        int* B,
                        int lenB,
                        double* a_init)
{

  Prox_rk1_groupl2l1 prox(x,x0,d,u,sigma,N);
  prox.B = B;
  prox.lenB = lenB;
  prox.use_a_init = true;
  prox.a_init = *a_init;
  prox.solve();
  *a_init = prox.a_init;

  // return x (changed inside Prox.solve(bpts))
}



