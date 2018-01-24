#ifndef MYMATH_H__
#define MYMATH_H__

extern "C" {

void prox_rk1_groupl2l1(double* x,
                        double* x0, 
                        double* d,
                        double* u,
                        int N,
                        double sigma,
                        int* B,
                        int nbpts,
                        double* a_init);

}

#endif  // MYMATH_H__

