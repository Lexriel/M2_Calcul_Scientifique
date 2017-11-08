# ifndef _TS_CHINESE_REMAINDER_
# define _TS_CHINESE_REMAINDER_

__device__ void funcPOS(sfixn i, sfixn s, sfixn *primes, sfixn *out);

__global__ void createDL(sfixn *D, sfixn *L, sfixn *primes, sfixn s);

__global__ void copyX(sfixn *X, sfixn *Polynomial_shift, sfixn n, sfixn step);

# endif // _TS_CHINESE_REMAINDER_
