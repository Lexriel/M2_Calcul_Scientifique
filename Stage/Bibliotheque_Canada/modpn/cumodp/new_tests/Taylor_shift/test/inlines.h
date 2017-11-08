#ifndef _INLINES_H_
#define _INLINES_H_

#include <cstdlib>
#include <cassert>
#include <ctype.h>


////////////////////////////////////////////////////////////////////////////////
// modular arithmetic over a finite field
////////////////////////////////////////////////////////////////////////////////
__device__ __inline__ 
sfixn add_mod(sfixn a, sfixn b, sfixn p) {
    sfixn r = a + b;
    r -= p;
    r += (r >> BASE_1) & p;
    return r;
}


// ninv is precomputed
/*__host__ __inline__ 
sfixn mul_mod(sfixn a, sfixn b, sfixn n, double ninv) {

#if DEBUG > 0
    double ninv2 = 1 / (double)n;
#else
    double ninv2 = ninv;
#endif

    sfixn q  = (sfixn) ((((double) a) * ((double) b)) * ninv2);
    sfixn res = a * b - q * n;
    res += (res >> BASE_1) & n;
    res -= n;
    res += (res >> BASE_1) & n;
    return res;
    }*/

__device__ int __double2loint(double a)
{
  volatile union {
    double     d;
    signed int i[2];
  } cvt;

 cvt.d = a;

 return cvt.i[0];
}

__device__ __inline__ sfixn mul_mod(sfixn a, sfixn b, sfixn n, double ninv) 
{
  sfixn  hi = __umulhi(a*2, b*2);
  double rf = (double) (hi) * ninv + (double) (3<<51);
  sfixn r = a * b - __double2loint(rf) * n;
  return (r<0 ? r+n : r);
}

////////////////////////////////////////////////////////////////////////////////
__device__  __inline__ 
void egcd(sfixn x, sfixn y, sfixn *ao, sfixn *bo, sfixn *vo) {
    sfixn t, A, B, C, D, u, v, q;

    u = y; v = x;
    A = 1; B = 0;
    C = 0; D = 1;

    do {
        q = u / v;
        t = u;
        u = v;
        v = t - q * v;
        t = A;
        A = B;
        B = t - q * B;
        t = C;
        C = D;
        D = t - q * D;
    } while (v != 0);

    *ao = A;
    *bo = C;
    *vo = u;
}

////////////////////////////////////////////////////////////////////////////////
__device__  __inline__ 
sfixn inv_mod(sfixn n, sfixn p) {
    sfixn a, b, v;
    egcd(n, p, &a, &b, &v);
    if (b < 0) b += p;
    return b % p;
}



__device__  __inline__ 
sfixn quo_mod(sfixn a, sfixn b, sfixn n, double ninv) {
    return mul_mod(a, inv_mod(b, n), n, ninv);
}

# endif // __INLINES_H_
