#ifndef _INLINES_H_
#define _INLINES_H_

#include "defines.h"
#include <cstdlib>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// modular arithmetic over a finite field
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ 
sfixn add_mod(sfixn a, sfixn b, sfixn p) {
    sfixn r = a + b;
    r -= p;
    r += (r >> BASE_1) & p;
    return r;
}

////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ 
sfixn sub_mod(sfixn a, sfixn b, sfixn p) {
    sfixn r = a - b;
    r += (r >> BASE_1) & p;
    return r;
}

////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ 
sfixn neg_mod(sfixn a, sfixn p) {
    sfixn r = - a;
    r += (r >> BASE_1) & p;
    return r;
}

// ninv is not precomputed
__device__ __host__ __inline__ 
sfixn mul_mod(sfixn a, sfixn b, sfixn n) {
    double ninv = 1 / (double)n;
    sfixn q  = (sfixn) ((((double) a) * ((double) b)) * ninv);
    sfixn res = a * b - q * n;
    res += (res >> BASE_1) & n;
    res -= n;
    res += (res >> BASE_1) & n;
    return res;
}

// ninv is precomputed
__device__ __host__ __inline__ 
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
}

////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ 
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
__device__ __host__ __inline__ 
sfixn inv_mod(sfixn n, sfixn p) {
    sfixn a, b, v;
    egcd(n, p, &a, &b, &v);
    if (b < 0) b += p;
    return b % p;
}

////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ 
sfixn quo_mod(sfixn a, sfixn b, sfixn n) {
    return mul_mod(a, inv_mod(b, n), n);
}

__device__ __host__ __inline__ 
sfixn quo_mod(sfixn a, sfixn b, sfixn n, double ninv) {
    return mul_mod(a, inv_mod(b, n), n, ninv);
}

////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ 
sfixn pow_mod(sfixn a, sfixn ee, sfixn n) {
    sfixn x, y;
    usfixn e;

    if (ee == 0) return 1;
    if (ee == 1) return a;
    if (ee < 0) e = - ((usfixn) ee); else e = ee;

    x = 1;
    y = a;
    while (e) {
        if (e & 1) x = mul_mod(x, y, n);
        y = mul_mod(y, y, n);
        e = e >> 1;
    }

    if (ee < 0) x = inv_mod(x, n);
    return x;
}

__device__ __host__ __inline__ 
sfixn pow_mod(sfixn a, sfixn ee, sfixn n, double ninv) {
    sfixn x, y;
    usfixn e;

    if (ee == 0) return 1;
    if (ee == 1) return a;
    if (ee < 0) e = - ((usfixn) ee); else e = ee;

    x = 1;
    y = a;
    while (e) {
        if (e & 1) x = mul_mod(x, y, n, ninv);
        y = mul_mod(y, y, n, ninv);
        e = e >> 1;
    }

    if (ee < 0) x = inv_mod(x, n);
    return x;
}

/**
 * Compute a^(2^e) mod n.
 *
 * @a, integer 
 * @e, nonnegative unsigned integer
 * @n, modulus
 * @ninv, inverse of n, ninv = 1 / n; 
 *
 */
__device__ __host__ __inline__
sfixn pow_mod_pow2(sfixn a, usfixn e, sfixn n) {
    if (e == 0) return a;
    double ninv = 1/(double)n;

    for (; e > 0; --e) {
        a = mul_mod(a, a, n, ninv);
    }
    return a;
}

__device__ __host__ __inline__
sfixn pow_mod_pow2(sfixn a, usfixn e, sfixn n, double ninv) {
    if (e == 0) return a;
    for (; e > 0; --e) {
        a = mul_mod(a, a, n, ninv);
    }
    return a;
}

/**
 *  Compute a % 2^e, e >= 0
 */
__device__ __host__ __inline__
sfixn rem2e(sfixn a, sfixn e) { return a & ((sfixn(1) << e) - 1); }

/**
 * Compute a / 2^e, e >= 0
 */
__device__ __host__ __inline__
sfixn quo2e(sfixn a, sfixn e) { return a >> e; }


////////////////////////////////////////////////////////////////////////////////
/* random number generator for a vector */
__device__ __host__ __inline__
unsigned int my_rand_vec(unsigned int *SEED, unsigned int tid) {
    #define MULTIPLIER ((unsigned int) 1664525)
    #define OFFSET  ((unsigned int) 1013904223)
    unsigned int sNew = SEED[tid] * MULTIPLIER + OFFSET;
    SEED[tid] = sNew;
    return sNew;
    #undef MULTIPLIER
    #undef OFFSET
}

////////////////////////////////////////////////////////////////////////////////
/* random number generator */
__device__ __host__ __inline__ 
unsigned int my_rand(unsigned int *s) {
    #define MULTIPLIER ((unsigned int) 1664525)
    #define OFFSET  ((unsigned int) 1013904223)
    unsigned int snew = (*s) * MULTIPLIER + OFFSET;
    *s = snew;
    return snew;
    #undef MULTIPLIER
    #undef OFFSET
}

////////////////////////////////////////////////////////////////////////////////
/* return the index of current thread in the block */
__device__ unsigned int tIdxBlock () {
    unsigned int bx = blockDim.x;
    unsigned int by = blockDim.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z;
    return bx*by*tz + bx*ty + tx;
}

////////////////////////////////////////////////////////////////////////////////
/* return the index of current block in the grid */
__device__ unsigned int bIdxGrid() {
    unsigned int gx = gridDim.x;
    unsigned int gy = gridDim.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bz = blockIdx.z;
    return  gx*gy*bz + gx*by + bx;
}

////////////////////////////////////////////////////////////////////////////////
/* return the index of current thread in the grid */
__device__ unsigned int tIdxGrid () {
    unsigned int numOfThreadInBlock = blockDim.x * blockDim.y * blockDim.z;
    return bIdxGrid() * numOfThreadInBlock + tIdxBlock();
}

////////////////////////////////////////////////////////////////////////////////
// @x, the integer to be inversed
// @n, the number of bits to use
// Return value: rev(x) as a n bit integer.
// Only work for 32-bit integers.
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ __inline__ uint32 bit_reverse(uint32 x, int32 n) {
    uint32 y = 0x55555555;
    x = (((x >> 1) & y) | ((x & y) << 1));
    y = 0x33333333;
    x = (((x >> 2) & y) | ((x & y) << 2));
    y = 0x0f0f0f0f;
    x = (((x >> 4) & y) | ((x & y) << 4));
    y = 0x00ff00ff;
    x = (((x >> 8) & y) | ((x & y) << 8));
    y = ((x >> 16) | (x << 16));
    return y >> (32 - n);
}

/**
 * Returns the floor form of binary logarithm for an unsigned integer.
 * -1 is returned if n is 0.
 */
__host__ __device__ __inline__ int32 floor_log2(usfixn n) {
    int32 pos = 0, s;

    for(s = (BASE>>1); s>0; s >>= 1) {
        if ( n >= (usfixn)1<<s ) { n >>= s; pos += s; }
    }
    return (n==0) ? (-1) : pos;
}

//////////////////////////////////////////////////
//  ceil(log2(x)) = floor(log(x-1)) + 1, for x>1.
//////////////////////////////////////////////////  
/**
 * Returns the ceiling form of binary logarithm for an unsigned integer.
 * -1 is returned if n is 0.
 */
__host__ __device__ __inline__ int32 ceiling_log2(usfixn n) {
    return floor_log2(n - 1) + 1;
}


// f(i) = i if i >=0
// f(i) = 0 if i < 0;

__host__ __device__ __inline__ sfixn cut_negative(sfixn i) {
    return  i - (i & (i >> BASE_1));
}

/**
 * Given positive integer n and B, find positive integer x, y such that 
 * (1)  1 <= x < B
 * (2)  1 <= y < B
 * (3)  x * y >= n
 *
 * The assumption is 0 < n < B^2. Returns false if no x and y are found.
 * For example, n = 53, B = 32 produce x = 27, y = 2.
 */   
__host__ __device__ __inline__ 
bool approx_factor(sfixn n, sfixn *x, sfixn *y, sfixn B) {
    int count = 10;
    *x = n, *y = 1;
    while (*x >= B || *y >= B) {
        if (n % 2 == 0) {
            *y <<= 1, n >>= 1, *x = n;
            --count;
        } else {
            ++n;
        }
        if (count <= 0) return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Fourier prime structure related
///////////////////////////////////////////////////////////////////////////////

/**
 * @p, a machine prime number 
 * 
 * Rreturns the Fourier degree e of p, that is, 
 * write p = 2^e * q + 1 for some odd q.  
 **/
__host__ __device__ __inline__ sfixn fourier_degree(sfixn p) {
    sfixn e = 1;
    --p;
    while (!(p % ((sfixn)1 << e))) ++e;
    return e - 1;
}

/**
 * @n: a positive machine integer
 * Return true if n is prime, otherwise return false.  
 * This is a deterministic algorithm by Miller & Rabin.
 *
 * Assume that n is an integer of at most 49 bits. 
 **/
static inline bool is_prime(sfixn n) {
    if (n < 2 || n % 2 == 0) return false;
    switch (n) {
        case (2)  : return true;
        case (3)  : return true;
        case (5)  : return true;
        case (7)  : return true;
        case (11) : return true;
        case (13) : return true;
        case (17) : return true;
        default :  break;
    }

    // now n is an odd positive integer
    sfixn s = fourier_degree(n);
    sfixn d = (n - 1) >> s;
    const sfixn A[7] = {2, 3, 5, 7, 11, 13, 17};
    for (int i = 0; i < 7; i++) {
        if (pow_mod(A[i], d, n) != 1) {
            bool flag = true;
            for (sfixn r = 0; r < s; ++r) {
                if (pow_mod(A[i], ((sfixn)1 << r ) * d, n) == n - 1) {
                    flag = false; break;    
                }
            }
            if (flag) return false;
        }
    }
    return true;
}

/**
 * @p:  machine prime p = q * 2^E + 1 with q being odd.
 * @n:  n = 2^e and e <= E  
 * @e:  e = log[2](n)
 * 
 * Returns a primtive n-th root of unity
 **/
static inline sfixn primitive_root(sfixn e, sfixn p) {
    // Let E = fourierDegree(p), and write p = q * 2^E + 1 for
    // some odd q. Then k = q * 2^(E - e) and n2 = 2^(e-1).
    //
    // Given a random number 1 < r < p, by FLT, 
    //       r^(p - 1) = 1 mod p.
    // Hence, r^(q * 2^E) = 1 mod p. Let w = r^(q * 2^(E - e)) mod p.
    // Then w^(2^e) = w^(q * 2^E) = 1 mod p. 
    // If w^(2^(e-1)) != 1 then w is a 2^e-th primitive root of unity. 
    // Half of the random numbers are good.
    
    if (DEBUG) assert(e <= fourier_degree(p));

    sfixn root = 0;
    sfixn k = (p - 1) >> e;
    sfixn n2 = (sfixn)1 << (e - 1);
    // Remove randomness
    srand(1);
    while (p > 2) {
        root = 1;
        while (root < 2) root = rand() % p;
        root = pow_mod(root, k, p);
        if (pow_mod(root, n2, p) != 1) { break; }
    }
    return root;	
}

/**
 * init_fourier_prime:
 * @fpp : (output) pointer to prime number structure
 * @p : prime number.
 * 
 **/
static inline void init_fourier_prime(fprime_t *fpp, sfixn p) {
    fpp->p = p;
    sfixn p1 = p - 1;
    sfixn npow = 1;
    while (!(p1 % ((sfixn)1 << npow))) { ++npow; }
    --npow;
    fpp->npow = npow;
    fpp->c = (p1 >> npow);
    fpp->rpow = ceiling_log2(p);
    fpp->r = ((sfixn)1 << fpp->rpow) % p;
    fpp->rnpow = fpp->rpow - fpp->npow; 
}

////////////////////////////////////////////////////////////////////////////////
/**
 * Gentleman-Sande FFT : natural input, revsersed output
 *
 * n  : fft size, power of 2
 * A  : input array
 * p  : prime number
 * wn : n-th primitive root of unit
 *
 * 18 registers will be used by this kernel.
 * 
 * TESTED
 *
 */
__device__ __host__ __inline__
void gsnr_fft_ip(sfixn n, sfixn *A, sfixn p, sfixn wn) {
    sfixn NumOfProblems = 1;
    sfixn ProblemSize = n;
    sfixn wk = wn;
    while (ProblemSize > 1) {
        sfixn halfSize = (ProblemSize >> 1);
        for (sfixn k = 0; k < NumOfProblems; ++k) {
            sfixn JFirst = k * ProblemSize;
            sfixn JLast = JFirst + halfSize - 1;
            sfixn w = 1;
            for (sfixn j = JFirst; j <= JLast; ++j) {
                sfixn Temp = A[j];
                A[j] = add_mod(Temp, A[j + halfSize], p);
                A[j + halfSize] = mul_mod(w, sub_mod(Temp, A[j + halfSize], p), p);  
                w = mul_mod(w, wk, p);
            }
        }
        NumOfProblems <<= 1;
        ProblemSize = halfSize; 
        wk = mul_mod(wk, wk, p);
    }
}

/*
 * A should be in the shared memory. Transform A into natural order,
 * taking the bit-reverse ordering bitReverse takes only 32-bit integers!
 *
 * */
__device__ __host__ __inline__
void gsnn_fft_ip(sfixn n, sfixn e, sfixn *A, sfixn p, sfixn wn)
{
    gsnr_fft_ip(n, A, p, wn);
    sfixn v, target;
    for (sfixn i = 0; i < n; ++i) {
        target = bit_reverse(i, e);
        if (target > i) { v = A[target]; A[target] = A[i]; A[i] = v; }
    }
}

#endif // END_OF_FILE
