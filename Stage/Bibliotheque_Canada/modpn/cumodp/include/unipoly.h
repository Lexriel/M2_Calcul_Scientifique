#ifndef _UNI_POLY_
#define _UNI_POLY_

#include "defines.h"
#include "inlines.h"

////////////////////////////////////////////////////////////////////////////////
// Unvariate polynomials are encoded as a plain vector of coefficient in 
// increasing order. We assume that the coefficients are all
// from 0 to p - 1, if the input is over a finite field Z/pZ.  
//
// For example, f(x) = 1 + 2x + 3x^2 + 4x^3 is encoded as <1, 2, 3, 4>.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * is_zero_poly_uni :
 * @d   : the degree 
 * @vec : the coefficient vector 
 * 
 * If vec is a zero vector returns true, otherwise returns false. 
 **/
__device__ __host__ __inline__
bool is_zero_poly_uni(sfixn d, const sfixn *vec) {
    for (sfixn i = d; i >= 0; --i) if (vec[i]) return false;
    return true;
}

/**
 * copy_poly_uni :
 * @d : degree of A
 * @A : (source) coefficient vector of A
 * @B : (couput) coefficient vector of B
 *
 * Assumption : A and B has the same degree d.
 **/
__device__ __host__ __inline__
void copy_poly_uni(sfixn d, sfixn *B, const sfixn *A) {
    while (d >= 0) { B[d] = A[d]; --d; }
}

/**
 * negate_poly_uni_ip :
 * @d : the degree 
 * @A : the coefficient vector 
 * @p : prime number
 * return : void
 *
 * Negate each coefficient of a univariate polynomial of degree d.
 *
 **/
__device__ __host__ __inline__
void negate_poly_uni_ip(sfixn d, sfixn *A, sfixn p) {
    sfixn c;
    for (sfixn i = 0; i <= d; ++i) {
        c = A[i];
        if (c) { A[i] = p - c; }
    }
}

__device__ __host__ __inline__
void negate_poly_uni(sfixn d, sfixn *desV, const sfixn *srcV, sfixn p) {
  for (sfixn i = 0; i <= d; ++i) {
      desV[i] = (srcV[i]) ? (p - srcV[i]) : 0;
  }
}

/**
 * @d : the original degree.
 * @A : the coefficient vector.
 *
 * Return value: the true degree. 
 * 
 * This function is the same as the shrinkDegUni in modpn. 
 * It returns 0 whenever A is a constant including 0.
 **/
__device__ __host__ __inline__
sfixn shrink_deg_uni(sfixn d, const sfixn *A) {
    while ( (d > 0) && (! A[d]) ) --d;
    return d;
}

/**
 * @c : scalar 
 * @d : degree of the univeriate polynomial f
 * @A : coefficient vector of the univeriate polynomial f
 * @p : prime number
 *
 * Return value: c * f mod p.
 **/
__device__ __host__ __inline__
void scalar_mul_poly_uni_ip(sfixn c, sfixn d, sfixn *A, sfixn p) {
    sfixn a;
    double pinv =  1 / (double)p;
    for (sfixn i = 0; i <= d; ++i) {
        a = A[i];
        if (a) { A[i] = mul_mod(a, c, p, pinv); }
    }
}

__device__ __host__ __inline__
void scalar_mul_poly_uni_ip(sfixn c, sfixn d, sfixn *A, sfixn p, double pinv)
{
    sfixn a;
    for (sfixn i = 0; i <= d; ++i) {
        a = A[i];
        if (a) { A[i] = mul_mod(a, c, p, pinv); }
    }
}

/**
 * scalar_mul_poly_uni:
 * @c : scalar 
 * @d : degree of the univeriate polynomial A
 * @A : coefficient vector of the univeriate polynomial A
 * @B : coefficient vector of the univeriate polynomial B
 * @pPtr : prime number structure.
 *
 * The size of B is at least d;
 *
 * Return value: B = c * A mod p.
 **/
__device__ __host__ __inline__
void scalar_mul_poly_uni(sfixn c, sfixn d, sfixn *B, const sfixn *A, sfixn p) 
{
    double pinv =  1 / (double)p;
    sfixn a;
    for (sfixn i = 0; i <= d; ++i) {
        a = A[i];
        if (a) B[i] = mul_mod(A[i], c, p, pinv);
    }
}

__device__ __host__ __inline__
void scalar_mul_poly_uni(sfixn c, sfixn d, sfixn *B, const sfixn *A, 
    sfixn p, double pinv) 
{
    sfixn a;
    for (sfixn i = 0; i <= d; ++i) {
        a = A[i];
        if (a) B[i] = mul_mod(A[i], c, p, pinv);
    }
}

/**
 * rem_poly_uni_ip:
 *
 * Computes the remainder of A by B.
 *
 * @dgA  : degree of A
 * @A    : coefficient vector of A
 * @dgB  : degree of B
 * @B    : coefficient vector of B
 * @p    : prime
 *
 * Uses classical division and works for all Fourier Primes.
 * Degree A and B are true degrees.
 * Return value: void, A = rem(A, B).
 **/
__device__ __host__ __inline__
void rem_poly_uni_ip(sfixn dgA, sfixn *A, sfixn dgB, const sfixn *B, sfixn p)
{
    if (dgA < dgB) return;
    sfixn u, i, j, t, q;
    double pinv = 1 / (double)p;
    u = inv_mod(B[dgB], p);
    for (i = dgA - dgB; i >= 0; --i) {
        t = A[dgB + i];
        if (t != 0) {
            q = mul_mod(t, u, p, pinv);
            for (j = 0; j <= dgB; ++j) 
                A[i + j] = sub_mod(A[i + j], mul_mod(B[j], q, p, pinv), p);
        }
    }
}

/**
 * Computes the pseudo-remainder of A by B.
 *
 * @dgA : degree of A
 * @A   : coefficient vector of A
 * @dgB : degree of B
 * @B   : coefficient vector of B
 * @p   : fourier prime
 *
 * Based on the formula :
 *
 *    prem(A, B) = rem(lc(B)^delta * A, B)
 *
 * where delta = 1 + degree(A) - degree(B). 
 *
 * Return value: void, A = prem(A, B).
 **/
__device__ __host__ __inline__
void prem_poly_uni_ip(sfixn dgA, sfixn *A, sfixn dgB, const sfixn *B, sfixn p)
{
    if (dgA < dgB) return;
    sfixn u = pow_mod(B[dgB], 1 + dgA - dgB, p);
    scalar_mul_poly_uni_ip(u, dgA, A, p);
    rem_poly_uni_ip(dgA, A, dgB, B, p);
}

/**
 * Compute the pseudo remainder of two univariate polynomials.
 * 
 * @d : degree of A
 * @A : coefficient vector of a polynomial 
 * @e : degree of B
 * @B : coefficient vector of a polynomial 
 * @T : result polynomial of degree <= e - 1
 * @p : Fourier prime
 * @W : workspace of size at least d + 1 
 *
 * Notes : prem(A, -B) = (-1)^delta * prem(A, B);
 * Assumptions : d >= e > 0
 * return T = prem(A, -B) 
 **/
__device__ __host__ __inline__
void negate_prem_poly_uni(sfixn d, const sfixn *A, sfixn e,
    const sfixn *B, sfixn *T, sfixn p, sfixn *W)
{
    sfixn delta = 1 + d - e;
    // poly A --> W
    copy_poly_uni(d, W, A);
    // prem(W, B) --> W
    prem_poly_uni_ip(d, W, e, B, p); 
    // if delta is odd, then prem(A, B) = - prem(A, -B)
    // if delta is even, then prem(A, B) = prem(A, -B) 
    if (delta & ((sfixn)1)) negate_poly_uni_ip(e - 1, W, p);
    // result W --> T
    copy_poly_uni(e - 1, T, W); 
}

/**
 * Compute the pseudo remainder of two univariate polynomials.
 * 
 * @d : degree of A
 * @A : coefficient vector of a polynomial 
 * @e : degree of B
 * @B : coefficient vector of a polynomial 
 * @T : workspace and result of size d + 1 
 * @p : Fourier prime
 *
 * Notes : prem(A, -B) = (-1)^delta * prem(A, B);
 * Assumptions : d >= e > 0
 * return void; T = prem(A, -B) 
 *
 * !!!! although degree of T should be at most e - 1, we require that
 * the size of T is at least d + 1 since the pseudo remainder is 
 * computed in place.
 *
 **/
__device__ __host__ __inline__
void negate_prem_poly_uni_ip(sfixn d, const sfixn *A, 
    sfixn e, const sfixn *B, sfixn *T, sfixn p)
{
    sfixn delta = 1 + d - e;
    // poly A --> T
    copy_poly_uni(d, T, A);
    // prem(T, B) --> T
    prem_poly_uni_ip(d, T, e, B, p); 
    // if delta is odd, then prem(A, B) = - prem(A, -B)
    // if delta is even, then prem(A, B) = prem(A, -B) 
    if (delta & ((sfixn)1)) negate_poly_uni_ip(e - 1, T, p);
}

/**
 * Compute the division of A by B, returning quotient Q and remainder R
 *
 * @degQ : degree of the quotient
 * @QPtr : (output) Coefficient vector of the quotient
 * @degA : degree of A
 * @APtr : (overwritten) Coefficient vector of A (input) by R (output)
 * @degB : degree of B
 * @BPtr : Coefficient vector of B
 * @p    : prime 
 *
 * The coefficient vector of A is overwritten by that of the remainder R. 
 * The first degB + 1 slots of A will be the coefficient vector of R. 
 * The last degA - degB + 1 slots should be zero. 
 * The relation is A = Q * B + R.
 *
 **/
__device__ __host__ __inline__
void div_poly_uni_ip(sfixn degQ, sfixn *QPtr, sfixn degA, sfixn *APtr, 
    sfixn degB, const sfixn *BPtr, sfixn p)
{
    sfixn i, j, u, t, q;
    if (degA < degB) { for (j = 0; j <= degQ; ++j) QPtr[j] = 0; return; }
    
    // the true degree of B should not be degB.
    double pinv = 1/(double)p;
    u = inv_mod(BPtr[degB], p);
    for (i = degA - degB; i >= 0; --i) {
        t = APtr[degB + i];
        if (t != 0) {
            QPtr[i] = q = mul_mod(t, u, p, pinv);
            for (j = 0; j <= degB; ++j) 
                APtr[i + j] = sub_mod(APtr[i + j], 
                    mul_mod(BPtr[j], q, p, pinv), p);
        } else {
            QPtr[i] = 0;
        }
    }
}

#endif
