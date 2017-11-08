#ifndef _FFT_AUX_
#define _FFT_AUX_

#include "defines.h"

/**
 * Compute 1, w, w^2, ... w^{n-1}, n = 2^e, for DEVICE ARRAY W
 * 
 * @W, device array of length n = 2^e
 * @w, integer in the finite field Zp
 * @p, prime number
 */
void get_powers_binary (sfixn e, sfixn *W, sfixn w, sfixn p);

/**
 * Fill the primitive roots of unity
 *
 * Example. Let m = 2, es[0] = 3, es[1] = 4, s be a 8-th primitive 
 * root of unity and t be a 16-th primitive root of unity. 
 * Then the output W consists of two segments
 *
 * [1, s, s^2, s^3, 1, t, t^2, t^3, t^4, t^5, t^6, t^7]
 */
void get_powers_of_roots(int m, sfixn *W, const sfixn *es, const sfixn p);

/**
 * Inverse  
 *
 * [1, s, s^2, s^3, 1, t, t^2, t^3, t^4, t^5, t^6, t^7]
 *
 * to
 *
 * [1, s^{-1}, s^{-2}, s^{-3}, 
 *  1, t^{-1}, t^{-2}, t^{-3}, t^{-4}, t^{-5}, t^{-6}, t^{-7}]
 * 
 * This can be used in the interpolation stage
 */
void inv_powers_of_roots(int m, sfixn *W, const sfixn *es, const sfixn p);

/**
 * Compute Y[i] = Y[i] * X[i] mod p
 *
 * @n    : the length of the vector, n = 2^k 
 * @Y    : device vector of length n
 * @X    : device vector of length n
 * @p    : prime number
 * @pinv : inverse of p
 */
void pointwise_mul_dev(sfixn n, sfixn k, sfixn *Y, const sfixn *X, sfixn p); 

/**
 * Compute X[i] = X[i] * a mod p
 *
 * @n = 2^k 
 */
void scale_vector_dev(sfixn a, sfixn n, sfixn *X, sfixn p);

/**
 * @X, input and output coefficient vector 
 * @m, power of 2
 * @n, m >= n
 *
 * Example, let n = 7, m = 16, given
 *
 * X = [0, 1, 2, 3, 4, 5, 6, *, *, *, *, *, *, *, *, *],  
 *
 * Only reset the last m - n = 9 elements.
 */ 
void expand_to_fft_dev(sfixn m, sfixn n, sfixn *X);

/* reset a device vector of length n */
void reset_vector_dev(sfixn n, sfixn *X);

/** 
 * Expand to do a list of 1d fft
 *
 * Example, let 
 *
 * F = (1 + 2*x + 3*x^2) + (4 + 5*x + 6*x^2)*y + (7 + 8*x + 9*x^2)*y^2
 *
 * Expand F into
 *
 * F = (1 + 2*x + 3*x^2 + 0*x^3) + 
 *     (4 + 5*x + 6*x^2 + 0*x^3)*y + 
 *     (7 + 8*x + 9*x^2 + 0*x^3)*y^2
 *
 * That is, 
 *
 * {1, 2, 3, 4, 5, 6, 7, 8, 9} ==> {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0}
 *
 * */
void expand_to_list_fft_dev(sfixn q, sfixn m, sfixn k, sfixn *dout, 
                            sfixn n, const sfixn *din);

/**
 * @coeffs1: device array of size nx1 * ny1
 * @coeffs2: device array of size nx2 * ny2
 * @nx2, nx2 = 2^ex2
 * @ny2, ny2 = 2^ey2
 *
 * Given a (nx1, ny1)-rdr bivariate polynomial F, expand it coefficient vector
 * into (nx2, ny2)-rdr such that nx2 >= nx1 and ny2 >= ny1.
 */
void expand_to_fft2_dev(sfixn ex2, sfixn ey2, sfixn *coeffs2, 
                        sfixn nx1, sfixn ny1, const sfixn *coeffs1);

void expand_to_list_fft2_dev(sfixn q, sfixn ex2, sfixn ey2, sfixn *coeffs2, 
                        sfixn nx1, sfixn ny1, const sfixn *coeffs1);

void extract_from_fft2_dev(sfixn nx1, sfixn ny1, sfixn *coeff1,
                           sfixn ex2, const sfixn *coeff2);

/* matrix transposition */
void transpose_dev(sfixn *odata, const sfixn *idata, sfixn w, sfixn h);

/*
 * Classic polynomial multiplication for univariate polynomials
 * 
 * Assume dH = dF + dG
 * 
 **/
void plain_polymul_uni_host(sfixn dH, sfixn *H, sfixn dF, 
    const sfixn *F, sfixn dG, const sfixn *G, sfixn p);

/**
 * Check if two univariate polynomials are the same.
 */
bool is_the_same_poly_host(sfixn d, const sfixn *H, const sfixn *G);

void check_polymul(sfixn d1, sfixn d2, sfixn p); 

bool has_zero_in_vector(const sfixn *Ad, sfixn n);

#endif
