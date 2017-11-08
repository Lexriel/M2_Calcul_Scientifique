#ifndef _STOCKHAM_H_
#define _STOCKHAM_H_

#include "defines.h"

/**
 * X will be filled by DFT_n(X)
 *
 * @X, input data array of length n = 2^k residing in the device
 * @w, n-th primitive root of unity
 * @p, fourier prime number
 */
void stockham_dev(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p);
void stockham_dev(sfixn *X, sfixn n, sfixn k, const sfixn *W, sfixn p);
void inverse_stockham_dev(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p);
void inverse_stockham_dev(sfixn *X, sfixn n, sfixn k, const sfixn *W, sfixn p);

/**
 * X will be filled by DFT_n(X)
 *
 * @X, input data array of length n = 2^k residing in the host
 * @w, n-th primitive root of unity
 * @p, fourier prime number
 */
void stockham_host(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p);
void inverse_stockham_host(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p);

/**
 * Multiply two polynomials of size POT, in place version
 *
 * @n : FFT size
 * @e : n = 2^e
 * @F : coefficient vector of F, padded to size n, input & output
 * @G : coefficient vector of G, padded to size n, input
 * @p : prime number
 *
 * F <-- DFT^{-1}(DFT(F) * DFT(G))
 **/
void stockham_poly_mul_dev(sfixn n, sfixn e, sfixn *F, sfixn *G, sfixn p);

/**
 * Multiply two univariate polynomials
 *
 * @dh, degree of the product
 * @H , coefficient vector of polynomial H = F * G 
 * @df, degree of polynomial F
 * @dg, degree of polynomial G
 * @F , coefficient vector of polynomial F
 * @G , coefficient vector of polynomial G,
 *
 * If dh > df + dg, then only the first s = (df + dg + 1) slots will be filled 
 * into H. Otherwise, dh <= df + dg, first dh + 1 slots will be filled into H.
 */
void stockham_poly_mul_host(sfixn dh, sfixn *H, sfixn df, const sfixn *F, 
    sfixn dg, const sfixn *G, sfixn p); 

#endif
