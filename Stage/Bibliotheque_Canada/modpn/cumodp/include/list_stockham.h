#ifndef _LIST_STOCKHAM_H_
#define _LIST_STOCKHAM_H_

#include "../include/defines.h"
#include "../include/rdr_poly.h"

void list_stockham_host(sfixn *X, sfixn m, sfixn k, sfixn w, sfixn p);

/**
 * @X, input / output device array of length m * n 
 * @m, number of items in the list
 * @k, n = 2^k
 * @w, n-th primitive root of unity
 * @W, [1, w, w^2, ..., w^{n/2-1}]
 * @p, fourier prime number
 *
 * X will be filled by I_m @ DFT_n(X, w)
 *
 * :::Warning::: m could be a non-power of 2
 *
 */
void list_stockham_dev(sfixn *X, sfixn m, sfixn k, const sfixn *W, sfixn p);
void list_stockham_dev(sfixn *X, sfixn m, sfixn k, sfixn w, sfixn p);

/**
 * @X, input / output device array of length n * m 
 * @en, n = 2^en
 * @em, m = 2^em
 * @wn, n-th primitive root of unity
 * @Wn, [1, wn, wn^2, ..., wn^{n/2-1}]
 * @wm, m-th primitive root of unity
 * @Wm, [1, wm, wm^2, ..., wm^{m/2-1}]
 * @p, fourier prime number
 *
 * Compute X = DFT_{m, n}(X)
 *
 * Requirements: m, n >= 2 * N_THD 
 *
 */
void bivariate_stockham_dev(sfixn *X, sfixn em, const sfixn *Wm, 
    sfixn en, const sfixn *Wn, sfixn p);

void inverse_bivariate_stockham_dev(sfixn *X, sfixn em, const sfixn *Wm, 
    sfixn en, const sfixn *Wn, sfixn p);

void bivariate_stockham_host(sfixn *X, sfixn em, sfixn wm, 
    sfixn en, sfixn wn, sfixn p);

void inverse_bivariate_stockham_host(sfixn *X, sfixn em, sfixn wm, 
    sfixn en, sfixn wn, sfixn p);

/**
 * @X, device array of length q * m * n
 * @q, positive integer, the number of 2d FFTs
 * @em, m = 2^em
 * @en, n = 2^en
 * @wm, m-th primitive root of unity
 * @wn, n-th primitive root of unity
 * @Wm, [1, wm, ..., wm^{m/2-1}]
 * @Wn, [1, wn, ..., wn^{n/2-1}]
 * @p, prime number
 *
 * Compute X = I_q @ DFT_{m, n}(X, wm, wn) 
 *
 * Requirements: q >= 1, m, n >= 2 * N_THD 
 *
 */
void list_bivariate_stockham_dev(sfixn *X, sfixn q, sfixn em, 
    const sfixn *Wm, sfixn en, const sfixn *Wn, sfixn p); 

/*
 * Bivariate FFT multiplication 
 */
rdr_poly* bi_stockham_poly_mul_host(const rdr_poly &F, 
    const rdr_poly &G, sfixn p);

#endif
