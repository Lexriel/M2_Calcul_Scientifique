#ifndef _SUBRES_H_
#define _SUBRES_H_

#include <iostream>
#include <cassert>
#include "defines.h"
#include "inlines.h"
#include "unipoly.h"
#include "fft_aux.h"
#include "stockham.h"
#include "list_stockham.h"
#include "cudautils.h"
#include "printing.h"

/**
 * Compute subresultant chains subres(P, Q, y).
 *
 * @S   : (Output) the subresultant chain, host array
 * @B   : the degree bound of the resultant
 * @w   : B-th primitive root of unity modulo p
 * @P   : bivariate rdr-poly over Z_p[x, y], host array
 * @Q   : bivaraite rdr-poly over Z_p[x, y], host array
 * @npx : the partial size of P in x
 * @npy : the partial size of P in y
 * @nqx : the partial size of Q in x
 * @nqy : the partial size of Q in y
 *
 * Assume that npy >= nqy holds and S is of size B * (nqy - 1) * nqy / 2.
 * Assume that none of initials of P and Q are zero on the powers of w.
 *
 * The resulting S will have the data following layout (B = 4, nqy = 6)
 * 
 * S4(1)    S3(1)    S2(1)    S1(1)    S0(1)  
 * S4(w)    S3(w)    S2(w)    S1(w)    S0(w)  
 * S4(w^2)  S3(w^2)  S2(w^2)  S1(w^2)  S0(w^2)  
 * S4(w^3)  S3(w^3)  S2(w^3)  S1(w^3)  S0(w^3) 
 *
 * The total size is 4 * (5 + 4 + 3 + 2 + 1) = 60. 
 *
 * To interpolate the subresultants, matrix transpositions are needed.
 */
sfixn subres_chain2_coarse_host(sfixn *S, sfixn B, sfixn w, 
    sfixn npx, sfixn npy, const sfixn *P, 
    sfixn nqx, sfixn nqy, const sfixn *Q, sfixn p);

sfixn subres_chain3_coarse_host(sfixn *S, sfixn Bx, sfixn By, 
    sfixn wx, sfixn wy, sfixn npx, sfixn npy, sfixn npz, const sfixn *P, 
    sfixn nqx, sfixn nqy, sfixn nqz, const sfixn *Q, sfixn p);

sfixn subres_chain2_fine_host(sfixn *S, sfixn B, sfixn w,
    sfixn npx, sfixn npy, const sfixn *P, 
    sfixn nqx, sfixn nqy, const sfixn *Q, sfixn p); 

sfixn subres_chain3_fine_host(sfixn *S, sfixn Bx, sfixn By, 
    sfixn wx, sfixn wy, sfixn npx, sfixn npy, sfixn npz, const sfixn *P, 
    sfixn nqx, sfixn nqy, sfixn nqz, const sfixn *Q, sfixn p);

/**
 * Coarse-grained subresultant chain construction for a list of
 * univariate polynomials of the same shape.
 */
bool coarse_subres_uni_dev(sfixn *S, sfixn Bd, sfixn dgP, 
    const sfixn *LP, sfixn dgQ, const sfixn *LQ, sfixn p);

/**
 * Fine-grained subresultant chain construction for a list of
 * univariate polynomials of the same shape.
 */
cumodp_err fine_subres_uni_dev(sfixn *S, sfixn Bd, sfixn dgP, 
    const sfixn *LP, sfixn dgQ, const sfixn *LQ, sfixn p);

#endif // END_OF_FILE
