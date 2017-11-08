///////////////////////////////////////////////////////////////////////////////
// This file contains functions for debugging
///////////////////////////////////////////////////////////////////////////////
#ifndef _PRINTING_H_
#define _PRINTING_H_

#include <stdio.h>
#include "defines.h"

static inline void print_vector(sfixn N, const sfixn *v) {
    sfixn i;
    if (N < 1) return; 
    printf("size = %d, [", N);
    for (i = 0; i < N - 1 ; ++i) printf("%3d, ", v[i]);
    printf("%3d];\n", v[N-1]);
}

static inline void print_matrix(sfixn w, sfixn h, const sfixn *v) {
    sfixn i, j;
    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) printf("%3d ", v[w * i + j]);
        printf("\n");
    }
}

/*
 * println_poly
 *
 * @d: degree of the polynomial 
 * @A: coefficient vector of the polynomial
 *
 * Output: Print A[0] + A[1]*x + ... + A[t]*x^t to the screen,
 *         where t is the true degree. 
 */
static inline void println_poly(sfixn d, const sfixn *A, char c) {
    sfixn i;
    sfixn t = d; // true degree
    while ( (A[t] == 0) && (t>0) ) --t;
    // The case t=0
    if (t == 0) {
        printf("%u\n", A[0]); return;
    }
    // The case t>0, all terms except the last one
    for (i=0; i<t; ++i) {
        if (A[i] != 0) {
            if (A[i] == 1) {
                switch (i) {
                    case 0:  printf("%u + ", A[i]); break; 
                    case 1:  printf("%c + ", c); break; 
                    default: printf("%c^%u + ", c, i);
                }
            } else {
                switch (i) {
                    case 0:  printf("%u + ", A[i]);   break; 
                    case 1:  printf("%u*%c + ", A[i], c); break; 
                    default: printf("%u*%c^%u + ", A[i], c, i);
                }
            }
        }
    }
    // The last term, t>0
    if (t==1) {
        if (A[t]==1) { 
            printf("%c\n", c);
        } else {
            printf("%u*%c\n", A[t], c);
        }
    } else {
        if (A[t]==1) {
            printf("%c^%u\n", c, t);
        } else {
            printf("%u*%c^%u\n", A[t], c, t);
        }
    }
}

/*
 * print_poly
 *
 * @d: degree of the polynomial 
 * @A: coefficient vector of the polynomial
 *
 * Output: Print A[0] + A[1]*x + ... + A[t]*x^t to the screen,
 *         where t is the true degree. 
 */

static inline void print_poly(sfixn d, const sfixn *A, char c) {
    sfixn i, t = d; // true degree
    
    while ((A[t] == 0) && (t > 0)) --t;

    // The case t=0
    if (t==0) { printf("%u", A[0]); return; }

    // The case t>0, all terms except the last one
    for (i=0; i<t; ++i) {
        if (A[i] != 0) {
            if (A[i] == 1) {
                switch (i) {
                    case 0:  printf("%u + ", A[i]); break; 
                    case 1:  printf("%c + ", c); break; 
                    default: printf("%c^%u + ", c, i);
                }
            } else {
                switch (i) {
                    case 0:  printf("%u + ", A[i]);   break; 
                    case 1:  printf("%u*%c + ", A[i], c); break; 
                    default: printf("%u*%c^%u + ", A[i], c, i);
                }
            }
        }
    }
    // The last term, t>0
    if (t==1) {
        if (A[t]==1) { 
            printf("%c", c);
        } else {
            printf("%u*%c", A[t], c);
        }
    } else {
        if (A[t]==1) {
            printf("%c^%u", c, t);
        } else {
            printf("%u*%c^%u", A[t], c, t);
        }
    }
}

static inline void print_fprime(const fprime_t * const fpp) {
    printf("fpp->p     = %ld\n", fpp->p);
    printf("fpp->c     = %ld\n", fpp->c);
    printf("fpp->r     = %ld\n", fpp->r);
    printf("fpp->npow  = %ld\n", fpp->npow);
    printf("fpp->rpow  = %ld\n", fpp->rpow);
    printf("fpp->rnpow = %ld\n", fpp->rnpow);
}

static inline void print_error(cumodp_err e) {

    switch (e) {
        case CUMODP_SUCCESS :
            printf("success"); break;
        case CUMODP_FAILURE :
            printf("failure"); break;
        case CUMODP_ASSUMPTION_ERROR :
            printf("assumption error"); break;

        // Device Query
        case CUMODP_NO_CUDA_DEVICE :
            printf("no cuda device"); break;
        case CUMODP_HAS_CUDA_DEVICE :
            printf("has cuda device"); break;

        // FFT
        case CUMODP_FFT_ERROR :
            printf("fft error"); break;
        case CUMODP_FOURIER_DEGREE_TOO_SMALL :
            printf("Fourier degree too small"); break;
        case CUMODP_FFT_SIZE_TOO_LARGE :
            printf("fft size too large"); break;
        case CUMODP_FFT_SIZE_TOO_SMALL :
            printf("fft size too small"); break;

        // Subresultant chain
        case CUMODP_SUBRES_ERROR :
            printf("subresultant chain error"); break;
        case CUMODP_SUBRES_COARSE_ERROR :
            printf("coarse subresultant chain error"); break;
        case CUMODP_SUBRES_FINE_ERROR :
            printf("fine subresultant chain error"); break;
        case CUMODP_SUBRES_NON_REGULAR :
            printf("non-regular subresultant chain error"); break;
        // Others
        case CUMODP_UNKNOWN_ERROR :
            printf("unknown error"); break;
        default :
            printf("not capatured error");
    }
}

static inline void println_error(cumodp_err e) {
    print_error(e);
    printf("\n");
}

///////////////////////////////////////////////////////////////////////////////
#endif
