#ifndef _CU_MODP_H_
#define _CU_MODP_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

//////////////////////
// CUDA Device Query
//////////////////////
int is_cuda_enabled();

int num_of_cuda_devices(); 

int is_double_float_enabled();

unsigned int global_memory_in_bytes();

float global_memory_in_megabytes();

int can_call_fftmul_uni(sfixn df, sfixn dg);

int can_call_subres2(sfixn dx, sfixn d1, sfixn d2);

int can_call_subres3(sfixn dx, sfixn dy, sfixn d1, sfixn d2);

///////////////////////////////
//  FFT & FFT Multiplications
///////////////////////////////
cumodp_err 
cumodp_fft_uni(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p);

cumodp_err 
cumodp_invfft_uni(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p);

cumodp_err
cumodp_fft_bivariate(sfixn *X, sfixn em, sfixn wm, sfixn en, 
    sfixn wn, sfixn p);

cumodp_err
cumodp_invfft_bivariate(sfixn *X, sfixn em, sfixn wm, sfixn en, 
    sfixn wn, sfixn p);

cumodp_err
cumodp_fftmul_uni(sfixn dh, sfixn *H, sfixn df, const sfixn *F, 
    sfixn dg, const sfixn *G, sfixn p);

///////////////////////////////////////////
// subresultant chain construction for CPU
///////////////////////////////////////////
sfixn
cumodp_subres_chain2_fine(sfixn *S, sfixn B, sfixn w, 
    sfixn npx, sfixn npy, const sfixn *P, 
    sfixn nqx, sfixn nqy, const sfixn *Q, sfixn p);

sfixn
cumodp_subres_chain2_coarse(sfixn *S, sfixn B, sfixn w, 
    sfixn npx, sfixn npy, const sfixn *P, 
    sfixn nqx, sfixn nqy, const sfixn *Q, sfixn p);

sfixn
cumodp_subres_chain3_fine(sfixn *S, sfixn Bx, sfixn By, sfixn wx, sfixn wy,
    sfixn npx, sfixn npy, sfixn npz, const sfixn *P, 
    sfixn nqx, sfixn nqy, sfixn nqz, const sfixn *Q, sfixn p); 

sfixn
cumodp_subres_chain3_coarse(sfixn *S, sfixn Bx, sfixn By, sfixn wx, sfixn wy,
    sfixn npx, sfixn npy, sfixn npz, const sfixn *P, 
    sfixn nqx, sfixn nqy, sfixn nqz, const sfixn *Q, sfixn p);

///////////////////////////////////////////
// subresultant chain construction for GPU
///////////////////////////////////////////

void *init_cuda_scube(sfixn N, const sfixn *sz_p, const sfixn *sz_q, sfixn fp);

void free_cuda_scube(void *scb);

void print_cuda_scube(const void *scb);

cumodp_err 
build_cuda_scube(void *scb, const sfixn *sz_p, const sfixn *P,
    const sfixn *sz_q, const sfixn *Q);

const sfixn *interp_subres_coeff2(sfixn *nx, void *scb, 
    sfixn i, sfixn j);

const sfixn *interp_subres_coeff3(sfixn *nx, sfixn *ny, void *scb, 
    sfixn i, sfixn j);

#ifdef __cplusplus
}
#endif

#endif
