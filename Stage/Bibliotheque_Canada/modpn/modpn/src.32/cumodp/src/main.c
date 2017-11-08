#include <stdio.h>
#include <stdlib.h>

#include "../include/cumodp.h"
#include "../include/printing.h"

void device_query_tst() {
    printf("Has cuda devices   : %d\n", is_cuda_enabled());
    printf("Is double enabled  : %d\n", is_double_float_enabled());
    printf("Global memory size : %.2f(MB)\n", global_memory_in_megabytes());
}

void uni_fft_tst() {
    
    sfixn i, p = 469762049, k = 24, n = (1L << k), w = 37;
    sfixn *X = (sfixn *)malloc(n*sizeof(sfixn));
    for (i = 0; i < n; ++i) X[i] = i % p;
    cumodp_err err = cumodp_fft_uni(X, n, k, w, p);
    
    if (err != CUMODP_SUCCESS) {
        fprintf(stdout, "=====================\n");
        fprintf(stderr, "Fail to do fft\n");
        fprintf(stdout, "=====================\n");
        free(X); return;
    }

    err = cumodp_invfft_uni(X, n, k, w, p);
    if (err != CUMODP_SUCCESS) {
        fprintf(stdout, "=======================\n");
        fprintf(stderr, "Fail to do inverse fft\n");
        fprintf(stdout, "=======================\n");
        free(X); return;
    }

    for (i = 0; i < n; ++i) {
        if (X[i] != i) {
            fprintf(stderr, "Incorrect result for fft & inverse fft\n");
            free(X); return;
        }
    }
    free(X);
}

void fftmul_uni_tst() {
    sfixn i, p = 469762049;
    sfixn m = 1000000, n = 1000000;
    //sfixn m = 800, n = 800;
 
    sfixn *F = (sfixn*) malloc(m*sizeof(sfixn));
    sfixn *G = (sfixn*) malloc(n*sizeof(sfixn));
    sfixn *H = (sfixn*) malloc((m+n-1)*sizeof(sfixn));
 
    for (i = 0; i < m; ++i) F[i] = (i + 3) % p;
    for (i = 0; i < n; ++i) G[i] = (i * 3 + 2) % p;
    for (i = 0; i < m + n - 1; ++i) H[i] = 0;

    printf("F := "); print_poly(m - 1, F, 'x'); printf(":\n");
    printf("G := "); print_poly(n - 1, G, 'x'); printf(":\n");
    cumodp_fftmul_uni(m + n - 2, H, m - 1, F, n - 1, G, p);
    printf("H := "); print_poly(m + n - 2, H, 'x'); printf(":\n");
    printf("Expand(H - F * G) mod %d;\n", p);

    free(F); 
    free(G); 
    free(H); 
}

void bivariate_fft_tst() {

    sfixn W[] = {16, 4, 8, 17, 11, 9, 3};

    sfixn em = 5;
    sfixn en = em;
    sfixn m = (1 << em);
    sfixn n = (1 << en);
    sfixn wn = W[en-1];
    sfixn wm = W[em-1];
    sfixn p = 257;

    sfixn *X = (sfixn*) malloc(sizeof(sfixn)*m*n);
    sfixn i;
    
    for (i = 0; i < m * n; ++i) { X[i] = i % p; }
    cumodp_fft_bivariate(X, em, wm, en, wn, p);
    cumodp_invfft_bivariate(X, em, wm, en, wn, p);
    for (i = 0; i < m * n; ++i) {
        if (X[i] != i % p) {
            fprintf(stderr, "Incorrect result for bivariate FFTs\n");
            break;
        }
    }
    free(X);
}

void subres2_chain_tst() {
    sfixn m = 10;
    sfixn n = m;
    sfixn B = 256;
    sfixn w = 3;
    sfixn p = 257;
    sfixn *T, *S = (sfixn*)malloc(sizeof(sfixn)*B*(n-1)*n/2);
    sfixn *P = (sfixn*)malloc(sizeof(sfixn)*m*n);
    sfixn *Q = (sfixn*)malloc(sizeof(sfixn)*m*n);
    cumodp_err err;
    sfixn i, j;

    for (i = 0; i < m * n; ++i) {
        P[i] = i % p; 
        Q[i] = (i + 1) % p;
    }
    
    err = cumodp_subres_chain2_coarse(S, B, w, m, m, P, n, n, Q, p);
    
    //T = S + (n * n - n) / 2;
    //for (i = n - 1; i >= 1; --i) {
    //    for (j = 0; j < i; ++j) { printf("%3d ", T[j]); }
    //    T += i; 
    //    printf("\n");
    //}

    if (err == CUMODP_SUCCESS) {
        printf("Pass bivariate coarse subresultant chain\n");
    } else {
        printf("Fail bivariate coarse subresultant chain\n");
    }

    free(P);
    free(Q);
    free(S);
}

void subres3_chain_tst() {
    sfixn nx = 5;
    sfixn ny = 5;
    sfixn nz = 5;
    sfixn Bx = 32;
    sfixn By = 32;
    sfixn wx = 528992597;
    sfixn wy = 528992597;
    sfixn p = 962592769;
    sfixn *T, *S = (sfixn*)malloc(sizeof(sfixn)*Bx*By*(nz-1)*nz/2);
    sfixn *P = (sfixn*)malloc(sizeof(sfixn)*nx*ny*nz);
    sfixn *Q = (sfixn*)malloc(sizeof(sfixn)*nx*ny*nz);
    sfixn i, j;
    cumodp_err err;

    for (i = 0; i < nx * ny * nz; ++i) {
        P[i] = i % p; 
        Q[i] = (i + 1) % p;
    }

    // err = cumodp_subres_chain3_coarse(S, Bx, By, wx, wy, nx, ny, nz, P, nx, ny, nz, Q, p);
    err = cumodp_subres_chain3_fine(S, Bx, By, wx, wy, nx, ny, nz, P, nx, ny, nz, Q, p);

    // the data-layout is different for coarse- or fine- grained algorithms!
    // coarse
    //printf("coarse ::\n");
    //T = S; 
    //for (i = nz - 1; i >= 1; --i) {
    //    for (j = 0; j < i; ++j) { printf("%3d ", T[j]); }
    //    T += i; 
    //    printf("\n");
    //} 
    //printf("fine ::\n");
    //T = S; 
    //for (i = nz - 1; i >= 1; --i) {
    //    for (j = 0; j < i; ++j) { printf("%3d ", T[j]); }
    //    T += Bx*By*i; 
    //    printf("\n");
    //}

    if (err == CUMODP_SUCCESS) {
        printf("Pass trivariate fine subresultant chain\n");
    } else {
        printf("Fail trivariate fine subresultant chain : ");
        println_error(err);
    }

    free(P);
    free(Q);
    free(S);
}

void cuda_scube_tst() {
    sfixn sz_p[2] = {11, 11};
    sfixn sz_q[2] = {11, 11};
    sfixn fp = 13313;
    sfixn *P, *Q, i;
    sfixn nx;
    const sfixn *coeff;
    P = (sfixn*) malloc(sizeof(sfixn)*sz_p[0]*sz_p[1]);
    Q = (sfixn*) malloc(sizeof(sfixn)*sz_q[0]*sz_q[1]);
    for (i = 0; i < sz_p[0] * sz_p[1]; ++i) P[i] = rand() % fp;
    for (i = 0; i < sz_q[0] * sz_q[1]; ++i) Q[i] = rand() % fp;
    void* scb = init_cuda_scube(2, sz_p, sz_q, fp);
    if (scb != NULL) {
        cumodp_err err = build_cuda_scube(scb, sz_p, P, sz_q, Q);
        if (err != CUMODP_SUCCESS) { printf("FFT based method failed\n"); }
        coeff = interp_subres_coeff2(&nx, scb, 0, 0);
        for (i = 0; i < nx; ++i) printf("%d, ", coeff[i]);
        printf("\n");
        print_cuda_scube(scb);
        free_cuda_scube(scb);
    }
    free(P);
    free(Q);
}

int main(int argc, char** argv) {
    cuda_scube_tst();
    return 0;
}
