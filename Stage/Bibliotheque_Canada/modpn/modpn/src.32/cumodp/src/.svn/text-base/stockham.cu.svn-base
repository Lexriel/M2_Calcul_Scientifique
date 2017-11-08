#include <iostream>
#include <cassert>

#include "../include/stockham.h"
#include "../include/fft_aux.h"
#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/cudautils.h"

///////////////////////////////////////////////////////////////////////////////
//  File created at Wed Jul 14 EDT 2010, WP
///////////////////////////////////////////////////////////////////////////////

/**
 *  The Stockham FFT can be computed with the following sequence of operations
 *
 *  (DFT_2 @ I_{2^{k-1}}) (D_{2, 2^{k-i-1}}@I_{2^i}) (L_2^{2^{k-i}}@I_{2^i})
 *
 *  for i from k - 1 down to 0.
 */

///////////////////////////////////////////////////////////////////////////////
//          Implementation of stride permutation, device version             //
///////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input array of length n = 2^k
 * @Y, output array of length n = 2^k
 * @i, index from 0 to k - 1
 *
 * Compute 
 * 
 * Y = (L_2^{2^{k-i}}@I_{2^i}) X 
 *
 * with the case that 
 *
 * s = 2^i >= N_THD 
 *
 * or at least one thread block move s data.
 *
 */
__global__ 
void stride_transpose2a_ker(sfixn *Y, const sfixn * const X, sfixn k, sfixn i)
{
    __shared__ sfixn block[N_THD];

    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // delta = s / N_THD;
    sfixn edelta = i - E_THD;
    // iq = quo(bid, delta) and ir = rem(bid, delta)
    sfixn iq = bid >> edelta;
    sfixn ir = bid & ((1 << edelta) - 1);
    // iqq = quo(iq, 2) and iqr = rem(iq, 2)
    sfixn iqq = (iq >> 1);
    sfixn iqr = (iq & 1);

    // read in data from X
    // the input offset for this block is iq * s + ir * N_THD
    const sfixn *din = X + (iq << i) + (ir << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();

    // write data out to Y
    // the output offset for this block is
    // rem(iq, 2) * n / 2 + quo(iq, 2) * s + ir * N_THD
    sfixn *dout = Y + (iqr << (k - 1)) +  (iqq << i) + (ir << E_THD);
    dout[threadIdx.x] = block[threadIdx.x];
    __syncthreads();
}

/**
 * @X, input array of length n = 2^k
 * @Y, output array of length n = 2^k
 * @i, index from 0 to k - 1
 *
 * Compute 
 * 
 * Y = (L_2^{2^{k-i}}@I_{2^i}) X 
 *
 * with the case that 
 *
 * s = 2^i < N_THD 
 *
 * or one thread block moves more than one s data.
 *
 */
__global__ void 
stride_transpose2b_ker(sfixn *Y, const sfixn * const X, sfixn k, sfixn i) 
{
    __shared__ sfixn block[N_THD];
    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // read in data from X
    // the input offset for this block is bid * N_THD
    const sfixn *din = X + (bid << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();
    
    // the following code is to the in-block shuffle, 
    // hard to explain and check the note

    sfixn tid = threadIdx.x;
    // iq = quo(tid, s) and ir = rem(tid, s);
    sfixn iq = tid >> i;
    sfixn ir = tid & ((1 << i) - 1);
    // f(i) = (rem(2iq, N_THD/s) + quo(2iq, N_THD/s)) * s + ir
    sfixn fi = (iq << 1) >> (E_THD - i);
    fi += (iq << 1) & ((1 << (E_THD - i)) - 1);
    fi <<= i;
    fi += ir;
      
    //if (tid < N_THD/2)
    //    dout[tid] = block[fi];
    //else
    //    dout[tid - N_THD / 2 + (1 << (k-1))] = block[fi];
        
    // offset0 = bid * N_THD / 2
    // offset1 = bid * N_THD / 2 + n / 2
    // Yb = Y + offset0
    sfixn *Yb = Y + (bid << (E_THD - 1));
    sfixn *dout = Yb + (tid >> (E_THD - 1)) * ((1 << (k - 1)) 
                     - (1 << (E_THD - 1)));
    dout[tid] = block[fi];
    __syncthreads();
}

/**
 * @X, device array of length n = 2^k
 * @Y, device array of length n = 2^k (output)
 * @i, the exponent of the stride s = 2^i 
 *
 * Require  0 <= i <= k - 2
 */
void stride_transpose2_dev(sfixn *Y, const sfixn * const X, sfixn k, sfixn i)
{
    if (DEBUG) assert((i >= 0) && (i < k - 1) && (k >= E_THD));

    sfixn nb = ((sfixn)1 << (k - E_THD));
    dim3 nBlk(nb, 1, 1);
    // the maximal possible dimension is 2^15 = 32768 < 65535
    if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }

    if (i >= E_THD) {
        stride_transpose2a_ker<<<nBlk, N_THD>>>(Y, X, k, i);
    } else {
        stride_transpose2b_ker<<<nBlk, N_THD>>>(Y, X, k, i);
    }
    cudaThreadSynchronize();
}
 
#undef E_THD
#undef N_THD

///////////////////////////////////////////////////////////////////////////////
//                      D_{2, 2^{k-i-1}} @ I_{2^i}                           // 
///////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input/output array of length n = 2^k
 * @W, array of primitive roots
 * @i, step index
 *
 * Compute X = (D_{2, 2^{k - i - 1}} @ I_{2^i}) X, where
 *
 * D_{2, 2^{k - i - 1}} is a matrix of size 2^{k-i} X 2^{k-i}  
 *
 * ------------------------------------------------------------
 *
 * For example, let w^8 = -1 be a 16-th primitive root of unity
 *
 * i = 0, D_{2, 8} = 
 *
 * [ 1                                 ]
 * [   1                               ]
 * [     1                             ]
 * [       1                           ]
 * [         1                         ]
 * [           1                       ]
 * [             1                     ]
 * [               1                   ]
 * [                 1                 ] 
 * [                   w               ]
 * [                     w^2           ]
 * [                       w^3         ]
 * [                         w^4       ]
 * [                           w^5     ]
 * [                             w^6   ]
 * [                               w^7 ]
 * 
 * i = 1, D_{2, 4} =   
 *            
 * [ 1                   ]            
 * [   1                 ]            
 * [     1               ]            
 * [       1             ]            
 * [         1           ]            
 * [           w^2       ]            
 * [              w^4    ]      
 * [                 w^6 ] 
 *
 * and i = 2, D_{2, 2} = 
 *
 * [ 1         ]           
 * [   1       ]           
 * [     1     ]           
 * [       w^4 ]
 *
 * Hence the primitive root of unity will be used as follows:
 *
 * i = 2, [1, w^4]
 * i = 1, [1, w^2, w^4, w^6]
 * i = 0, [1, w, w^2, w^3, w^4, w^5, w^6, w^7]
 *
 * ------------------------------------------------------------
 *
 * D_{2, 2^{k - i - 1}} @ I_{2^i} is the diagonal matrix with each entry 
 * repeated 2^i times, resulting a matrix of size 2^k X 2^k
 * 
 * Each time, only half of data will be touched since the first half
 * of the diagonal matrix consists of 1 only. The total number of blocks
 * is 2^{k-1} / N_THD. 
 *
 */

/**
 * Multiple thread blocks (>1) handle a stride
 *
 * the number of blocks is s / N_THD
 *
 * Require k > i > E_THD. 
 *
 */
__global__ void stride_twiddle2a_ker(sfixn *X, const sfixn * const W, sfixn k, 
                                     sfixn i, sfixn p, double pinv)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // root used for the thread block is wn^(s*e), with e = quo(bid, s / N_THD)
    sfixn w = W[(bid >> (i - E_THD)) << i];
    // starting position for this block
    // the first 2^(k - i - 1) * 2^i = 2^(k - 1) elements will be unchanged.
    sfixn *base = X + ((sfixn)1 << (k - 1)) + (bid << E_THD);
    sfixn tid = threadIdx.x;
    base[tid] = mul_mod(w, base[tid], p, pinv);
}

/**
 * One thread blocks handles multiple strides
 *
 * the number of strides is N_THD / s
 *
 * Require 0 <= i <= E_THD. 
 *
 */
__global__ void stride_twiddle2b_ker(sfixn *X, const sfixn * const W, 
                                     sfixn k, sfixn i, sfixn p, double pinv) 
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // starting position for this block
    // the first 2^(k - i - 1) * 2^i = 2^(k - 1) elements will be unchanged.
    sfixn *base = X + ((sfixn)1 << (k - 1)) + (bid << E_THD);
    // the starting root for the thread block is wn^(e * s) 
    // with e = bid * (N_THD / s). Thus e * s = bid * N_THD. 
    sfixn tid = threadIdx.x;
    // the exponent is e * s + s * quo(tid, s) 
    sfixn iq = (bid << E_THD) + ((tid >> i) << i);
    base[tid] = mul_mod(W[iq], base[tid], p, pinv);
}

/**
 * @X, input/output array of length 2^k
 * @W, array of primitive roots
 * @i, step index
 *
 * Require 0 <= i <= k - 2
 */
void 
stride_twiddle2_dev(sfixn *X, const sfixn * const W, sfixn k, sfixn i, sfixn p)
{
    if (DEBUG) assert((i >= 0) && (i < k - 1) && (k > E_THD));
    sfixn nb = (sfixn(1) << (k - 1 - E_THD));
    dim3 nBlk(nb, 1, 1);
    // the maximal possible dimension is 2^15 = 32768 < 65535
    if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); } 
    
    double pinv = 1 / (double)p;
    if (i > E_THD) {
        stride_twiddle2a_ker<<<nBlk, N_THD>>>(X, W, k, i, p, pinv);
    } else {
        stride_twiddle2b_ker<<<nBlk, N_THD>>>(X, W, k, i, p, pinv);
    }
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD

///////////////////////////////////////////////////////////////////////////////
//                Implementation of the butterfly operations                 //
///////////////////////////////////////////////////////////////////////////////

#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

// a butterfly operation is defined as
//
//   x0  y0
//     \/
//     /\
//   xs  ys
//
//  with y0 = x0 + ss and ys = x0 - xs. 
//  In total, 2 + 2 elements are involved for each butterfly

/**
 * @X, device array of length n = 2^k
 * @Y, device array of length n = 2^k (output)
 *
 * DFT2 @ I_{2^{k - 1}} 
 *
 */
__global__ 
void butterfly_ker(sfixn *Y, const sfixn * const X, sfixn k, sfixn p) 
{
    // block id, 
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn halfn = ((sfixn )1 << (k - 1));
    sfixn *B = Y + (bid << E_THD); 
    const sfixn *A = X + (bid << E_THD);
    B[threadIdx.x] = add_mod(A[threadIdx.x], A[threadIdx.x + halfn], p);
    B[threadIdx.x + halfn] = sub_mod(A[threadIdx.x], A[threadIdx.x + halfn], p);
}

void butterfly_dev(sfixn *Y, const sfixn * const X, sfixn k, sfixn p) {
    // the smallest FFT size would be 16, E_THD = 3;
    if (k <= E_THD) {
        if (DEBUG) assert(k >= 4);
        sfixn nb = ((sfixn)1 << (k - 4));
        dim3 nBlk(nb, 1, 1);
        butterfly_ker<<<nBlk, 8>>>(Y, X, k, p);
        cudaThreadSynchronize();
    } else {
        sfixn nb = ((sfixn)1 << (k - E_THD - 1));
        dim3 nBlk(nb, 1, 1);
        if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }
        butterfly_ker<<<nBlk, N_THD>>>(Y, X, k, p);
        cudaThreadSynchronize();
    }
    if (DEBUG) checkCudaError("butterfly_dev");
}

#undef E_THD
#undef N_THD
/////////////////////////////
//       Stockham FFT      //
/////////////////////////////

/**
 * @X, input data array of length n = 2^k residing in the device
 * @w, n-th primitive root of unity
 * @p, fourier prime number
 *
 * X will be filled by DFT_n(X)
 *
 */
void stockham_dev(sfixn *X_d, sfixn n, sfixn k, sfixn w, sfixn p) 
{
    // Compute powers of the root
    sfixn *W_d;
    cudaMalloc((void **)&W_d, sizeof(sfixn) << (k - 1));
    get_powers_binary(k - 1, W_d, w, p);

    sfixn *Y_d;
    cudaMalloc((void **)&Y_d, sizeof(sfixn) * n);
    butterfly_dev(Y_d, X_d, k, p);
    for (sfixn i = k - 2; i >= 0; --i) {
        stride_transpose2_dev(X_d, Y_d, k, i);
        stride_twiddle2_dev(X_d, W_d, k, i, p);
        butterfly_dev(Y_d, X_d, k, p);
    }
    cudaMemcpy(X_d, Y_d, sizeof(sfixn)*n, cudaMemcpyDeviceToDevice);

    cudaFree(W_d);
    cudaFree(Y_d);

    if (DEBUG) checkCudaError("error found in stockham_dev");
}

/* Powers of root have been precomputed. */
void stockham_dev(sfixn *X_d, sfixn n, sfixn k, const sfixn *W_d, sfixn p)
{
    // sequence of applications 
    sfixn *Y_d;
    cudaMalloc((void **)&Y_d, sizeof(sfixn) * n);

    butterfly_dev(Y_d, X_d, k, p);
    for (sfixn i = k - 2; i >= 0; --i) {
        stride_transpose2_dev(X_d, Y_d, k, i);
        stride_twiddle2_dev(X_d, W_d, k, i, p);
        butterfly_dev(Y_d, X_d, k, p);
    }
    cudaMemcpy(X_d, Y_d, sizeof(sfixn)*n, cudaMemcpyDeviceToDevice);

    cudaFree(Y_d);
    if (DEBUG) checkCudaError("error found in stockham_dev");
}

/**
 * @X, input data array of length n = 2^k residing in the host 
 * @w, n-th primitive root of unity
 * @p, fourier prime number
 *
 * X will be filled by DFT_n(X)
 */
void inverse_stockham_dev(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p) {
    sfixn winv = inv_mod(w, p);
    sfixn ninv = inv_mod(n, p);
    stockham_dev(X, n, k, winv, p);
    scale_vector_dev(ninv, n, X, p);
}

/**
 *  W is the inversed primitive roots of unity 
 */
void inverse_stockham_dev(sfixn *X, sfixn n, sfixn k, const sfixn *W, sfixn p) 
{
    sfixn ninv = inv_mod(n, p);
    stockham_dev(X, n, k, W, p);
    scale_vector_dev(ninv, n, X, p);
}

void stockham_host(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p) {
    sfixn *X_d;
    cudaMalloc((void **)&X_d, sizeof(sfixn) * n);
    cudaMemcpy(X_d, X, sizeof(sfixn) * n, cudaMemcpyHostToDevice);
    ///////////////////////////////////////
    stockham_dev(X_d, n, k, w, p);
    ///////////////////////////////////////
    cudaMemcpy(X, X_d, sizeof(sfixn) * n, cudaMemcpyDeviceToHost);
    cudaFree(X_d);
}

void inverse_stockham_host(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p) 
{
    sfixn *X_d;
    cudaMalloc((void **)&X_d, sizeof(sfixn) * n);
    cudaMemcpy(X_d, X, sizeof(sfixn) * n, cudaMemcpyHostToDevice);
    ///////////////////////////////////////
    sfixn winv = inv_mod(w, p);
    sfixn ninv = inv_mod(n, p);
    stockham_dev(X_d, n, k, winv, p);
    scale_vector_dev(ninv, n, X_d, p);
    ///////////////////////////////////////
    cudaMemcpy(X, X_d, sizeof(sfixn) * n, cudaMemcpyDeviceToHost);
    cudaFree(X_d);
}

///////////////////////////////////////////////////////////////////////////////
//        FFT based fast polynomial multiplication over finite fields        //
///////////////////////////////////////////////////////////////////////////////

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
 *
 **/
void stockham_poly_mul_dev(sfixn n, sfixn e, sfixn *F, sfixn *G, sfixn p) 
{
    sfixn w = primitive_root(e, p);
    sfixn winv = inv_mod(w, p);
    sfixn ninv = inv_mod(n, p);
    
    sfixn *W;
    cudaMalloc((void **)&W, sizeof(sfixn) << (e - 1));
    get_powers_binary(e - 1, W, w, p);

    stockham_dev(F, n, e, W, p); 

    stockham_dev(G, n, e, W, p); 
    pointwise_mul_dev(n, e, F, G, p);
    get_powers_binary(e - 1, W, winv, p);
    stockham_dev(F, n, e, W, p);
    scale_vector_dev(ninv, n, F, p);

    cudaFree(W);

    if (DEBUG) checkCudaError("stockham_poly_mul_dev");
}

// Cutoff values
#define GPU_FFT_UNIVARIATE_MUL_CUTOFF (1 << 8)

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
 * into H. Otherwise, dh <= df + dg, dh + 1 slots will be filled into H.
 *
 * The static cutoff value to be determined. 
 */
void stockham_poly_mul_host(sfixn dh, sfixn *H, sfixn df, const sfixn *F, 
                            sfixn dg, const sfixn *G, sfixn p) 
{
    // the size of the output
    sfixn s = df + dg + 1; 
    sfixn e = ceiling_log2(s);
    sfixn n = (1L << e);
    sfixn mem_size = (sizeof(sfixn) << e);
    sfixn *F_d, *G_d;

    if (n < GPU_FFT_UNIVARIATE_MUL_CUTOFF) {
        fprintf(stderr, "fft size < %d is not supported by gpu\n", 
            GPU_FFT_UNIVARIATE_MUL_CUTOFF);
        return;
    }

    // move data in
    cudaMalloc((void **)&F_d, mem_size);
    cudaMalloc((void **)&G_d, mem_size);

    cudaMemcpy(F_d, F, sizeof(sfixn)*(1 + df), cudaMemcpyHostToDevice);
    cudaMemcpy(G_d, G, sizeof(sfixn)*(1 + dg), cudaMemcpyHostToDevice);

    expand_to_fft_dev(n, df + 1, F_d);
    expand_to_fft_dev(n, dg + 1, G_d);
    
    // compute the product
    stockham_poly_mul_dev(n, e, F_d, G_d, p); 

    // move data out
    if (dh >= s) {
        cudaMemcpy(H, F_d, mem_size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(H, F_d, sizeof(sfixn)*(1 + dh), cudaMemcpyDeviceToHost);
    }

    cudaFree(F_d);
    cudaFree(G_d);
}

///////////////////////////////////////////////////////////////////////////////
// Functions for testing
///////////////////////////////////////////////////////////////////////////////

void test_stockham(int argc, char** argv) {
    //const sfixn p = 469762049;
    sfixn p = 257;
    sfixn k = 8; 
    if (argc > 1) { k = atoi(argv[1]); }

    sfixn n = ((sfixn)1 << k);
    sfixn *X = new int[n];
    sfixn w = primitive_root(k, p);
    sfixn invw = inv_mod(w, p);
    for (int i = 0; i < n; ++i) X[i] = i;

    print_vector(n, X);
    printf("w = %d\n", w);
    stockham_host(X, n, k, w, p);
    print_vector(n, X);

    delete [] X;
}

void test_poly_mul() {
    sfixn d = 64;
    sfixn *F = new sfixn[d+1];
    sfixn *G = new sfixn[d+1];
    for(int i = 0; i <= d; ++i) { F[i] = i; G[i] = i + 1; }
    sfixn d2 = 2*d;
    sfixn *H = new sfixn[d2+1]();
    sfixn p = 257;

    printf("Input polys : \n");
    println_poly(d, F, 'x'); 
    println_poly(d, G, 'x'); 

    stockham_poly_mul_host(d2, H, d, F, d, G, p);

    printf("The product of input polys : \n");
    println_poly(d2, H, 'x');

    delete [] F;
    delete [] G;
    delete [] H;
}

void test_poly_mul(sfixn d) {
    sfixn *F = new sfixn[d+1];
    sfixn *G = new sfixn[d+1];
    for(int i = 0; i <= d; ++i) { F[i] = i; G[i] = i + 1; }
    sfixn d2 = 2*d;
    sfixn *H = new sfixn[d2+1]();
    const sfixn p = 469762049;

    clock_t t;
    t = clock();
    stockham_poly_mul_host(d2, H, d, F, d, G, p);
    t = clock() - t;
    
    double tis = (double)t / (double)CLOCKS_PER_SEC;

    printf("d = %10d\t%8.3f seconds\n", d, tis);

    delete [] F;
    delete [] G;
    delete [] H;
}
