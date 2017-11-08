#include <cassert>
#include <iostream>
#include "../include/stockham_mont.h"
#include "../include/defines.h"
#include "../include/montmulmod.h"
#include "../include/cudautils.h"

/**
 *  The Stockham FFT can be computed with the following sequence of operations
 *
 *  (DFT_2 @ I_{2^{k-1}}) (D_{2, 2^{k-i-1}}@I_{2^i}) (L_2^{2^{k-i}}@I_{2^i})
 *
 *  for i from k - 1 down to 0.
 */

///////////////////////////////////////////////////////////////////////////////

/**
 * @fp, the Fourier prime struct used in this file
 */
__constant__ fprime_t fp_d;

/**
 *  Initialize of fourier prime structure
 */
static inline void setup_const(const fprime_t * const fpp) {
    cudaMemcpyToSymbol(fp_d, fpp, sizeof(fprime_t));
}

///////////////////////////////////////////////////////////////////////////////
//          Implementation of stride permutation, device version             //
///////////////////////////////////////////////////////////////////////////////

// each thread block consists of N_THD number of threads 
#define E_THD (7)
#define N_THD (1 << E_THD)
// each thread block uses N_SHD integers in the shared memory
#define E_SHD (1 + E_THD)
#define N_SHD (1 << E_SHD)
// the number of data each thread moves
#define N_DAT (1 << (E_SHD - E_THD))

/**
 * @X, input array of length n = 2^k
 * @Y, output array of length n = 2^k
 * @e, index from 0 to k - 1
 *
 * Compute 
 * 
 * Y = (L_2^{2^{k-e}}@I_{2^e}) X 
 *
 * with the case that 
 *
 * s = 2^e >= N_SHD 
 *
 * or at least one thread block move s data.
 *
 */
__global__ 
void stride_transpose2_kernel_a(sfixn *Y, const sfixn * const X, sfixn k, sfixn e)
{
    __shared__ sfixn block[N_SHD];

    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // delta = s / N_SHD;
    sfixn exp_delta = e - E_SHD;
    // iq = quo(bid, delta) and ir = rem(bid, delta)
    sfixn iq = bid >> exp_delta;
    sfixn ir = bid & ((1 << exp_delta) - 1);
    // iqq = quo(iq, 2) and iqr = rem(iq, 2)
    sfixn iqq = (iq >> 1);
    sfixn iqr = (iq & 1);

    // read in data from X
    // the input offset for this block is iq * s + ir * N_SHD
    sfixn i;
    sfixn *shared = block;
    const sfixn *din = X + (iq << e) + (ir << E_SHD);
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        shared[threadIdx.x] = din[threadIdx.x];
        din += N_THD;
        shared += N_THD;
    }
    __syncthreads();

    // write data out to Y
    //
    // the output offset for this block is
    // rem(iq, 2) * n / 2 + quo(iq, 2) * s + ir * N_SHD
    sfixn *dout = Y + (iqr << (k - 1)) +  (iqq << e) + (ir << E_SHD);
    shared = block;
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        dout[threadIdx.x] = shared[threadIdx.x];
        dout += N_THD;
        shared += N_THD;
    }
    __syncthreads();
}

/**
 * @X, input array of length n = 2^k
 * @Y, output array of length n = 2^k
 * @e, index from 0 to k - 1
 *
 * Compute 
 * 
 * Y = (L_2^{2^{k-e}}@I_{2^e}) X 
 *
 * with the case that 
 *
 * s = 2^e < N_SHD 
 *
 * or one thread block moves more than one s data.
 *
 */
__global__ void 
stride_transpose2_kernel_b(sfixn *Y, const sfixn * const X, sfixn k, sfixn e) 
{
    __shared__ sfixn block[N_SHD];
    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // read in data from X
    // the input offset for this block is bid * N_SHD
    sfixn i;
    sfixn *shared = block;
    const sfixn *din = X + (bid << E_SHD);
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        shared[threadIdx.x] = din[threadIdx.x];
        din += N_THD;
        shared += N_THD;
    }
    __syncthreads();
    
    // offset0 = bid * N_SHD / 2
    // offset1 = bid * N_SHD / 2 + n / 2
    // base = Y + offset0
    sfixn *base = Y + (bid << (E_SHD - 1));
    sfixn *dout;

    sfixn tid, iq, ir, fi; 
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        // virtual thread id in each loop 
        tid = (i << E_THD) + threadIdx.x;
        // iq = quo(tid, s) and ir = rem(tid, s);
        iq = tid >> e;
        ir = tid & ((1 << e) - 1);
        // f(i) = (rem(2iq, N_SHD/s) + quo(2iq, N_SHD/s)) * s + ir
        fi = (iq << 1) >> (E_SHD - e);
        fi += (iq << 1) & ((1 << (E_SHD - e)) - 1);
        fi <<= e;
        fi += ir;
      
        //if (tid < N_SHD/2)
        //    dout[tid] = block[fi];
        //else
        //    dout[tid - N_SHD / 2 + (1 << (k-1))] = block[fi];
        
        dout = base + (tid >> (E_SHD-1)) * ((1 << (k-1)) - (1 << (E_SHD-1))); 
        dout[tid] = block[fi];
    }
}

/**
 * @X, device array of length n = 2^k
 * @Y, device array of length n = 2^k (output)
 * @i, the exponent of the stride s = 2^i 
 *
 * Require  0 <= i <= k - 2
 */
void stride_transpose2(sfixn *Y, const sfixn * const X, sfixn k, sfixn i) 
{
    if (DEBUG) assert((i >= 0) && (i < k - 1));
    sfixn nThread = N_THD;
    sfixn nb = ((sfixn)1 << (k - E_SHD));
    dim3 nBlock(nb, 1, 1);
    // the maximal possible dimension is 2^15 = 32768 < 65535
    if (nb > (1 << 15)) { nBlock.x = (1 << 15); nBlock.y = (nb >> 15); }

    if (i >= E_SHD) {
        stride_transpose2_kernel_a<<<nBlock, nThread>>>(Y, X, k, i);
    } else {
        stride_transpose2_kernel_b<<<nBlock, nThread>>>(Y, X, k, i);
    }
    cudaThreadSynchronize();
}
 
#undef E_THD
#undef N_THD
#undef E_SHD
#undef N_SHD
#undef N_DAT

///////////////////////////////////////////////////////////////////////////////
//               Implementation of the twiddle matrix multiply               //
///////////////////////////////////////////////////////////////////////////////

/**
 * @ compute x[i] = frep(w^i) for i from 0 to n - 1 
 */
__device__ __host__ inline 
void get_mont_root_power(sfixn w, sfixn *X, sfixn n, const fprime_t * const fpp) 
{
    X[0] = fpp->r;
    X[1] = w = frep(w, fpp);
    for (sfixn i = 2; i < n; ++i) {
        X[i] = fourier_reduction(X[i-1], w, fpp);
    }
}

#define E_THD (7)
#define N_THD (1 << E_THD)
#define E_DAT (0)
#define N_DAT (1 << E_DAT)
#define E_ELE (E_THD + E_DAT)
#define N_ELE (1 << E_ELE)
/**
 * @X, input/output array of length n = 2^k
 * @W, array of primitive roots
 * @i, step index
 * @fpp, prime structure
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
 * is 2^{k-1} / N_ELE. 
 *
 */

/**
 * stride_twiddle_kernel_a, multiple thread blocks (>1) handle a stride.
 *
 * the number of blocks is  s / N_ELE
 *
 * Require k > i > E_ELE. 
 *
 */
__global__ 
void stride_twiddle_kernel_a(sfixn *X, const sfixn * const W, sfixn k, sfixn i) 
{
    // block index
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // root used for the thread block is wn^(s*e), with e = quo(bid, s/N_ELE)
    sfixn w = W[(bid >> (i - E_ELE)) << i];
    // starting position for this block
    // the first 2^(k - i - 1) * 2^i = 2^(k - 1) elements will be unchanged.
    sfixn *base = X + ((sfixn)1 << (k - 1)) + (bid << E_ELE);
    // virtual thread id
    sfixn tid;
    #pragma unroll
    for (sfixn j = 0; j < N_DAT; ++j) {
        tid = (j << E_THD) + threadIdx.x;
        base[tid] = fourier_reduction(w, base[tid], &fp_d);
    }
}

/**
 * stride_twiddle_kernel_b, one thread blocks handles multiple strides.
 *
 * the number of strides is N_ELE / s
 *
 * Require 0 <= i <= E_ELE. 
 *
 */
__global__ void 
stride_twiddle_kernel_b(sfixn *X, const sfixn * const W, sfixn k, sfixn i) 
{
    // block index
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // starting position for this block
    // the first 2^(k - i - 1) * 2^i = 2^(k - 1) elements will be unchanged.
    sfixn *base = X + ((sfixn)1 << (k - 1)) + (bid << E_ELE);
    // virtual thread id
    sfixn tid;
    // the starting root for the thread block is wn^(e * s) 
    // with e = bid * (N_ELE / s). Thus e * s = bid * E_ELE. 
    sfixn iq, iq_base = (bid << E_ELE);
    #pragma unroll
    for (sfixn j = 0; j < N_DAT; ++j) {
        tid = (j << E_THD) + threadIdx.x;
        // the exponent is iq_base + s * quo(tid, s) 
        iq = iq_base + ((tid >> i) << i);
        base[tid] = fourier_reduction(W[iq], base[tid], &fp_d);
    }
}

/**
 * @X, input/output array of length 2^k
 * @W, array of primitive roots
 * @i, step index
 *
 * Require 0 <= i <= k - 2
 */
void stride_twiddle(sfixn *X, const sfixn * const W, sfixn k, sfixn i, 
                    sfixn p, bool mont)
{
    if (DEBUG) assert((i >= 0) && (i < k - 1) && (k > E_ELE));
    sfixn nThread = (1 << E_THD);
    // nblock is 2^{k-1} / N_ELE
    sfixn nb = (1 << (k - 1 - E_ELE));
    dim3 nBlock(nb, 1, 1);
    // the maximal possible dimension is 2^15 = 32768 < 65535
    if (nb > (1 << 15)) { nBlock.x = (1 << 15); nBlock.y = (nb >> 15); } 
    
    if (i > E_ELE) {
        stride_twiddle_kernel_a<<<nBlock, nThread>>>(X, W, k, i);
    } else {
        stride_twiddle_kernel_b<<<nBlock, nThread>>>(X, W, k, i);
    }
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
#undef E_DAT
#undef N_DAT
#undef E_ELE
#undef N_ELE

///////////////////////////////////////////////////////////////////////////////
//                Implementation of the butterfly operations                 //
///////////////////////////////////////////////////////////////////////////////

// each thread block consists of N_THD number of threads 
#define E_THD (7)
#define N_THD (1 << E_THD)
// the number of butterflyers each thread computes
#define E_BUT (0)
#define N_BUT (1 << E_BUT)
// the number of butterflyers each thread block handles
#define E_ELE (E_THD + E_BUT)
#define N_ELE (1 << E_ELE)

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
void butterfly_kernel(sfixn *Y, const sfixn * const X, sfixn k, sfixn p) {
    // block id, 
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn tid; 
    sfixn halfn = ((sfixn )1 << (k - 1));
    sfixn *B = Y + (bid << E_ELE); 
    const sfixn *A = X + (bid << E_ELE);
    
    #pragma unroll
    for (sfixn i = 0; i < N_BUT; ++i) {
        // virtual thread id
        tid = (i << E_THD) + threadIdx.x;
        B[tid] = add_mod(A[tid], A[tid + halfn], p);
        B[tid + halfn] = sub_mod(A[tid], A[tid + halfn], p);
    }
}

void butterfly(sfixn *Y, const sfixn * const X, sfixn k, sfixn p) {
    sfixn nThread = ((sfixn)1 << (E_THD));
    sfixn nb = ((sfixn)1 << (k - E_ELE - 1));
    dim3 nBlock(nb, 1, 1);
    if (nb > (1 << 15)) { nBlock.x = (1 << 15); nBlock.y = (nb >> 15); }
    
    if (DEBUG) assert(k >= E_ELE + 1);
    
    butterfly_kernel<<<nBlock, nThread>>>(Y, X, k, p);
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
#undef E_BUT
#undef N_BUT
#undef E_ELE
#undef N_ELE

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
 * Montgomery's reduction will be used for modular multiplications
 *
 */
void stockham_mont(sfixn *X_d, sfixn n, sfixn k, sfixn w, sfixn p) 
{
    // initialize fourier prime structure
    fprime_t fp_h, *fpp = &fp_h;
    init_fourier_prime(fpp, p);
    setup_const(fpp);

    // initialize the primitive roots
    sfixn *W_h = new sfixn[n];
    get_mont_root_power(w, W_h, n / 2, fpp);
    sfixn *W_d;
    cudaMalloc((void **)&W_d, sizeof(sfixn) * n / 2);
    cudaMemcpy(W_d, W_h, sizeof(sfixn) * n / 2, cudaMemcpyHostToDevice);
    
    // sequence of applications 
    sfixn *Y_d;
    cudaMalloc((void **)&Y_d, sizeof(sfixn) * n);
    butterfly(Y_d, X_d, k, p);
    for (sfixn i = k - 2; i >= 0; --i) {
        stride_transpose2(X_d, Y_d, k, i);
        stride_twiddle(X_d, W_d, k, i, p, true);
        butterfly(Y_d, X_d, k, p);
    }
    cudaMemcpy(X_d, Y_d, sizeof(sfixn)*n, cudaMemcpyDeviceToDevice);

    delete [] W_h;
    cudaFree(W_d);
    cudaFree(Y_d);
}

/**
 * @X, input data array of length n = 2^k residing in the host 
 * @w, n-th primitive root of unity
 * @p, fourier prime number
 *
 * X will be filled by DFT_n(X)
 *
 */
void stockham_mont_host(sfixn *X, sfixn n, sfixn k, sfixn w, sfixn p) {
    sfixn *X_d;
    cudaMalloc((void **)&X_d, sizeof(sfixn) * n);
    cudaMemcpy(X_d, X, sizeof(sfixn) * n, cudaMemcpyHostToDevice);
    float elapsedTime;
    start_timer(0);
    ///////////////////////////////////////
    stockham_mont(X_d, n, k, w, p);
    ///////////////////////////////////////
    stop_timer(0, elapsedTime);
    printf("%2d\tmont_fft_no_transfer\t%8.3f\t", k, elapsedTime);
    cudaMemcpy(X, X_d, sizeof(sfixn) * n, cudaMemcpyDeviceToHost);
    cudaFree(X_d);

    if (DEBUG) checkCudaError("error found in stockham_mont");
}
