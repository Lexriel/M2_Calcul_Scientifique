#include <cassert>
#include <iostream>

#include "../include/list_stockham.h"
#include "../include/defines.h"
#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/cudautils.h"
#include "../include/fft_aux.h"
#include "../include/rdr_poly.h"

///////////////////////////////////////////////////////////////////////////////
// Reversion history:
//
//     File created at Mon Jul 12 EDT 2010, WP
//
//     -- list of 1d FFT implemented
//     -- 2d FFT implemented
//     -- list of 2d FFT implemented
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// When the fft size is small, say 256, it is not a good idea to run a single
// fft using GPUs. We implement the formula I_m @ DFT_n with the condition 
// m * n large enough. Currently, we assume the n-th primitive root unity w
// is used for all DFTs.  
//
// We use Stockham's FFT inside and let m = 2^j, n = 2^k. 
//
// I_m @ DFT_n = Prod_{i = 0}^{k - 1} 
//               (I_m @ DFT2 @ I_{2^{k - 1}})
//               (I_m @ D_{2, 2^{k - i - 1}} @ I_{2^i})
//               (I_m @ L_2^{2^{k - i}} @ I_{2^i})
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//   List of stride permutations:  (I_m @ L_2^{2^{k - i}} @ I_{2^i}) 
///////////////////////////////////////////////////////////////////////////////

// each thread block consists of N_THD number of threads 
// each thread block use N_THD X sizeof(sfixn) bytes shared memory
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input array of length m * n
 * @Y, output array of length m * n
 * @m,  
 * @k, n = 2^k
 * @i, index from 0 to k - 1
 *
 * Compute 
 * 
 * Y = (I_m @ L_2^{2^{k-i}} @ I_{2^i}) X 
 *
 * with the case that s = 2^i >= N_THD, multiple thread blocks move one stride. 
 *
 * The total number of thread blocks required is m * n / N_THD.
 * 
 * Each group of n / N_THD thread blocks handle a subvector of size n. 
 *
 * Requirements:
 *
 * (1) m >= 1 (2) k > i >= E_THD
 */
__global__ 
void list_stride_transpose2a_ker(sfixn *Y, const sfixn * const X, 
                                 sfixn k, sfixn i) 
{
    __shared__ sfixn block[N_THD];

    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // determine which subvector to work
    // squo = bid / (n / N_THD)
    // sbid = bid % (n / N_THD)
    sfixn squo = (bid >> (k - E_THD));
    sfixn sbid = (bid & ((1 << (k - E_THD)) - 1));
    // now sbid is the block index inside each group
    // delta = s / N_THD;
    sfixn edelta = i - E_THD;
    // iq = quo(sbid, delta) and ir = rem(sbid, delta)
    sfixn iq = (sbid >> edelta);
    sfixn ir = (sbid & ((1 << edelta) - 1));
    // iqq = quo(iq, 2) and iqr = rem(iq, 2)
    sfixn iqq = (iq >> 1);
    sfixn iqr = (iq & 1);
    
    // read in data from X
    // the input offset for this block is squo * n + iq * s + ir * N_THD
    const sfixn *din = X + (squo << k) + (iq << i) + (ir << E_THD);
    // each thread read in one element
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();

    // write data out to Y
    // the output offset for this block is 
    // squo * n + rem(iq, 2) * n / 2 + quo(iq, 2) * s + ir * N_THD
    sfixn *dout = Y + (squo << k) + (iqr << (k - 1)) 
                    + (iqq << i) + (ir << E_THD);

    dout[threadIdx.x] = block[threadIdx.x];
}

/**
 * @X, input array of length m * n
 * @Y, output array of length m * n
 * @m,
 * @k, n = 2^k
 * @i, index from 0 to k - 1
 *
 * Compute 
 * 
 * Y = (I_m @ L_2^{2^{k-i}} @ I_{2^i}) X 
 *
 * with the case that s = 2^i < N_THD, one thread block moves at least 
 * two strides. The total number of thread blocks required is m * n / N_THD.
 * Each group of n / N_THD thread blocks handle a subvector of size n. 
 *
 * Requirements:
 *
 * (1) m >= 1, (2) 0 <= i < E_THD, (3) k >= E_THD
 */
__global__ 
void list_stride_transpose2b_ker(sfixn *Y, const sfixn * const X, 
                                 sfixn k, sfixn i) 
{
    __shared__ sfixn block[N_THD];
    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // read in data from X
    // the input offset for this block is bid * N_THD
    const sfixn *din = X + (bid << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();
    
    // determine which subvector to work
    // squo = bid / (n / N_THD)
    // sbid = bid / (n / N_THD)
    sfixn squo = (bid >> (k - E_THD));
    sfixn sbid = (bid & ((1 << (k - E_THD)) - 1));
   
    // now sbid is the block index inside each group
    // the following code is to the in-block shuffle, 
    // hard to explain and check the note
    
    // offset0 = squo * n + sbid * N_THD / 2
    // offset1 = squo * n + sbid * N_THD / 2 + n / 2
    // base = Y + offset0
    sfixn *base = Y + (squo << k) + (sbid << (E_THD - 1));
    sfixn tid = threadIdx.x;
    // iq = quo(tid, s) and ir = rem(tid, s);
    sfixn iq = (tid >> i);
    sfixn ir = tid & ((1 << i) - 1);
    // f(i) = (rem(2iq, N_THD/s) + quo(2iq, N_THD/s)) * s + ir
    sfixn fi = (iq << 1) >> (E_THD - i);
    fi += ((iq << 1) & ((1 << (E_THD - i)) - 1));
    fi <<= i;
    fi += ir;
    
    // replace the following code by the branch-free code
    // if (tid < N_THD/2)
    //     dout[tid] = block[fi];
    // else
    //     dout[tid - N_THD / 2 + (1 << (k-1))] = block[fi];
    sfixn *dout = base + (tid >> (E_THD - 1)) 
                       * ((1 << (k - 1)) - (1 << (E_THD - 1))); 

    dout[tid] = block[fi];
}

/**
 * @X, device array of length m * n (input)
 * @Y, device array of length m * n (output)
 * @m,
 * @k, n = 2^k
 * @i, index from 0 to k - 2
 *
 * Compute 
 * 
 * Y = (I_m @ L_2^{2^{k-i}} @ I_{2^i}) X 
 *
 * Requirememts:
 *
 * (1) m >= 1, (2) 0 <= i <= k - 2, (3) k >= N_THD
 *
 * TESTED
 *
 */
void list_stride_transpose2_dev(sfixn *Y, const sfixn * const X, sfixn m, 
                                sfixn k, sfixn i) 
{
    if (DEBUG) assert((m >= 1) && (i >= 0) && (k >= E_THD) && (i < k - 1));

    sfixn nb = (m << (k - E_THD));
    dim3 nBlk(nb, 1, 1);
    // the maximal possible dimension is 2^15 = 32768 < 65535
    // this requires nb <= 2^30, OK for now.
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }
    if (i >= E_THD) {
        // printf("Calling transpose2a_ker k = %d, i = %d\n", k, i);
        list_stride_transpose2a_ker<<<nBlk, N_THD>>>(Y, X, k, i);
    } else {
        // printf("Calling transpose2b_ker k = %d, i = %d\n", k, i);
        list_stride_transpose2b_ker<<<nBlk, N_THD>>>(Y, X, k, i);
    }
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//               (I_m @ D_{2, 2^{k-i-1}} @ I_{2^i})                          // 
///////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/** 
 * D_{2, 2^{k - i - 1}} is a matrix of size 2^{k-i} X 2^{k-i}
 *
 * For example, let j = 0, k = 4, i = 1 and w^8 = -1
 *
 *             [ 1                   ]  0          
 *             [   1                 ]  1          
 *             [     1               ]  2          
 *             [       1             ]  3          
 * D_{2, 4} =  [         1           ]  4          
 *             [           w^2       ]  5          
 *             [              w^4    ]  6    
 *             [                 w^6 ]  7
 *
 *                  [ 1                                 ] 0  
 *                  [   1                               ] 1 
 *                  [     1                             ] 2 
 *                  [       1                           ] 3       
 *                  [         1                         ] 4
 *                  [           1                       ] 5 
 *                  [             1                     ] 6 
 * D_{2, 4} @ I_2 = [               1                   ] 7  
 *                  [                 1                 ] 8
 *                  [                   1               ] 9
 *                  [                     w^2           ] 10
 *                  [                       w^2         ] 11
 *                  [                         w^4       ] 12
 *                  [                           w^4     ] 13
 *                  [                             w^6   ] 14
 *                  [                               w^6 ] 15
 * 
 * If N_THD = 4, then 2 blocks are needed. Block 0 handles [1, 1, w^2, w^2]
 * and block 1 handle [w^4, w^4, w^6, w^6].
 *
 * For each group, only half of the elements needs to be modified, 
 * and the index range is from n / 2 to n - 1. 
 *
 * Hence the number of thread blocks for each group is n / (2 * N_THD).
 * The total number of thread blocks is m * n / (2 * N_THD).
 *
 **/

/**
 * @X, input/output array of length m * n
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @m,
 * @k, n = 2^k
 * @i, index from 0 to k - 1
 *
 * Multiple thread blocks (>1) handle a stride, (s = 2^i is big)
 *
 * Requirements: m >= 1, k > i > E_THD
 *
 */
__global__ void 
list_stride_twiddle2a_ker(sfixn *X, const sfixn * const W, sfixn k, sfixn i, 
                          sfixn p, double pinv) 
{
    // block index
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // determine which subvector to work
    // squo = bid / (n / (2 * N_THD))
    // sbid = bid % (n / (2 * N_THD))
    // sbid is the block index inside each group
    sfixn squo = (bid >> (k - 1 - E_THD));
    sfixn sbid = (bid & ((1 << (k - 1 - E_THD)) - 1));

    // all threads in a thread block are using the same power!
    // This power is w^(s*e), with e = quo(sbid, s/N_THD)
    sfixn w = W[(sbid >> (i - E_THD)) << i];
    // starting position for the block, the first n / 2 elements unchanged
    sfixn *base = X + (squo << k) + ((sfixn)1 << (k - 1)) + (sbid << E_THD);
    base[threadIdx.x] = mul_mod(w, base[threadIdx.x], p, pinv);
}

/**
 * @X, input/output array of length m * n
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @m, 
 * @k, n = 2^k
 * @i, index from 0 to k - 1
 *
 * A thread block handle multiple strides (s = 2^i is small)
 *
 * Requirements:
 *
 * (1) m >= 1 (2) k > E_THD >= i >= 0
 *
 */
__global__ void 
list_stride_twiddle2b_ker(sfixn *X, const sfixn * const W, sfixn k, sfixn i, 
                          sfixn p, double pinv)
{
    // block index
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // determine which subvector to work
    // squo = bid / (n / (2 * N_THD))
    // sbid = bid % (n / (2 * N_THD))
    // sbid is the block index inside each group
    sfixn squo = (bid >> (k - 1 - E_THD));
    sfixn sbid = (bid & ((1 << (k - 1 - E_THD)) - 1));
 
    // starting position for this block
    // the first n / 2 elements will be unchanged.
    sfixn *base = X + (squo << k) + ((sfixn)1 << (k - 1)) + (sbid << E_THD);
    // the starting root for the thread block is w^(e*s) 
    // with e = sbid * (N_THD / s). Thus e*s = sbid * N_THD. 
    sfixn tid = threadIdx.x;
    // the power for the thread e * s + s * quo(tid, s) 
    sfixn iq = (sbid << E_THD) + ((tid >> i) << i);
    base[tid] = mul_mod(W[iq], base[tid], p, pinv);
}

/**
 * @X, input/output array of length m * n = 2^{k+j}
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @m, 
 * @k, n = 2^k
 * @i, index from 0 to k - 1
 *
 * Requirements: 0 <= i <= k - 2, n > N_THD, m >= 1
 *
 * TESTED
 *
 */
void list_stride_twiddle2_dev(sfixn *X, const sfixn * const W, sfixn m, 
                              sfixn k, sfixn i, sfixn p)
{
    if (DEBUG) assert((i >= 0) && (i < k - 1));
    if (DEBUG) assert((m >= 1) && (k > E_THD));

    // number of blocks is m * (n / 2) / N_THD
    sfixn nb = (m << ( k - 1 - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    double pinv = 1 / (double)p;
    if (i > E_THD) {
        // printf("Calling twiddle2a_ker m = %d, k = %d, i = %d\n", m, k, i);
        // printf("nBlk = %d, nThd = %d\n", nBlk.x, N_THD);
        list_stride_twiddle2a_ker<<<nBlk, N_THD>>>(X, W, k, i, p, pinv);
    } else {
        // printf("Calling twiddle2b_ker m = %d, k = %d, i = %d\n", m, k, i);
        // printf("nBlk = %d, nThd = %d\n", nBlk.x, N_THD);
        list_stride_twiddle2b_ker<<<nBlk, N_THD>>>(X, W, k, i, p, pinv);
    }

    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//      List of butterflies (I_m @ DFT2 @ I_{n/2})                            //
////////////////////////////////////////////////////////////////////////////////
//
// a butterfly operation is defined as
//
//   x0  y0
//     \/
//     /\
//   xs  ys
//
//  with y0 = x0 + xs and ys = x0 - xs. 
//  In total, 2 + 2 elements are involved for each butterfly
////////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, device array of length m * n (input)
 * @Y, device array of length m * n (output)
 * @m,
 * @k, n = 2^k
 * @p, prime number
 *
 * Implements I_m @ DFT2 @ I_{n/2}
 *
 * Requires m >= 1, k > E_THD
 *
 */
__global__ void 
list_butterfly_ker(sfixn *Y, const sfixn * const X, sfixn k, sfixn p)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // determine which subvector to work
    // squo = bid / ((n / 2) / N_THD)
    // sbid = bid % ((n / 2) / N_THD)
    // sbid is the block index inside each group
    sfixn squo = (bid >> (k - 1 - E_THD));
    sfixn sbid = (bid & ((1 << (k - 1 - E_THD)) - 1));
    sfixn *B = Y + (squo << k) + (sbid << E_THD); 
    const sfixn *A = X + (squo << k) + (sbid << E_THD);

    sfixn tid = threadIdx.x;
    sfixn halfn = ((sfixn )1 << (k - 1));
    B[tid] = add_mod(A[tid], A[tid + halfn], p);
    B[tid + halfn] = sub_mod(A[tid], A[tid + halfn], p);
}

/**
 * @X, device array of length m * n (input)
 * @Y, device array of length m * n (output)
 * @m, 
 * @k, n = 2^k
 * @p, prime number
 *
 * I_m @ DFT2 @ I_{n/2}
 *
 * Requirements:  m >= 1, k > E_THD
 *
 * TESTED
 *
 */
void list_butterfly_dev(sfixn *Y, const sfixn *X, sfixn m, sfixn k, sfixn p)
{
    if (DEBUG) assert(m >= 1 && k > E_THD);
    sfixn nb = (m << (k - E_THD - 1));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    list_butterfly_ker<<<nBlk, N_THD>>>(Y, X, k, p);
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
////////////////////////////////////////////////////////////////////////////////
// List of FFTs: I_m @ DFT_n, Main Program                                    //
////////////////////////////////////////////////////////////////////////////////
/**
 * @X, input / output device array of length m * n 
 * @m, 
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
void list_stockham_dev(sfixn *X, sfixn m, sfixn k, const sfixn *W, sfixn p)
{
    // TODO: check the size of k
    if (DEBUG) assert((m >= 1));

    sfixn *Y;
    cudaMalloc((void**)&Y, sizeof(sfixn) * (m << k));

    // sequence of applications 
    // i = k - 1 is trival for the other operations
    list_butterfly_dev(Y, X, m, k, p); 
    
    for (sfixn i = k - 2; i >= 0; --i) {
        list_stride_transpose2_dev(X, Y, m, k, i);
        list_stride_twiddle2_dev(X, W, m, k, i, p);
        list_butterfly_dev(Y, X, m, k, p);
    }
    cudaMemcpy(X, Y, sizeof(sfixn) * (m << k), cudaMemcpyDeviceToDevice);
    cudaFree(Y);
    if (DEBUG) checkCudaError("error found in list_stockham_dev");
}

void list_stockham_dev(sfixn *X, sfixn m, sfixn k, sfixn w, sfixn p)
{
    // initialize the primitive roots
    sfixn *W;
    cudaMalloc((void**)&W, sizeof(sfixn) << (k - 1));
    get_powers_binary(k - 1, W, w, p);
    
    list_stockham_dev(X, m, k, W, p);
    cudaFree(W);
    if (DEBUG) checkCudaError("error found in list_stockham_dev");
}

/**
 * @X, input / output host array of length m * n 
 * @m,
 * @k, n = 2^k
 * @w, n-th primitive root of unity
 * @p, fourier prime number
 *
 * X will be filled by I_m @ DFT_n(X, w)
 */
void list_stockham_host(sfixn *X, sfixn m, sfixn k, sfixn w, sfixn p) {
    sfixn *X_d;
    cudaMalloc((void**)&X_d, sizeof(sfixn) * (m << k));
    cudaMemcpy(X_d, X, sizeof(sfixn) * (m << k), cudaMemcpyHostToDevice);
    ///////////////////////////////////////
    list_stockham_dev(X_d, m, k, w, p);
    ///////////////////////////////////////
    cudaMemcpy(X, X_d, sizeof(sfixn) * (m << k), cudaMemcpyDeviceToHost);
    cudaFree(X_d);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// The goal is to implement DFT_n @ I_m. We use the Stockham FFT, that is  
//
// DFT_n @ I_m = Prod_{i = 0}^{k - 1} 
//               DFT2 @ I_{2^{k - 1}} @ I_{2^j}
//               D_{2, 2^{k - i - 1}} @ I_{2^i} @ I_{2^j}
//               L_2^{2^{k - i}} @ I_{2^i} @ I_{2^j}
//
//             = DFT2 @ I_{2^{k - 1 + j}}               (1)
//               D_{2, 2^{k - i - 1}} @ I_{2^{i+^j}}    (2)  
//               L_2^{2^{k - i}} @ I_{2^{i+j}}          (3)
// 
// Note that (1) has been impmenented by stockham.cu, however both (2) and (3)
// invalidate its assumptions. We now relax them.
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  Implementation of L_2^{n} @ I_s, extended version, that is, it works
//  for any input such that n >= 4 and s >= 1 and n * s >= N_THD.
////////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input array of length n * s = 2^{u + v}
 * @Y, output array of length n * s = 2^{u + v}
 * @u, n = 2^u
 * @v, s = 2^v
 *
 * Compute the general stride transposition
 * 
 * Y = (L_2^n @ I_s) X 
 *
 * Multiple thread blocks (>= 1) move a stride of s elements 
 *
 * Requirements:  u >= 2 and v >= E_THD
 *
 * If u = 1, then do nothing and the transiposition is trivial. 
 *
 * TESTED
 */
__global__ 
void ext_stride_transpose2a_ker(sfixn *Y, const sfixn *X, sfixn u, sfixn v)
{
    __shared__ sfixn block[N_THD];
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // delta = s / N_THD; 
    // the number of thread block needed for each stride
    sfixn edelta = v - E_THD;
    // iq = quo(bid, delta) and ir = rem(bid, delta)
    // iq tells which stride the block is working on
    // ir tells which portion of the stride the block is working on
    sfixn iq = bid >> edelta;
    sfixn ir = bid & ((1 << edelta) - 1);

    // iqq = quo(iq, 2) and iqr = rem(iq, 2)
    sfixn iqq = (iq >> 1);
    sfixn iqr = (iq & 1);

    // read in data from X
    // the input offset for this block is iq * s + ir * N_THD
    const sfixn *din = X + (iq << v) + (ir << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();

    // write data out to Y
    //
    // if iqr = 0 (even), write to Y + iqq * s + ir * N_THD
    // if iqr = 1 (odd), write to Y + (n / 2) * s + iqq * s + ir * N_THD
    // that is, iqr * (n / 2) * s + iqq * s + ir * N_THD
    sfixn *dout = Y + (iqr << (u + v - 1)) +  (iqq << v) + (ir << E_THD);
    dout[threadIdx.x] = block[threadIdx.x];
    __syncthreads();
}

/**
 * @X, input array of length n * s = 2^{u + v}
 * @Y, output array of length n * s = 2^{u + v}
 * @u, n = 2^u
 * @v, s = 2^v
 *
 * Compute the general stride transposition
 * 
 * Y = (L_2^n @ I_s) X 
 *
 * A thread block moves multiple strides
 *
 * Requirements:  u >= 2 and v < E_THD
 *
 * If u = 1, then do nothing and the transiposition is trivial. 
 *
 * TESTED
 */
__global__ 
void ext_stride_transpose2b_ker(sfixn *Y, const sfixn *X, sfixn u, sfixn v) {

    __shared__ sfixn block[N_THD];
    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // read in data from X
    // the input offset for this block is bid * N_THD
    const sfixn *din = X + (bid << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();
    
    // offset0 = bid * N_THD / 2
    // offset1 = bid * N_THD / 2 + (n / 2) * s
    // base = Y + offset0
    sfixn *base = Y + (bid << (E_THD - 1));
    sfixn tid = threadIdx.x;
    // iq = quo(tid, s) and ir = rem(tid, s);
    sfixn iq = (tid >> v);
    sfixn ir = tid & ((1 << v) - 1);
    // the following code is to the in-block shuffle 
    // f(i) = (rem(2iq, N_THD/s) + quo(2iq, N_THD/s)) * s + ir
    sfixn fi = (iq << 1) >> (E_THD - v);
    fi += ((iq << 1) & ((1 << (E_THD - v)) - 1));
    fi <<= v;
    fi += ir;
    
    // replace the following code by the branch-free code
    // if (tid < N_THD/2)
    //     dout[tid] = block[fi];
    // else
    //     dout[tid - N_THD / 2 + (1 << (u + v - 1))] = block[fi];
    sfixn *dout = base + (tid >> (E_THD - 1)) 
                       * ((1 << (u + v - 1)) - (1 << (E_THD - 1))); 

    dout[tid] = block[fi];
}

void ext_stride_transpose2_dev(sfixn *Y, const sfixn *X, sfixn u, sfixn v) 
{
    if (DEBUG) assert((u >= 2) && (v >= 0) && (u + v >=  E_THD));

    sfixn nb = ((sfixn)1 << (u + v - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    if (v >= E_THD) {
        // printf("Calling transpose2a_ker with u = %d, v = %d\n", u, v);
        ext_stride_transpose2a_ker<<<nBlk, N_THD>>>(Y, X, u, v);
    } else {
        // printf("Calling transpose2b_ker with u = %d, v = %d\n", u, v);
        ext_stride_transpose2b_ker<<<nBlk, N_THD>>>(Y, X, u, v);
    }

    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  Implementation of D_{2, m} @ I_{s}, extended version, that is, it works
//  for any input such that m >= 2 and s >= 1 and m * s >= N_THD. Note that
//  the FFT size n is still in use and we also require m <= n / 2. 
//
//  For example, let n = 16 be the FFT size. We have 
//
//             [ 1         ]  0          
//             [   1       ]  1          
// D_{2, 2} =  [     1     ]  2          
//             [       w^4 ]  3          
//
//                  [ 1                                  ]  0          
//                  [   1                                ]  1          
//                  [     1                              ]  2          
//                  [       1                            ]  3          
// D_{2, 2} @ I_4 = [         1                          ]  4          
//                  [           1                        ]  5          
//                  [             1                      ]  6    
//                  [               1                    ]  7
//                  [                 1                  ]  8
//                  [                   1                ]  9
//                  [                     1              ]  10
//                  [                       1            ]  11
//                  [                         w^4        ]  12
//                  [                           w^4      ]  13
//                  [                             w^4    ]  14
//                  [                                w^4 ]  15
//
//                  [ 1                                 ] 0  
//                  [   1                               ] 1 
//                  [     1                             ] 2 
//                  [       1                           ] 3       
//                  [         1                         ] 4
//                  [           1                       ] 5 
//                  [             1                     ] 6 
// D_{2, 4} @ I_2 = [               1                   ] 7  
//                  [                 1                 ] 8
//                  [                   1               ] 9
//                  [                     w^2           ] 10
//                  [                       w^2         ] 11
//                  [                         w^4       ] 12
//                  [                           w^4     ] 13
//                  [                             w^6   ] 14
//                  [                               w^6 ] 15
//
// Warning:
//
// The purpuse to handle the case 2 * m * s >= n. If this does not hold,
// be aware of the meaning of the computation result. 
//
// For example, let n = 16, s = 1 and m = 4, (2 * m * s = 8 < 16)
// Its matrix representation is 
//
// [ 1                  ]
// [   1                ]
// [     1              ]
// [       1            ]
// [         1          ]
// [           w        ]
// [             w^2    ]
// [                w^3 ]
// 
// where w is a 16-primitive root of unity.
//
// The usual D_{2, 4} @ I_1 is represented by
//
// [ 1                  ]
// [   1                ]
// [     1              ]
// [       1            ]
// [         1          ]
// [           x        ]
// [             x^2    ]
// [                x^3 ]
// 
// with x being a 8-th primitive root of unity.
//
// For the above reason, we require the condition 2 * m * s >= n, for safety.
//
////////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input/output array of length 2 * m * s = 2^{u + v + 1}
 * @n, FFT size n = 2^e 
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @u, m = 2^u
 * @v, s = 2^v
 * 
 * Implements D_{2, m} @ I_{s} with FFT size n
 *
 * Multiple thread blocks handle a stride, (s = 2^v is big)
 *
 * Requirements: (1) v >= E_THD, (2) 2 <= m < n
 *
 * TESTED
 */
__global__ void 
ext_stride_twiddle2a_ker(sfixn *X, const sfixn * const W, sfixn e, sfixn u,
                         sfixn v, sfixn p, double pinv)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    // all threads in a thread block are using the same power!
    // the base power for the grid is exp = (n / 2) / m
    // The power for the block is w^(exp * e), with e = quo(bid, s / N_THD)
    sfixn w = W[(bid >> (v - E_THD)) << (e - 1 - u)];
    // starting position for the block, the first m * s elements unchanged
    sfixn *base = X + ((sfixn)1 << (u + v)) + (bid << E_THD);
    base[threadIdx.x] = mul_mod(w, base[threadIdx.x], p, pinv);
}

/**
 * @X, input/output array of length 2 * m * s = 2^{u + v + 1}
 * @n, FFT size n = 2^e 
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @u, m = 2^u
 * @v, s = 2^v
 * 
 * Implements: D_{2, m} @ I_{s} with FFT size n
 *
 * A thread block handles multiple strides, (s = 2^v is small)
 *
 * Requirements: (1) v < E_THD, (2) 2 <= m < n
 *
 * TESTED
 */
__global__ void 
ext_stride_twiddle2b_ker(sfixn *X, const sfixn * const W, sfixn e, sfixn u,
                         sfixn v, sfixn p, double pinv) 
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // the first m * s elements will be unchanged.
    sfixn *base = X + ((sfixn)1 << (u + v)) + (bid << E_THD);
    // threads in a thread block use different powers.
    // the base power for the grid is egrid = (n / 2) / m.
    // the base power for the block is eblock = bid * (N_THD / s) * egrid.
    sfixn eblock = bid << (E_THD - v + e - 1 - u);
    sfixn tid = threadIdx.x;
    // the power for the thread eblock + s * quo(tid, s) 
    sfixn iq = eblock + ((tid >> v) << v);
    base[tid] = mul_mod(W[iq], base[tid], p, pinv);
}

/**
 * @X, input/output array of length 2 * m * s = 2^{u + v + 1}
 * @n, FFT size n = 2^e 
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @u, m = 2^u
 * @v, s = 2^v
 * 
 * Implements: D_{2, m} @ I_{s} with FFT size n
 *
 * Requirements: 2 <= m < n, s >= 1, 2 * m * s >= n and m * s >= N_THD
 *
 * TESTED
 */
void ext_stride_twiddle2_dev(sfixn *X, const sfixn * const W, sfixn e, sfixn u, 
                             sfixn v, sfixn p)
{
    if (DEBUG) assert((v >= 0) && (u > 0) && (u < e));
    if (DEBUG) assert((u + v >= E_THD) && (1 + u + v >= e));

    // number of blocks is m * s / N_THD
    sfixn nb = (1 << (u + v - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }
    
    double pinv = 1 / (double)p;
    if (v >= E_THD) {
        // printf("Calling twiddle2a_ker e = %d, u = %d, v = %d\n", e, u, v);
        ext_stride_twiddle2a_ker<<<nBlk, N_THD>>>(X, W, e, u, v, p, pinv);
    } else {
        // printf("Calling twiddle2b_ker e = %d, u = %d, v = %d\n", e, u, v);
        ext_stride_twiddle2b_ker<<<nBlk, N_THD>>>(X, W, e, u, v, p, pinv);
    }

    cudaThreadSynchronize();
}

/**
 * @X, device array of length 2 * s = 2^{v + 1)
 * @Y, device array of length 2 * s = 2^(v + 1) (output)
 * @v, s = 2^v
 * @p, prime number
 *
 * Implements: DFT2 @ I_{s}
 *
 * Requires: s >= N_THD
 *
 */
__global__
void ext_butterfly_ker(sfixn *Y, const sfixn * const X, sfixn v, sfixn p)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn *B = Y + (bid << E_THD); 
    const sfixn *A = X + (bid << E_THD);
    sfixn s = ((sfixn)1 << v);
    B[threadIdx.x] = add_mod(A[threadIdx.x], A[threadIdx.x + s], p);
    B[threadIdx.x + s] = sub_mod(A[threadIdx.x], A[threadIdx.x + s], p);
}

/**
 * @X, device array of length 2 * s = 2^{v + 1)
 * @Y, device array of length 2 * s = 2^(v + 1) (output)
 * @v, s = 2^v
 * @p, prime number
 *
 * Implements: DFT2 @ I_{s}
 *
 * Requires: s >= N_THD
 * 
 * TESTED
 */
void ext_butterfly_dev(sfixn *Y, const sfixn * const X, sfixn v, sfixn p)
{
    if (DEBUG) assert(v >= E_THD);
    sfixn nb = ((sfixn)1 << (v - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    ext_butterfly_ker<<<nBlk, N_THD>>>(Y, X, v, p);
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD

////////////////////////////////////////////////////////////////////////////////
// The main program of
//
// DFT_n @ I_m = Prod_{i = 0}^{k - 1} 
//               DFT2 @ I_{2^{k - 1 + j}}               (1)
//               D_{2, 2^{k - i - 1}} @ I_{2^{i+j}}     (2)  
//               L_2^{2^{k - i}} @ I_{2^{i+j}}          (3)
////////////////////////////////////////////////////////////////////////////////

/**
 * @X, input / output device array of length n * m 
 * @k, n = 2^k
 * @j, m = 2^j
 * @w, n-th primitive root of unity
 * @W, [1, w, w^2, ..., w^{n/2-1}]
 * @p, fourier prime number
 *
 * X will be filled by DFT_n @ I_m(X, w)
 *
 */
void ext_stockham_dev(sfixn *X, sfixn k, sfixn j, const sfixn *W, sfixn p) {
    sfixn *Y;
    cudaMalloc((void**)&Y, sizeof(sfixn) << (k + j));

    // sequence of applications 
    // i = k - 1 is trival for the other operations
    ext_butterfly_dev(Y, X, k + j - 1, p); 
    for (sfixn i = k - 2; i >= 0; --i) {
        // u = k - i, v = i + j
        ext_stride_transpose2_dev(X, Y, k - i, i + j);
        // u = k - i - 1, v = i + j
        ext_stride_twiddle2_dev(X, W, k, k - i - 1, i + j, p);
        ext_butterfly_dev(Y, X, k + j - 1, p);
    }
    cudaMemcpy(X, Y, sizeof(sfixn) << (k + j), cudaMemcpyDeviceToDevice);
    cudaFree(Y);

    if (DEBUG) checkCudaError("error found in ext_stockham_dev");
}

/* Without precomputed powers of the root */
void ext_stockham_dev(sfixn *X, sfixn k, sfixn j, sfixn w, sfixn p)
{
    // initialize the primitive roots
    sfixn *W;
    cudaMalloc((void**)&W, sizeof(sfixn) << (k - 1));
    get_powers_binary(k - 1, W, w, p);
    ext_stockham_dev(X, k, j, W, p);
    cudaFree(W);

    if (DEBUG) checkCudaError("error found in ext_stockham_dev");
}

///////////////////////////////////////////////////////////////////////////////
// 2-d FFT, the row-column algorithm is
//
// DFT_{m, n} (X) = (DFT_m @ I_n) (I_m @ DFT_n) (X)
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @X, input / output device array of length m * n 
 * @em, m = 2^em (rows)
 * @en, n = 2^en (columns)
 * @wn, n-th primitive root of unity
 * @Wn, [1, wn, wn^2, ..., wn^{n/2-1}]
 * @wm, m-th primitive root of unity
 * @Wm, [1, wm, wm^2, ..., wm^{m/2-1}]
 * @p, fourier prime number
 *
 * Compute X = DFT_{m, n}(X) = (DFT_m @ I_n) (I_m @ DFT_n)X
 *
 */
void bivariate_stockham_dev(sfixn *X, sfixn em, const sfixn *Wm, sfixn en, 
                            const sfixn *Wn, sfixn p)
{
    sfixn *Y;
    cudaMalloc((void**)&Y, sizeof(sfixn) << (em + en));
    
    // list_stockham_dev(X, 1 << em, en, Wn, p);
    sfixn m = ((sfixn)1 << em);
    
    list_butterfly_dev(Y, X, m, en, p); 
    for (sfixn i = en - 2; i >= 0; --i) {
        list_stride_transpose2_dev(X, Y, m, en, i);
        list_stride_twiddle2_dev(X, Wn, m, en, i, p);
        list_butterfly_dev(Y, X, m, en, p);
    }

    // ext_stockham_dev(X, en, em, Wm, p);
    ext_butterfly_dev(X, Y, en + em - 1, p); 
    for (sfixn i = en - 2; i >= 0; --i) {
        // u = en - i, v = em + i 
        ext_stride_transpose2_dev(Y, X, en - i, em + i);
        // u = en - i - 1, v = em + i
        ext_stride_twiddle2_dev(Y, Wm, en, en - i - 1, em + i, p);
        ext_butterfly_dev(X, Y, en + em - 1, p);
    }
    cudaFree(Y);
    if (DEBUG) checkCudaError("bivariate_stockham_dev");
}

/**
 * @X, input / output host array of length n * m 
 * @en, n = 2^en
 * @em, m = 2^em
 * @wn, n-th primitive root of unity
 * @wm, m-th primitive root of unity
 * @p, fourier prime number
 *
 * Compute X = DFT_{m, n}(X) = (DFT_m @ I_n) (I_m @ DFT_n) X
 *
 * It computes if X is in the rdr-representation
 * 
 * F(1, 1)         F(wn, 1)         ...   F(wn^(n-1), 1)
 * F(1, wm)        F(wn, wm)        ...   F(wn^(n-1), wm)
 * ...
 * F(1, wm^(m-1))  F(wn, wm^(m-1))  ...   F(wn^(n-1), wm^(m-1))
 *
 */
void bivariate_stockham_host(sfixn *X, sfixn em, sfixn wm, 
    sfixn en, sfixn wn, sfixn p) 
{
    sfixn *X_d;
    cudaMalloc((void**)&X_d, sizeof(sfixn)<<(em + en));
    cudaMemcpy(X_d, X, sizeof(sfixn)<<(em + en), cudaMemcpyHostToDevice);
    
    sfixn *Wm, *Wn;
    cudaMalloc((void**)&Wm, sizeof(sfixn) << (em - 1));
    cudaMalloc((void**)&Wn, sizeof(sfixn) << (en - 1));
    get_powers_binary(em - 1, Wm, wm, p);
    get_powers_binary(en - 1, Wn, wn, p);

    bivariate_stockham_dev(X_d, em, Wm, en, Wn, p);
    cudaMemcpy(X, X_d, sizeof(sfixn)<<(em + en), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Wm);
    cudaFree(Wn);

    if (DEBUG) checkCudaError("bivariate_stockham_host");
}


/**
 * Assume that the inverse roots have been precomputed.
 */
void inverse_bivariate_stockham_dev(sfixn *X, sfixn em, const sfixn *invWm, 
    sfixn en, const sfixn *invWn, sfixn p) 
{
    sfixn m = sfixn(1) << em;
    sfixn n = sfixn(1) << en;

    sfixn minv = inv_mod(m, p);
    sfixn ninv = inv_mod(n, p);
    sfixn mninv = mul_mod(minv, ninv, p);

    bivariate_stockham_dev(X, em, invWm, en, invWn, p);
    scale_vector_dev(mninv, m * n, X, p);
}

/**
 * Since (DFT_m^(-1) @ DFT_n^(-1)) (DFT_m @ DFT_n) = I_m @ I_n, we have
 *
 * (DFT_m @ DFT_n)^(-1) = DFT_m^(-1) @ DFT_n^(-1)
 */
void inverse_bivariate_stockham_host(sfixn *X, sfixn em, sfixn wm, 
    sfixn en, sfixn wn, sfixn p) 
{
    sfixn m = sfixn(1) << em;
    sfixn n = sfixn(1) << en;
    sfixn wminv = inv_mod(wm, p);
    sfixn wninv = inv_mod(wn, p);
    sfixn minv = inv_mod(m, p);
    sfixn ninv = inv_mod(n, p);
    sfixn mninv = mul_mod(minv, ninv, p);

    sfixn *X_d;
    cudaMalloc((void**)&X_d, sizeof(sfixn)<<(em + en));
    cudaMemcpy(X_d, X, sizeof(sfixn)<<(em + en), cudaMemcpyHostToDevice);
    
    sfixn *Wm, *Wn;
    cudaMalloc((void**)&Wm, sizeof(sfixn) << (em - 1));
    cudaMalloc((void**)&Wn, sizeof(sfixn) << (en - 1));
    get_powers_binary(em - 1, Wm, wminv, p);
    get_powers_binary(en - 1, Wn, wninv, p);

    bivariate_stockham_dev(X_d, em, Wm, en, Wn, p);
    scale_vector_dev(mninv, m * n, X_d, p);

    cudaMemcpy(X, X_d, sizeof(sfixn)<<(em + en), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Wm);
    cudaFree(Wn);

    if (DEBUG) checkCudaError("inverse_bivariate_stockham_host");
}

/**
 * Multiply two bivariate polynomials of size POT, in place version
 *
 * @m : FFT size in y (rows)
 * @n : FFT size in x (columns)
 * @em : m = 2^em
 * @en : n = 2^en
 * @F : coefficient vector of F, padded to size n, input & output
 * @G : coefficient vector of G, padded to size n, input
 * @p : prime number
 *
 * F <-- DFT_{m, n}^{-1}(DFT_{m, n}(F) * DFT_{m, n}(G))
 *
 **/
void bi_stockham_poly_mul_dev(sfixn m, sfixn em, sfixn n, sfixn en, 
                              sfixn *F, sfixn *G, sfixn p) 
{
    sfixn wm = primitive_root(em, p);
    sfixn wn = primitive_root(en, p);
    sfixn wninv = inv_mod(wn, p);
    sfixn wminv = inv_mod(wm, p);
    sfixn minv = inv_mod(m, p);
    sfixn ninv = inv_mod(n, p);
    sfixn mninv = mul_mod(minv, ninv, p);

    sfixn *Wm, *Wn;
    cudaMalloc((void**)&Wm, sizeof(sfixn) << (em - 1));
    cudaMalloc((void**)&Wn, sizeof(sfixn) << (en - 1));
    get_powers_binary(em - 1, Wm, wm, p);
    get_powers_binary(en - 1, Wn, wn, p);
 
    bivariate_stockham_dev(F, em, Wm, en, Wn, p); 

    bivariate_stockham_dev(G, em, Wm, en, Wn, p); 
    pointwise_mul_dev(n * m, em + en, F, G, p);

    get_powers_binary(em - 1, Wm, wminv, p);
    get_powers_binary(en - 1, Wn, wninv, p);
    bivariate_stockham_dev(F, em, Wm, en, Wn, p);
    scale_vector_dev(mninv, m * n, F, p);
    
    cudaFree(Wm);
    cudaFree(Wn);

    if (DEBUG) checkCudaError("bi_stockham_poly_mul_dev");
}

/**
 * Multiply two balanced bivariate polynomials
 */
rdr_poly* 
bi_stockham_poly_mul_host(const rdr_poly &F, const rdr_poly &G, sfixn p)
{
    sfixn lx = F.ns[0] + G.ns[0] - 1;      
    sfixn ly = F.ns[1] + G.ns[1] - 1;      
    sfixn ex = ceiling_log2(lx);
    sfixn ey = ceiling_log2(ly);
    sfixn nx = (sfixn)1 << ex;
    sfixn ny = (sfixn)1 << ey;
    sfixn szh = ((sfixn)1 << (ex + ey));

    sfixn *F_d;
    cudaMalloc((void**)&F_d, sizeof(sfixn)*F.sz);
    cudaMemcpy(F_d, F.coeffs, sizeof(sfixn)*F.sz, cudaMemcpyHostToDevice);
    
    sfixn *F2_d;
    cudaMalloc((void**)&F2_d, sizeof(sfixn)*szh);
    expand_to_fft2_dev(ex, ey, F2_d, F.ns[0], F.ns[1], F_d);
    cudaFree(F_d);

    sfixn *G_d;
    cudaMalloc((void**)&G_d, sizeof(sfixn)*G.sz);
    cudaMemcpy(G_d, G.coeffs, sizeof(sfixn)*G.sz, cudaMemcpyHostToDevice);

    sfixn *G2_d;
    cudaMalloc((void**)&G2_d, sizeof(sfixn)*szh);
    expand_to_fft2_dev(ex, ey, G2_d, G.ns[0], G.ns[1], G_d);
    cudaFree(G_d);

    bi_stockham_poly_mul_dev(ny, ey, nx, ex, F2_d, G2_d, p);

    // use G2_d to store the compacted result
    // construct the result directly from G2_d
    extract_from_fft2_dev(lx, ly, G2_d, ex, F2_d);
    if (DEBUG) checkCudaError("bi_stockham_poly_mul_host");

    rdr_poly *H = new rdr_poly(lx, ly, G2_d, false);
    cudaFree(F2_d);
    cudaFree(G2_d);
    return H;
}

////////////////////////////////////////////////////////////////////////////////
//  Most general formulas to realize a list of bivariate FFTs
//
//  I_q @ DFT_{m, n} = (I_q @ (DFT_m @ I_n)) (I_q @ (I_m @ DFT_n))
//                   = (I_q @ ( Prod_{i=0}^{u - 1} 
//                              (DFT_2 @ I_{m / 2})
//                              (D_{2, 2^{u - i - 1}} @ I_{2^i})
//                              (L_2^{2^{u - i} @ I_{2^i}}) )
//                          @ I_n)
//                     (I_q @ I_m @ (Prod_{i = 0}^{k - 1} 
//                                    (DFT_2 @ I_{n / 2})
//                                    (D_{2, 2^{k - i - 1}} @ I_{2^i})
//                                    (L_2^{2^{k - i}} @ I_{2^i})))
//
//  where m = 2^u and n = 2^k are powers of two, but q usually not.
//
//  What are to be implemented,
//
// (1) I_q @ DFT_2 @ I_{m / 2} @ I_n
// (2) I_q @ D_{2, 2^{u - i - 1}} @ I_{2^i} @ I_n
// (3) I_q @ L_2^{2^{u - i}} @ I_{2^i} @ I_n
//
// with m = 2^u; 
//
// (4) I_q @ I_m @ DFT_2 @ I_{n / 2}               use (3)   
// (5) I_q @ I_m @ D_{2, 2^{k - i - 1}} @ I_{2^i}  use (2)
// (6) I_q @ I_m @ L_2^{2^{k - i}} @ I_{2^i}       use (1)
//
// with n = 2^k.
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  Implementation of I_q @ L_2^m @ I_s
////////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input array of length q * m * s = q * 2^{u + v}
 * @Y, output array of length q * m * s = q * 2^{u + v}
 * @u, m = 2^u
 * @v, s = 2^v
 *
 * Compute the general stride transposition
 * 
 * Y = (I_q @ L_2^m @ I_s) X 
 *
 * Multiple thread blocks (>= 1) move a stride of s elements 
 *
 * Requirements:  q >= 1, u >= 2 and v >= E_THD
 *
 * If u = 1, then do nothing and the transiposition is trivial. 
 */
__global__ void q_ext_stride_transpose2a_ker(sfixn *Y, const sfixn *X, 
                                             sfixn u, sfixn v)
{
    __shared__ sfixn block[N_THD];
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    
    // read in data from X
    // the input offset for this block is bid * N_THD
    const sfixn *din = X + (bid << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();

    // The number of thread blocks for one subvector is snb = m * s / N_THD
    // squo = bid / snb
    // sbid = bid % snb
    sfixn snb = ((sfixn)1 << ( u + v - E_THD));
    sfixn squo = bid >> (u + v - E_THD);
    sfixn sbid = bid & (snb - 1);

    // delta = s / N_THD; 
    // the number of thread block needed for each stride
    sfixn edelta = v - E_THD;
    // iq = quo(sbid, delta) and ir = rem(sbid, delta)
    // iq tells which stride the block is working on
    // ir tells which portion of the stride the block is working on
    sfixn iq = sbid >> edelta;
    sfixn ir = sbid & ((1 << edelta) - 1);

    // iqq = quo(iq, 2) and iqr = rem(iq, 2)
    sfixn iqq = (iq >> 1);
    sfixn iqr = (iq & 1);

    // write data out to Y
    //
    // if iqr = 0 (even), write to 
    //    Y + squo * m * s + iqq * s + ir * N_THD
    // if iqr = 1 (odd), write to 
    //    Y + squo * m * s + (n / 2) * s + iqq * s + ir * N_THD
    //
    // that is, squo * m * s + iqr * (n / 2) * s + iqq * s + ir * N_THD
    sfixn *dout = Y + (squo << (u + v)) 
                    + (iqr << (u + v - 1)) 
                    + (iqq << v) 
                    + (ir << E_THD);

    dout[threadIdx.x] = block[threadIdx.x];
    __syncthreads();
}

/**
 * @X, input array of length q * m * s = q * 2^{u + v}
 * @Y, output array of length q * m * s = q * 2^{u + v}
 * @u, m = 2^u
 * @v, s = 2^v
 *
 * Compute the general stride transposition
 * 
 * Y = (I_q @ L_2^m @ I_s) X 
 *
 * A thread block moves multiple strides
 *
 * Requirements:  q >= 1, u >= 2, 0 <= v < E_THD, m * s >= N_THD
 *
 */
__global__ void q_ext_stride_transpose2b_ker(sfixn *Y, const sfixn *X,
                                             sfixn u, sfixn v) 
{
    __shared__ sfixn block[N_THD];
    // block index in the kernel
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    
    // read in data from X
    // the input offset for this block is bid * N_THD
    const sfixn *din = X + (bid << E_THD);
    block[threadIdx.x] = din[threadIdx.x];
    __syncthreads();
    
    // The number of thread blocks for one subvector is snb = m * s / N_THD
    // squo = bid / snb
    // sbid = bid % snb
    sfixn snb = ((sfixn)1 << (u + v - E_THD));
    sfixn squo = bid >> (u + v - E_THD);
    sfixn sbid = bid & (snb - 1);

    // offset0 = squo * m * s + sbid * N_THD / 2
    // offset1 = squo * m * s + sbid * N_THD / 2 + (m / 2) * s
    // base = Y + offset0
    sfixn *base = Y + (squo << (u + v)) + (sbid << (E_THD - 1));
    sfixn tid = threadIdx.x;
    // iq = quo(tid, s) and ir = rem(tid, s);
    sfixn iq = (tid >> v);
    sfixn ir = tid & ((1 << v) - 1);
    // the following code is to the in-block shuffle 
    // f(i) = (rem(2iq, N_THD/s) + quo(2iq, N_THD/s)) * s + ir
    sfixn fi = (iq << 1) >> (E_THD - v);
    fi += ((iq << 1) & ((1 << (E_THD - v)) - 1));
    fi <<= v;
    fi += ir;
    
    // replace the following code by the branch-free code
    // if (tid < N_THD/2)
    //     dout[tid] = block[fi];
    // else
    //     dout[tid - N_THD / 2 + (1 << (u + v - 1))] = block[fi];
    sfixn *dout = base + (tid >> (E_THD - 1)) 
                       * ((1 << (u + v - 1)) - (1 << (E_THD - 1))); 

    dout[tid] = block[fi];
}

/**
 * @X, input array of length q * m * s = q * 2^{u + v}
 * @Y, output array of length q * m * s = q * 2^{u + v}
 * @u, m = 2^u
 * @v, s = 2^v
 *
 * Compute the general stride transposition
 * 
 * Y = (I_q @ L_2^m @ I_s) X 
 */
void q_ext_stride_transpose2_dev(sfixn *Y, const sfixn *X, sfixn q, 
                                 sfixn u, sfixn v) 
{
    if (DEBUG) assert((u >= 2) && (v >= 0) && (u + v >= E_THD) && (q >= 1));
    sfixn nb = (q << (u + v - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    if (v >= E_THD) {
        // printf("q_transpose2a_ker with q = %d, u = %d, v = %d\n", q, u, v);
        q_ext_stride_transpose2a_ker<<<nBlk, N_THD>>>(Y, X, u, v);
    } else {
        // printf("q_transpose2b_ker with q = %d, u = %d, v = %d\n", q, u, v);
        q_ext_stride_transpose2b_ker<<<nBlk, N_THD>>>(Y, X, u, v);
    }
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// I_q @ D_{2, m} @ I_s
////////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @X, input/output array of length q * 2 * m * s = q * 2^{u + v + 1}
 * @n, FFT size n = 2^e 
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @u, m = 2^u
 * @v, s = 2^v
 * 
 * Implements I_q @ D_{2, m} @ I_{s} with FFT size n
 *
 * Multiple thread blocks handle a stride, (s = 2^v is big)
 *
 * Requirements: (1) v >= E_THD, (2) 2 <= m < n
 */
__global__ void q_ext_stride_twiddle2a_ker(sfixn *X, const sfixn * const W, 
                            sfixn e, sfixn u, sfixn v, sfixn p, double pinv)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    
    // The number of thread blocks for one subvector is snb = m * s / N_THD
    // squo = bid / snb
    // sbid = bid % snb
    sfixn snb = ((sfixn)1 << (u + v - E_THD));
    sfixn squo = bid >> (u + v - E_THD);
    sfixn sbid = bid & (snb - 1);
    
    // all threads in a thread block are using the same power!
    // the base power for the grid is exp = (n / 2) / m
    // The power for the block is w^(exp * e), with e = quo(sbid, s / N_THD)
    sfixn w = W[(sbid >> (v - E_THD)) << (e - 1 - u)];
    // starting position for the block, the first m * s elements unchanged
    sfixn *base = X + (squo << (u + v + 1)) 
                    + ((sfixn)1 << (u + v)) 
                    + (sbid << E_THD);

    base[threadIdx.x] = mul_mod(w, base[threadIdx.x], p, pinv);
}

/**
 * @X, input/output array of length q * 2 * m * s = 2^{u + v + 1}
 * @n, FFT size n = 2^e 
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @u, m = 2^u
 * @v, s = 2^v
 * 
 * Implements: I_q @ D_{2, m} @ I_{s} with FFT size n
 *
 * A thread block handles multiple strides, (s = 2^v is small)
 *
 * Requirements: v < E_THD, 2 <= m < n, q >= 1, m * s >= N_THD
 */
__global__ void q_ext_stride_twiddle2b_ker(sfixn *X, const sfixn * const W, 
                            sfixn e, sfixn u, sfixn v, sfixn p, double pinv) 
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    
    // The number of thread blocks for one subvector is snb = m * s / N_THD
    // squo = bid / snb
    // sbid = bid % snb
    sfixn snb = ((sfixn)1 << (u + v - E_THD));
    sfixn squo = bid >> (u + v - E_THD);
    sfixn sbid = bid & (snb - 1);

    // the first m * s elements will be unchanged.
    sfixn *base = X + (squo << (u + v + 1)) 
                    + ((sfixn)1 << (u + v)) 
                    + (sbid << E_THD);

    // threads in a thread block use different powers.
    // the base power for the grid is egrid = (n / 2) / m.
    // the base power for the block is eblock = sbid * (N_THD / s) * egrid.
    sfixn eblock = sbid << (E_THD - v + e - 1 - u);
    sfixn tid = threadIdx.x;
    // the power for the thread eblock + s * quo(tid, s) 
    sfixn iq = eblock + ((tid >> v) << v);
    base[tid] = mul_mod(W[iq], base[tid], p, pinv);
}

/**
 * @X, input/output array of length q * 2 * m * s = q * 2^{u + v + 1}
 * @n, FFT size n = 2^e 
 * @w, n-th primitive root of unity
 * @W, powers of primitive root of unity [1, w, w^2, ..., w^{n/2-1}]
 * @u, m = 2^u
 * @v, s = 2^v
 * 
 * Implements: I_q @ D_{2, m} @ I_{s} with FFT size n
 *
 * Requirements: 2 <= m < n, s >= 1, 2 * m * s >= n and m * s >= N_THD
 */
void q_ext_stride_twiddle2_dev(sfixn *X, const sfixn * const W, sfixn e, 
                               sfixn q, sfixn u, sfixn v, sfixn p)
{
    if (DEBUG) assert((v >= 0) && (u > 0) && (u < e));
    if (DEBUG) assert((u + v >= E_THD) && (1 + u + v >= e));

    // number of blocks is q * m * s / N_THD
    sfixn nb = (q << (u + v - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }
    
    double pinv = 1 / (double)p;
    if (v >= E_THD) {
        //printf("q_twiddle2a_ker q = %d, e = %d, u = %d, v = %d\n", q, e, u, v);
        q_ext_stride_twiddle2a_ker<<<nBlk, N_THD>>>(X, W, e, u, v, p, pinv);
    } else {
        //printf("q_twiddle2b_ker q = %d, e = %d, u = %d, v = %d\n", q, e, u, v);
        q_ext_stride_twiddle2b_ker<<<nBlk, N_THD>>>(X, W, e, u, v, p, pinv);
    }
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// I_q @ DFT_2 @ I_s
////////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)
/**
 * @X, device array of length q * 2 * s = q * 2^{v + 1)
 * @Y, device array of length q * 2 * s = q * 2^(v + 1) (output)
 * @v, s = 2^v
 * @p, prime number
 *
 * Compute: Y = (I_q @ DFT2 @ I_s)X
 *
 * Requires: s >= N_THD
 *
 */
__global__
void q_ext_butterfly_ker(sfixn *Y, const sfixn * const X, sfixn v, sfixn p)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;

    // The number of thread blocks for one subvector is 
    // snb = s / N_THD
    // squo = bid / snb
    // sbid = bid % snb
    sfixn s = ((sfixn)1 << v);
    sfixn snb = (s >> E_THD);
    sfixn squo = bid >> (v - E_THD);
    sfixn sbid = bid & (snb - 1);

    sfixn *B = Y + (squo << (1 + v)) + (sbid << E_THD); 
    const sfixn *A = X + (squo << (1 + v)) + (sbid << E_THD);

    B[threadIdx.x] = add_mod(A[threadIdx.x], A[threadIdx.x + s], p);
    B[threadIdx.x + s] = sub_mod(A[threadIdx.x], A[threadIdx.x + s], p);
}

/**
 * @X, device array of length q * 2 * s = q * 2^{v + 1)
 * @Y, device array of length q * 2 * s = q * 2^(v + 1) (output)
 * @v, s = 2^v
 * @p, prime number
 *
 * Implements: I_q @ DFT2 @ I_{s}
 *
 * Requires: s >= N_THD
 */
void 
q_ext_butterfly_dev(sfixn *Y, const sfixn * const X, sfixn q, sfixn v, sfixn p)
{
    if (DEBUG) assert(v >= E_THD);

    sfixn nb = (q << (v - E_THD));
    dim3 nBlk(nb, 1, 1);
    if (nb > (1 << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    q_ext_butterfly_ker<<<nBlk, N_THD>>>(Y, X, v, p);
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD

////////////////////////////////////////////////////////////////////////////////
//
// List of 2d FFTs, main program
//
// I_q @ DFT_{m, n} = (I_q @ (DFT_m @ I_n)) (I_q @ (I_m @ DFT_n))
//
////////////////////////////////////////////////////////////////////////////////
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
 * Requirements: q >= 1, m >= , n >= 2 * N_THD 
 *
 */
void list_bivariate_stockham_dev(sfixn *X, sfixn q, sfixn em, 
    const sfixn *Wm, sfixn en, const sfixn *Wn, sfixn p) 
{
    // work space Y;
    sfixn *Y;
    cudaMalloc((void**)&Y, (q << (em + en)) * sizeof(sfixn) );

    // Step 1:
    // 
    // I_q @ I_m @ DFT_n = I_{qm} @ prod_{i = 0}^{en - 1}
    //                          (DFT_2 @ I_{n/2})                       (1.3) 
    //                          (D_{2, 2^{en - i - 1}} @ I_{2^i})       (1.2)  
    //                          (L_2^{2^{en - i}} @ I_{2^i})            (1.1) 

    // i = en - 1, both (1.1) and (1.2) are trivial
    // (1.3) becomes I_{qm} @ DFT_2 @ I_{n/2}
    q_ext_butterfly_dev(Y, X, (q << em), (en - 1), p);

    for (sfixn i = en - 2; i >= 0; --i) {
        // (1.1) 
        q_ext_stride_transpose2_dev(X, Y, (q << em), (en - i), i);
        // (1.2) 
        q_ext_stride_twiddle2_dev(X, Wn, en, (q << em), (en - i - 1), i, p);
        // (1.3)
        q_ext_butterfly_dev(Y, X, (q << em), (en - 1), p);
    }
    
    cudaMemcpy(X, Y, sizeof(sfixn)* (q << (em + en)), cudaMemcpyDeviceToDevice);

    // Step 2:
    // I_q @ (DFT_m @ I_n) = I_q @ prod_{i = 0}^{em - 1}
    //                          (DFT_2 @ I_{m/2} @ I_n)                   (2.3)     
    //                          (D_{2, 2^{em - i - 1}} @ I_{2^i} @ I_n)   (2.2)
    //                          (L_2^{2^{em - i}} @ I_{2^i} @ I_n)        (2.1) 
    
    // now effective data are in Y.
    // i = em - 1, both (2.1) and (2.2) are trivial
    // (2.3) becomes I_{q} @ DFT_2 @ I_{m/2} @ I_{n}
    q_ext_butterfly_dev(X, Y, q, em - 1 + en, p);
    for (sfixn i = em - 2; i >= 0; --i) {
        // (2.1)
        q_ext_stride_transpose2_dev(Y, X, q, (em - i), (i + en));
        // (2.2)
        q_ext_stride_twiddle2_dev(Y, Wm, em, q, (em - i - 1), (i + en), p);
        // (2.3)
        q_ext_butterfly_dev(X, Y, q, (em - 1 + en), p);
    }

    cudaFree(Y);
    if (DEBUG) checkCudaError("After list_bivariate_stockham_dev");
}

void list_bivariate_stockham_dev(sfixn *X, sfixn q, sfixn em, sfixn wm, 
                                 sfixn en, sfixn wn, sfixn p) 
{
    // initialize the primitive roots
    sfixn *Wn, *Wm;
    cudaMalloc((void**)&Wn, sizeof(sfixn) << (en - 1));
    cudaMalloc((void**)&Wm, sizeof(sfixn) << (em - 1));
    get_powers_binary(en - 1, Wn, wn, p);
    get_powers_binary(em - 1, Wm, wm, p);

    list_bivariate_stockham_dev(X, q, em, Wm, en, Wn, p);
    cudaFree(Wn);
    cudaFree(Wm);

    if (DEBUG) checkCudaError("After list_bivariate_stockham_dev");
}

////////////////////////////////////////////////////////////////////////////////
//  Some testing functions
////////////////////////////////////////////////////////////////////////////////

/**
 * DO NOT CHANGE THE FLAGS For TESTS
 */
////////////////////////////////////////////////////////////////////////////////
//  BEGIN:list_stockham_tst
////////////////////////////////////////////////////////////////////////////////
void test_list_stockham(sfixn prime,sfixn dimOfLists, sfixn lengthOfEach) 
{
    //sfixn p = 469762049;
    //sfixn p = 257;
	sfixn p = prime;
    sfixn m = dimOfLists;
    sfixn k = lengthOfEach; 
    sfixn n = ((sfixn)1 << k);
    sfixn *X = new sfixn[n*m];
    sfixn w = primitive_root(k, p);
    sfixn invw = inv_mod(w, p);
    for (sfixn u = 0; u < m; ++u) {
        for (sfixn v = 0; v < n; ++v) { X[u * n + v] = v; }
    }
 
    printf("Input: \n");
    printf("w = %d\n", w);
    for (sfixn i = 0; i < m; ++i) print_vector(n, X + i * n);    
    ///////////////////////////////////////
    list_stockham_host(X, m, k, w, p);
    ///////////////////////////////////////
    printf("Output: \n");
    for (sfixn i = 0; i < m; ++i) print_vector(n, X + i * n);    

    delete [] X;
    checkCudaError("Error found");
}

////////////////////////////////////////////////////////////////////////////////
//  END:list_stockham_tst
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//  BEGIN: TEST list_bivariate_stockham_dev
////////////////////////////////////////////////////////////////////////////////
void test_list_bivariate_stockham_dev() {
    sfixn q = 2;
    sfixn em = 3;
    sfixn en = 3;
    sfixn m = (1 << em);
    sfixn n = (1 << en);
    sfixn *X = new sfixn[q*m*n]();
    for (sfixn i = 0; i < q * m * n; ++i) X[i] = i % n;

    for (sfixn i = 0; i < q; ++i) {
        printf("matrix %d:\n", i);
        print_matrix(m, n, X + i * m * n);
    }

    sfixn p = 257;
    sfixn wn = primitive_root(en, p);
    sfixn wm = primitive_root(em, p);

    printf("n = %d, wn = %d\n", n, wn);
    printf("m = %d, wm = %d\n", m, wm);

    sfixn *X_d;   
    cudaMalloc((void**)&X_d, sizeof(sfixn)*m*n*q);
    cudaMemcpy(X_d, X, sizeof(sfixn)*m*n*q, cudaMemcpyHostToDevice);

    list_bivariate_stockham_dev(X_d, q, em, wm, en, wn, p);
    
    cudaMemcpy(X, X_d, sizeof(sfixn)*m*n*q, cudaMemcpyDeviceToHost);

    for (sfixn i = 0; i < q; ++i) {
        printf("matrix %d:\n", i);
        print_matrix(m, n, X + i * m * n);
    }

    delete [] X;
    cudaFree(X_d);
    checkCudaError("Error found");
}
////////////////////////////////////////////////////////////////////////////////
//  END: TEST list_bivariate_stockham_dev
////////////////////////////////////////////////////////////////////////////////

