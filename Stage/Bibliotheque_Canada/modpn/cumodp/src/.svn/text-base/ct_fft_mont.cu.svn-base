
#include <iostream>
#include <cassert>
#include <cstdlib>

#include "../include/ct_fft_mont.h"
#include "../include/cudautils.h"
#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/montmulmod.h"

///////////////////////////////////////////////////////////////////////////////
//  This implements Cooley-Tukey FFT without precomputation of powers
//  of roots. The modular multiplication is using Montgomery's reduction.
//  The implementation is highly inefficient. Kept as a historical copy.
///////////////////////////////////////////////////////////////////////////////

/**
 * @fp, the Fourier prime struct used in this file
 */
__constant__ fprime_t fp;

/**
 *  Initialize of fourier prime structure
 */
static inline void setup_const(const fprime_t * const fpp) {
    cudaMemcpyToSymbol(fp, fpp, sizeof(fprime_t));
}

///////////////////////////////////////////////////////////////////////////////

/*************** Implementation of the first stage (P0) ***********************
 *
 * The main task in this stage is to shuffle data in parallel. After this stage 
 * the array of data is ready for the second stage (P1), that is, performing
 * a list of small base FFTs on the shared memory space in parallel. 
 *
 * High level description
 *
 * We strive to use efficiently shared memory space which has size 2^14 bytes.
 * A thread block can use at most 2^12 integers.
 *
 * The general question is:
 *
 * Given an array of integers of length n, we treat it as a vector of matrices
 * of size h X 2. In total there are m = n / h / 2 matrcies.  
 * 
 * There are two cases to consider
 *
 * (1) A thread block can transpose several matrices if h is small.
 *
 * (2) Several blocks altogether transpose a matrix if h is large. 
 * 
 * Hence we need two types of kernels 
 *
 * (A) the height h is small, say 2^4 <= h <= 2^s
 *
 *     list_transpose2_kernel_a(sfixn m, sfixn *odata, const sfixn *idata, sfixn h);
 *
 *     In this case, m may be large.
 *
 * (B) the height h is large, say 2^{s+1} <= h <= 2^28
 * 
 *     list_transpose2_kernel_b(sfixn m, sfixn *odata, const sfixn *idata, sfixn h);
 * 
 *     In this case, we can assume that m is small.
 *
 * A thread block can hold an integer array of size 2^12 and
 * it has at most 2^9 = 512 threads. 
 * 
 * From the CUDA Occupancy Calculator, to achieve "Maximum Thread Blocks
 * Per Multiprocessor", we define a thread block to be 64 X 2 X 1.
 * Each thread block uses 2^10 bytes shared memory, that is, 2^8 int's.
 * Thus 2^8 / (2 * h) matrices can be loaded into a thread block.
 * If the cut-off exponent s = 7, then the number of matrices loaded into
 * a thread block is at least 1. Each thread needs to load 2^8 / 128 = 2
 * elements into the shared memory. 
 *
 * Hence, we set 7 as the cut-off value of s. The shared array is defined
 * as a flat array
 * 
 * __shared__ sfixn S[256];
 *
 * The total number of thread blocks is  n / 2^8. If n = 2^24 then the number 
 * of blocks is 2^16 which is greater than the maximal possible 
 * value 65535 of blockIdx.x. 
 *
 */

/**
 * list_transpose2_kernel_a : transpose a list of matrices in parallel
 *
 * n      : the size of the data array 
 * m      : positive integer, the number of matrices to be transposed.
 * idata  : the input data, entries of those row major matrices 
 * odata  : the output data of length n = m * 2 * h
 * h      : the height of each matrix
 *
 * One block handles multiple matrices. We assume that n is power of 2,
 * n >= 256, and 16 <= h <= 128.
 *
 * When n <= 2^23, the kernel configuration is
 *
 * nBlock(n / 256, 1, 1);
 * nThread(128, 1, 1);
 *
 * When n >= 2^24, the kernel configuration is
 *
 * nBlock(2^15, n / 2^23, 1);
 * nThread(128, 1, 1);
 *
 * TESTED
 *
 */

__global__ void
list_transpose2_kernel_a(sfixn *odata, const sfixn *idata, sfixn h)
{
    __shared__ sfixn block[256];
    
    // find the starting address of the matrices to be handled
    // if some blockIdx.y > 0, then the configuration must be
    // nBlock(2^15, n / 2^23, 1). Otherwise the configuration is
    // nBlock(n / 256, 1, 1), thus the starting offset is 
    // computed as blockIdx.y * 2^15 * 256 + blockIdx.x * 256
    int start = (blockIdx.y << 23) + blockIdx.x * 256;
    
    // read in several matrices into shared array of size 256.
    // since total number of threads is 128, two steps are needed.
    // first 128 elements
    block[threadIdx.x] = idata[start + threadIdx.x];   
    // second 128 elements
    block[threadIdx.x + 128] = idata[start + threadIdx.x + 128];   
    // now each block holds 256 elements
    __syncthreads();

    // write data back to global memory, column-wise
    // out = start + (threadIdx.x / h) * (2 * h) + threadIdx.x % h;
    int out = start + (threadIdx.x / h) * 2 * h + threadIdx.x % h;
    odata[out] = block[threadIdx.x * 2];
    odata[out + h] = block[threadIdx.x * 2 + 1];
    __syncthreads();
}

/**
 * list_transpose2_kernel_b : transpose a list of matrices in parallel
 *
 * n      : the size of the data array 
 * m      : positive integer, the number of matrices to be transposed.
 * idata  : the input data, entries of those row major matrices 
 * odata  : the output data of length n = m * 2 * h
 * h      : the height of each matrix
 *
 * Multiple blocks handle one matrix. We assume that n is power of 2,  
 * n >= 512 and h >= 2^8 = 256.
 *
 * When n <= 2^23, the kernel configuration is
 *
 * nBlock(n / 256, 1, 1);
 * nThread(128, 1, 1);
 *
 * When n >= 2^24 > 2^15, the kernel configuration is
 *
 * nBlock(2^15, n / 2^23, 1);
 * nThread(128, 1, 1);
 *
 * TESTED
 *
 */
__global__ void list_transpose2_kernel_b(sfixn *odata, const sfixn *idata, sfixn h)
{
    __shared__ sfixn block[256];

    // Find the starting address of the matrix to be handled.
    // If some blockIdx.y > 0, then the configuration must be
    // nBlock(2^15, n / 2^23, 1). Otherwise the configuration is
    // nBlock(n / 256, 1, 1). Thus the block id can always be
    // computed as bid = blockIdx.y * 2^15 + blockIdx.x.
    // Let r = h / 128 be the number of blocks for each matrix.
    // Then bid / r tells which matrix is working on, and bid % r
    // tells which portion has been working on.
    int bid = (blockIdx.y << 15) + blockIdx.x;
    int u = bid / (h / 128);
    int v = bid % (h / 128);
    // u * h * 2 is from the number of matrices 
    // v * 256 is from the number of blocks 
    int start = u * h * 2 + v * 256;

    // read a portion of the matrix into shared array of size 256.
    // since total number of threads is 128, two steps are needed.
    // first 128 elements
    block[threadIdx.x] = idata[start + threadIdx.x];   
    // second 128 elements
    block[threadIdx.x + 128] = idata[start + threadIdx.x + 128];   
    // now each block holds 256 elements
    __syncthreads();
    
    // write data back to global memory, column-wise
    // u * h * 2 is the starting address of the matrix
    // v * 128 is the offset for the first row
    // h + v * 128 is the offset for the second row
    //
    // Think of the following example :
    //
    // -----  
    // |A E|          ----------
    // |B F|   ====>  | A B C D|
    // |C G|          | E F G H| 
    // |D H|          ---------- 
    // -----
    // 
    // Let v = 2. Then the current block is handling a portion |C G|.
    // The starting address of C is u * h * 2 + v * 128;
    // The starting address of G is u * h * 2 + v * 128 + h. 
    int out = u * h * 2 + v * 128 + threadIdx.x;
    odata[out] = block[threadIdx.x * 2];
    odata[out + h] = block[threadIdx.x * 2 + 1];
    __syncthreads();
}

/**
 * list_transpose2 : perform a list of matrix transpositions  
 * 
 * n     : the size of the data
 * odata : an array of length n
 * idata : an array of length n
 * h     : the height of each matrix
 *
 * NOTE : this is the implementation of I_m @ L_2^h, where
 * @ denotes the tensor product, m and h are a power of 2. 
 * The product n = m * h * 2 is the size of the data, which 
 * is a power of 2. 
 *
 * We require that n >=256 and h is a multiple of 16.
 *
 * TESTED
 *
 */
void list_transpose2(sfixn n, sfixn *odata, const sfixn *idata, sfixn h)
{
    assert(odata != idata);
    // n and h are positive and power of 2.
    if (DEBUG) {
        assert((n & (n-1)) == 0);   // n is a power of 2
        assert((h & (h-1)) == 0);   // h is a power of 2
        assert(n >= 2 * h);         
        assert(n >= 256);          
        assert( (h >= 16) && ((h % 16) == 0) );
    }

    dim3 nBlock((n >> 8), 1, 1);
    dim3 nThread(128, 1, 1);
    sfixn n23 = (n >> 23);
    if (n23) {
        nBlock.x = (1 << 15);
        nBlock.y = n23;
    }

    if (h <= 128) {
        list_transpose2_kernel_a<<<nBlock, nThread>>>(odata, idata, h);
    } else {
        list_transpose2_kernel_b<<<nBlock, nThread>>>(odata, idata, h);
    }
    cudaThreadSynchronize();
    if (DEBUG) checkCudaError("Error found after inner_list_transpose2");
}

/**
 * data_shuffle : perform the data shuffle before calling the DFT bases.
 * 
 * data : device array of length n = 2^k
 *    k : exponent of n = 2^k
 *    l : exponent of m = 2^l
 *    m : size of DFT base 
 * 
 * This is implementing the stage (P0) of the notes.
 *
 * Assumptions : k >= 8 and l >= 4. 
 *
 * Example:
 *
 * If k = 8 and l = 4, then the shuffle is defined by the operator
 *
 * (I_8 @ L_2^16) (I_4 @ L_2^32) (I_2 @ L_2^64) (I_1 @ L_2^128)
 *
 * TESTED
 *
 **/
void data_shuffle(sfixn k, sfixn l, sfixn *data) 
{
    // if k equals l, do nothing.
    if (k == l) return;
    if (DEBUG) assert((k >= 8) && (l >= 4));
    sfixn n = ((sfixn)1 << k);
    // device array as work space
    sfixn *W = NULL;
    cudaMalloc((void**)&W, n*sizeof(sfixn));
    if (DEBUG) checkCudaError("Error found in data_shuffle, initial stage");
    
    // perform out-of-place matrix transpositions
    for (sfixn i = 1; i <= k - l; ++i) {
        if (i & (sfixn)1) {
            list_transpose2(n, W, data, (sfixn)1 << (k-i));
        } else {
            list_transpose2(n, data, W, (sfixn)1 << (k-i));
        }
    }
    
    // if k-l is odd then move to data, otherwise do nothing
    if ((k - l) & (sfixn)1) {
        cudaMemcpy(data, W, n*sizeof(sfixn), cudaMemcpyDeviceToDevice);
    }
    cudaFree(W);
    if (DEBUG) checkCudaError("Error found in data_shuffle, ending stage");
}

/*************** Implementation of the second stage (P1) ***********************
 * 
 * This stage is to implement an efficient base FFT of size m and to perform
 * a list of FFTs on shared memory space.   
 *
 **/

/**
 * list_fft_kernel : perform a list of 1-d FFTs in parallel  
 * 
 * A  : input array of length n
 * n  : power of 2
 * m  : small fft size, power of 2, m <= n
 * wn : n-th primitive root of unity in its Montgomery representation
 * wm : m-th primitive root of unity in its Montgomery representation
 * p  : prime number
 *
 * NOTE : this is implementing the stage (P1), that is, handling the formula
 *
 * I_{n / m} @ DFT_m
 *
 * The shared memory space is used for each small FFT. 
 * In total each block needs nThread * m * 4 bytes shared memory. 
 *
 * Each thread, nThread in total, performs a 1d FFT in place. 
 * Here m should be a power of 2, and at least 16.
 *
 * The constraints on nThread, m and n:
 *
 * (0) the number of blocks is at least 1
 * (1) nThread is at most 512.
 * (2) m is at least 16.
 * (3) the product of nThread and m is at most 4096 = 2^12, the shared memory size. 
 * (4) the total number of registers in each block is 8192 = 2^13. 
 * (5) DFT_m uses at most 18 registers.  
 * 
 * That is,
 *
 *  n >= m * nThread
 *  nThread <= 2^9
 *  m >= 2^4
 *  m * nThread <= 2^12  
 *  nThread * 18 <= 2^13 (redundant)
 *
 * That implies, 
 * 
 * [m, nThread] = [2^4,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8}]
 * [m, nThread] = [2^5,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7}]
 * [m, nThread] = [2^6,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6}]
 * [m, nThread] = [2^7,  {1, 2^1, 2^2, 2^3, 2^4, 2^5}]
 * [m, nThread] = [2^8,  {1, 2^1, 2^2, 2^3, 2^4}]
 * [m, nThread] = [2^9,  {1, 2^1, 2^2, 2^3}]
 * [m, nThread] = [2^10, {1, 2^1, 2^2}]
 * [m, nThread] = [2^11, {1, 2^1}]
 * [m, nThread] = [2^12, {1}]
 * 
 * If 1 <= n / (nThread * m) <= 2^15, then the kernel configuration is
 * 
 * dim3 nBlock(n / (nThread * m), 1, 1);
 *
 * otherwise it is
 *
 * dim3 nBlock(2^15, n / (nThread * m * 2^15), 1); 
 *
 * TESTED
 *
 */
__global__ void 
list_fft_kernel(sfixn n, sfixn *A, sfixn wm, sfixn p, int m, int nThread) 
{
    //extern __shared__ sfixn W[nThread * m];
    extern __shared__ sfixn W[];

    // each thread is in charge of a subvector of length m
    // each block is in charge of a subvector of length m * nThread
    sfixn *AA = A + ((blockIdx.y << 15) + blockIdx.x) * m * nThread;
    
    // copy data from global memory to shared memory
    // all memory accesses are coalesced if nThread is a multiple of 16
    sfixn i, k = threadIdx.x;
    for (i = 0; i < m; ++i) {
        W[k]  = AA[k];
        k += nThread;
    }
    __syncthreads();
    
    // in place DFT computation on the shared memory space
    k = threadIdx.x;
    gsnn_fft_ip(m, floor_log2(m), W + k * m, p, wm);
    __syncthreads();

    // copy data from shared memory back to global memory 
    // all memory accesses are coalesced if nThread is a multiple of 16
    for (i = 0; i < m; ++i) {
        AA[k] = W[k];
        k += nThread;
    }
    __syncthreads();
}

/**
 * n  : power of 2, array size of A
 * A  : input array of length n
 * m  : power of 2, small FFT size
 * ell: m = 2^ell
 * wn : n-th primitive root of unity, in its Montgomery REP
 * p  : prime number
 *
 * TESTED
 *
 */
void list_fft(sfixn k, sfixn *A, sfixn wn, sfixn p, int nThread, int ell, 
              const fprime_t * const fpp)
{   
    const sfixn n = ((sfixn)1 << k);
    const sfixn m = ((sfixn)1 << ell);

    // wm is in its Montgomery representation
    sfixn wm = fpow_mod(wn, (n >> ell), fpp);

    int nb = n / (nThread * m);
    dim3 nBlock(nb, 1, 1);
    int nb15 = nb >> 15;
    if (nb15) {
        nBlock.x = ((sfixn)1 << 15);
        nBlock.y = nb15;
    }
    list_fft_kernel<<<nBlock, nThread, 
        nThread * m * sizeof(sfixn)>>> (n, A, wm, p, m, nThread);

    cudaThreadSynchronize();
    if (DEBUG) checkCudaError("Error found in list_fft, ending stage");
}

/*************** Implementation of the third stage (P2) ************************
 *
 * After the second stage, we combine those base FFTs to the FFT of size n. 
 * A list of operations need to be performed:
 *
 * I_{2^{i-1}} @ DFT_2 @ I_{2^{k-i}} (I_{2^{i-1}} @ D_{2, 2^{k-i}})
 *
 * for i from k - \ell to 1. Hence, k - i is from ell to k - 1, where n = 2^k 
 * and m = 2^{\ell}.
 *
 * Each application of the above formula is a group of butterfly operations.
 * And each group of butterfly operations 
 *
 * ((DFT_2 @ I_{2^{k-i}}) D_{2, 2^{k-i}})
 *
 * operates on a subarray of length 2*q = 2^{k-i+1}. More precisely the task of 
 * those butterflies are
 *
 * for (j = 0; j < q; ++j) {
 *     y[j] = x[j] + x[q+j] * wi^j;
 *     y[q+j] = x[j] - x[q+j] * wi^j;
 * }
 *
 * In total, there are 2^{i-1} groups. 
 */

/**
 * list_butterfly_kernel_a : out of place parallel buttterfly operations
 * 
 * k  : exponent of n = 2^k
 * l  : exponent of m = 2^l
 * i  : the i-th step
 * p  : prime number
 * wi : 2^{k-i+1}-th primitive root of unity modulo p, in its Montgomery REP
 * odata : array of length n
 * idata : array of length n
 *
 * Note :
 *
 * This kernel implements the formula
 *
 * I_{2^(i-1)} @ ((DFT_2 @ I_{2^{k-i}}) D_{2, 2^{k-i}} )
 *
 * In total, there are 2^{i-1} groups of butterflies; 
 * each group consists of q = 2^{k-i} butterfly operations and 
 * uses 2^{k-i+1} elements in total.
 *
 * Each thread block contains 128 = 2^7 threads,
 * 
 * nThread(128, 1, 1);
 * 
 * We assume that each thread block can process multiple groups,
 * and at least 1 group. That is, q = 2^{k-i} is small.   
 * If we just process one group in a thread block, 
 * then the number of butterfly operations in this group should be 
 * 128. Since k - i is from ell to k - 1, we require that 
 * ell <= k - i  <= 7, or m <= q <= 128.
 *
 * Each block can process a subarray of length 256. In total
 * we need n / 256 = 2^{k - 8} blocks. Each block consists of
 * 256 / (2 * q) = 128 / q groups of butterflies.
 *
 * Two cases arise, that is, 
 * (1) 2^{k-8} at most 2^15, or k <= 23
 * 
 * nBlock(2^{k-8}, 1, 1); 
 *
 * (2) 2^{k-8} at least 2^16, or k >= 24
 *
 * nBlock(2^15, 2^{k-23}, 1); 
 *
 * Example,
 * 
 * If k = 9 and i = 5, the number of thread blocks is 2^{k - 8} = 2. 
 * The stride q = 2^{k - i} = 16.
 *
 * The first block handles the first 256 elements and the second block
 * handles the second 256 elements.
 *  
 * There are 128 / q = 8 groups for a thread block.
 * -----                   -----
 * | A | threads 0 - 15    | A |
 * | A'|                   | A'|
 * | B | threads 16 - 31   | B |
 * | B'|                   | B'|
 * | C | threads 32 - 47   | C |
 * | C'|                   | C'|
 * | D | threads 48 - 63   | D |
 * | D'|                   | D'|
 * | E | threads 64 - 79   | E |
 * | E'|                   | E'|
 * | F | threads 80 - 95   | F |
 * | F'|                   | F'|
 * | G | threads 96 - 111  | G |
 * | G'|                   | G'|
 * | H | threads 112 - 127 | H |
 * | H'|                   | H'|
 * -----                   -----
 */
__global__ void list_butterfly_kernel_a(sfixn k, sfixn q, sfixn *odata,
                    const sfixn *idata, sfixn wi, sfixn p)
{
    // find the starting position
    // if some blockIdx.y > 0 then the case (2),
    // otherwise it is the case (1).
    int bid = blockIdx.x + (blockIdx.y << 15);
    // threadIdx.x decides which group it handles with.
    const sfixn gval = threadIdx.x / q;  // gval is from 0 to 128 / q - 1
    const sfixn rval = threadIdx.x % q;  // rval is from 0 to q - 1.
    
    // twiddle factor
    // w is in its Montgomery REP
    const sfixn w = fpow_mod(wi, rval, &fp);  
    // input data x1 and x2
    const sfixn offset = bid * 256 + gval * 2 * q + rval;
    const sfixn x1 = idata[offset];
    const sfixn x2 = idata[offset + q];
    // y1 = x1 + x2 * w 
    // y2 = x1 - x2 * w;
    const sfixn t = fourier_reduction(x2, w, &fp);     // t = x2 * w
    odata[offset] = add_mod(x1, t, p);     // first output
    odata[offset + q] = sub_mod(x1, t, p); // second output
}

/**
 * list_butterfly_kernel_b : out of place parallel buttterfly operations
 * 
 * k  : exponent of n = 2^k
 * l  : exponent of m = 2^l
 * i  : the i-th step
 * p  : prime number
 * wi : 2^{k-i+1}-th primitive root of unity modulo p, in its Montgomery REP
 * odata : array of length n
 * idata : array of length n
 *
 * Note :
 *
 * This kernel implements the formula
 *
 * I_{2^(i-1)} @ ((DFT_2 @ I_{2^{k-i}}) D_{2, 2^{k-i}} )
 *
 * In total, there are 2^{i-1} groups of butterflies; 
 * each group consists of q = 2^{k-i} butterfly operations and 
 * uses 2 * q = 2^{k-i+1} elements in total.
 *
 * Each thread block contains 128 = 2^7 threads,
 * 
 * nThread(128, 1, 1);
 * 
 * We assume that multiple thread blocks (at least 2) process one groups,
 * that is, q = 2^{k-i} is big. Since k - i is from ell to k - 1,
 * we require that 8 <= k - i <=  k - l, or 256 <= q <=  n / m.
 *
 * Each block can process a subarray of length 256. In total we need
 * n / 256 = 2^{k - 8} blocks. Each group needs 
 *
 *   2^{k - 8}
 * ------------ = (q / 128) threads blocks.
 *   2^{i - 1}
 *
 * Two cases arise, that is, 
 *
 * (1) 2^{k-8} at most 2^15, or k <= 23
 * 
 * nBlock(2^{k-8}, 1, 1); 
 *
 * (2) 2^{k-8} at least 2^16, or k >= 24
 *
 * nBlock(2^15, 2^{k-23}, 1); 
 *
 * Example,
 * 
 * If q = 2^{k-i} = 512, then (2 * q) / 256 = q / 128 = 4 thread blocks are
 * needed to process a group of butterfly operations. 
 *
 *  ------                                    -------
 *  | A  |  the first thread block A and A'   | AA  | 
 *  | B  |  the second thread block B and B'  | BB  | 
 *  | C  |  the third thread block C and C'   | CC  | 
 *  | D  |  the fourth thread block D and D'  | DD  | 
 *  | A' |                                    | AA' | 
 *  | B' |  the results are stored at AA,     | BB' | 
 *  | C' |  BB, CC, DD, AA', BB', CC' and DD' | CC' | 
 *  | D' |                                    | DD' | 
 *  ------                                    -------
 *  
 *  The length of A (A'), B (B'), C (C'), and D (D') are 128. 
 *
 *  If k = 9 and i = 1, then the number of thread blocks is q / 128 = 2.
 *  And there is only one group of butterflies.
 *
 *  ---                                        ---    
 * | A |  the thread block 0 handles A and A' | AA | 
 * | B |                                      | BB |
 * | A'|  the thread block 1 handles B and B' | AA'|
 * | B'|                                      | BB'|
 *  ---                                        ---
 *
 *  The length of A (A'), B (B') are 128. 
 */
__global__ void list_butterfly_kernel_b(sfixn k, sfixn q, sfixn *odata,
                    const sfixn *idata, sfixn wi, sfixn p)
{
    // find the starting position
    // if some blockIdx.y > 0 then the case (2),
    // otherwise it is the case (1).
    int bid = blockIdx.x + (blockIdx.y << 15);
    // gval is the working group number
    const sfixn gval = bid / (q / 128);
    // rval is the part number in a group
    const sfixn rval = bid % (q / 128);
    // offset to idata
    const sfixn start = gval * 2 * q;
    const sfixn ind = rval * 128 + threadIdx.x; 
    // twiddle factor
    // w is in its Montgomery REP
    const sfixn w = fpow_mod(wi, ind, &fp);  
    // input data x1 and x2
    const sfixn x1 = idata[start + ind];
    const sfixn x2 = idata[start + ind + q];
    // y1 = x1 + x2 * w 
    // y2 = x1 - x2 * w;
    // t is in Montgomery REP
    const sfixn t = fourier_reduction(x2, w, &fp); // t = x2 * w
    odata[start + ind] = add_mod(x1, t, p);        // first output
    odata[start + ind + q] = sub_mod(x1, t, p);    // second output
}

/**
 * A  : the data vector of length n = 2^k
 * k  : exponent of n = 2^k
 * l  : exponent of m = 2^l
 * wn : n-th primitive root of unity modulo p, in its Montgomery REP
 * p  : prime number
 *
 * An implementation that can handle k >= 24.
 */
void list_butterfly(sfixn k, sfixn l, sfixn *A, sfixn wn, sfixn p, 
                    const fprime_t * const fpp)
{
    if (k <= l) return; 
    const sfixn n = ((sfixn)1 << k);
    // device array as work space
    sfixn *W = NULL;
    cudaMalloc((void**)&W, n*sizeof(sfixn));
    if (DEBUG) checkCudaError("Error found in list_butterfly, initial stage");
    
    dim3 nThread(128, 1, 1);
    dim3 nBlock((sfixn)1 << (k - 8), 1, 1);
    if (k >= 24) {
        nBlock.x = (sfixn)1 << 15;
        nBlock.y = (sfixn)1 << (k - 23); 
    }
    sfixn ng = (sfixn)1 << (k - l - 1);
    for (sfixn i = k - l; i >= 1; --i) {
        // wi = wn^{2^{i-1}} is a 2^{k-i+1}-th primitive root of unity
        // wi is in its Montgomery REP
        sfixn wi = fpow_mod(wn, ng, fpp);
        // if k - l - i is odd then data from W to A.
        // if k - l - i is even then data from A to W. 
        // During the first iteration, k - l - i = k - l - k + l = 0 is even
        // W will be initialized by A. Thus not need to initialize W. 
        sfixn q = (sfixn)1 << (k - i);
        if ((k - l - i) % 2) {
            if (q < 256) {
                list_butterfly_kernel_a<<<nBlock, nThread>>>(k, q, A, W, wi, p);
            } else {
                list_butterfly_kernel_b<<<nBlock, nThread>>>(k, q, A, W, wi, p);
            }
        } else {
            if (q < 256) {
                list_butterfly_kernel_a<<<nBlock, nThread>>>(k, q, W, A, wi, p);
            } else {
                list_butterfly_kernel_b<<<nBlock, nThread>>>(k, q, W, A, wi, p);
            }
        }
        ng >>= 1;
        cudaThreadSynchronize();
    }
    // if k - l is odd then move data from work space to A
    // if k - l is even then do nothing 
    if ((k - l) % 2) {
        cudaMemcpy(A, W, n*sizeof(sfixn), cudaMemcpyDeviceToDevice);
    }
    cudaFree(W);
    if (DEBUG) checkCudaError("Error found in list_butterfly, ending stage");
}

/**
 * tensor_mont_fft : compute DFT in parallel using tensor product.
 *
 * n  : DFT size, length of A 
 * k  : exponent of n = 2^k
 * m  : base case size
 * l  : exponent of m = 2^l
 * A  : coefficient vector on device
 * wn : n-th primitive root of unity modulo p, in its normal REP
 * p  : prime number
 *
 * [m, nThread] = [2^4,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8}]
 * [m, nThread] = [2^5,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7}]
 * [m, nThread] = [2^6,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6}]
 * [m, nThread] = [2^7,  {1, 2^1, 2^2, 2^3, 2^4, 2^5}]
 * [m, nThread] = [2^8,  {1, 2^1, 2^2, 2^3, 2^4}]
 * [m, nThread] = [2^9,  {1, 2^1, 2^2, 2^3}]
 * [m, nThread] = [2^10, {1, 2^1, 2^2}]
 * [m, nThread] = [2^11, {1, 2^1}]
 * [m, nThread] = [2^12, {1}]
 *
 */
void tensor_mont_fft(sfixn n, sfixn k, sfixn *A, sfixn wn, sfixn p) {
    fprime_t fp_h;
    init_fourier_prime(&fp_h, p);
    setup_const(&fp_h);

    const int ell = 4;
    const int nThread = 16;
    data_shuffle(k, ell, A);   
    if (DEBUG) checkCudaError("tensor_mont_fft : after data_shuffle");
    // convert wn into its Montgomery REP
    wn = frep(wn, &fp_h);
    list_fft(k, A, wn, p, nThread, ell, &fp_h);
    if (DEBUG) checkCudaError("tensor_mont_fft : after list_fft");
    list_butterfly(k, ell, A, wn, p, &fp_h);
    if (DEBUG) checkCudaError("tensor_mont_fft : after list_butterfly");
}

void tensor_mont_fft(sfixn n, sfixn k, sfixn *A, sfixn wn, sfixn p, 
    const uni_mont_fft_plan &plan) 
{
    fprime_t fp_h;
    init_fourier_prime(&fp_h, p);
    setup_const(&fp_h);

    int ell = plan.b_exp();
    int nThread = plan.n_thd();
    data_shuffle(k, ell, A);   
    if (DEBUG) checkCudaError("tensor_mont_fft : after data_shuffle");
    // convert wn into its Montgomery REP
    wn = frep(wn, &fp_h);
    list_fft(k, A, wn, p, nThread, ell, &fp_h);
    if (DEBUG) checkCudaError("tensor_mont_fft : after list_fft");
    list_butterfly(k, ell, A, wn, p, &fp_h);
    if (DEBUG) checkCudaError("tensor_mont_fft : after list_butterfly");
}

/*****************************************************************************/
using namespace std;

int main(int argc, char** argv) {
    const sfixn p = 469762049;
    //const sfixn p = 7340033;
    //const sfixn p = 257;
    //const sfixn p = 65537;
    //const sfixn p = 12289;
    //const sfixn p = 7681;
    sfixn E = fourier_degree(p);
    if (argc < 2) { cout << "Please enter the fft size 2^e" << endl; return 0; }
    sfixn e = atoi(argv[1]);
    if (e < 10) { cout << "fft size is at least 1024" << endl; return 0;}

    sfixn w = primitive_root(e, p);
    sfixn inw = inv_mod(w, p);
    sfixn n = (1 << e);

    sfixn *A = new int[n];
    srand(time(NULL));
    for (int i = 0; i < n; ++i) A[i] = rand() % p;
    //for (int i = 0; i < n; ++i) A[i] =  i % p;
    //print_vector(8, A);
    sfixn *Ad;

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    cudaMalloc((void **)&Ad, n*sizeof(sfixn));
    cudaMemcpy(Ad, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);
    uni_mont_fft_plan plan(n, 16, 6);
    if (plan.is_valid_plan()) {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        //////////////////////////////////
        tensor_mont_fft(n, e, Ad, w, p, plan);
        //tensor_mont_fft(n, e, Ad, inw, p, plan);
        //////////////////////////////////
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(A, Ad, sizeof(sfixn)*n, cudaMemcpyDeviceToHost);
        //for (int i = 0; i < n; ++i) A[i] = quo_mod(A[i], n, p);
        //print_vector(8, A);
        printf("Computing time for %d-FFT %8.3f (ms)\n", e, elapsedTime);
    }

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    printf("    Total time for %d-FFT %8.3f (ms)\n", e, elapsedTime2);

    cudaFree(Ad);
    delete [] A;
    return 0;
}
