
#include <iostream>
#include <cassert>
#include "../include/defines.h"
#include "../include/inlines.h"
#include "../include/cudautils.h"
#include "../include/printing.h"
#include "../include/ct_fft.h"

////////////////////////////////////////////////////////////////////////////////
//  This file implements Cooley-Tukey FFT with jumped powers computed from
//  host. Precomputation of roots is expensive, thus for experimentation only.
////////////////////////////////////////////////////////////////////////////////

/**
 * The question is:
 *
 * Given an array of integers of length n, we treat it as a vector of matrices
 * of size h X 2. In total there are m = n / h / 2 matrcies. Let h = 2^e. 
 * 
 * There are two cases to consider
 *
 * (A) the height h is small
 *
 * list_transpose2_kernel_a(sfixn m, sfixn *odata, const sfixn *idata, sfixn e);
 *
 * In this case, m may be large.
 *
 * (B) the height h is large
 * 
 * list_transpose2_kernel_b(sfixn m, sfixn *odata, const sfixn *idata, sfixn e);
 * 
 * In this case, we can assume that m is small.
 */

///////////////
// CONSTANTS //
///////////////

// each thread block consists of N_THD number of threads
#define E_THD (7)
#define N_THD (1 << E_THD)
// each thread block uses N_SHD integers in the shared memory
#define E_SHD (0 + E_THD)
#define N_SHD (1 << E_SHD)
// the number of data each thread moves
#define N_DAT (1 << (E_SHD - E_THD))

/**
 * list_transpose2_kernel_a : transpose a list of matrices in parallel
 *
 * n     : the size of the data array 
 * m     : positive integer, the number of matrices to be transposed
 * idata : the input data, entries of those row major matrices 
 * odata : the output data of length n = m * 2 * h
 * h     : the height of each matrix
 * e     : h = 2^e
 *
 * The case h being small, and one thread block transposes multiple matrices.
 * Each thread block handle (N_SHD / 2 / h) matrices, and we require
 * h < N_SHD.
 *
 * For example, let n = 16, m = 2 and h = 4
 *
 * 0  1  
 * 2  3  
 * 4  5    ====>  0  2  4  6
 * 6  7           1  3  5  7
 * 8  9           8  10 12 14   
 * 10 11          9  11 13 15
 * 12 13
 * 14 15
 */

__global__ void
list_transpose2_kernel_a(sfixn *odata, const sfixn *idata, sfixn e)
{
    __shared__ sfixn block[N_SHD];
    
    // block index in the kernel
    int bid = (blockIdx.y << 15) + blockIdx.x;
    int i;

    // read in several matrices into shared array of size N_SHD.
    sfixn *shared = block;
    const sfixn *din = idata + (bid << E_SHD);  
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        shared[threadIdx.x] = din[threadIdx.x];
        shared += N_THD;
        din += N_THD; 
    }
    // now each block holds N_SHD elements
    __syncthreads();

    int tid, iq, ir, offset;
    // the starting address of this block
    sfixn *dout = odata + (bid << E_SHD); 
    // write data back to global memory, column-wise
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        // virtual thread id in each loop 
        tid = (i << E_THD) + threadIdx.x;
        // iq = quo(tid, 2h) tells which matrix is to be transposed
        iq = (tid >> (1 + e)); 
        // ir = rem(tid, 2h) tells which entry is to be handled by the thread
        ir = (tid & ((2 << e) - 1));
        // if ir is even, write to start + iq * 2 * h + ir / 2
        // if ir is odd, write to start + iq * 2 * h + h + ir / 2
        offset = (iq << (1 + e)) + ((ir & 1) << e) + (ir >> 1);
        dout[offset] = block[tid];
    }
    __syncthreads();
}

/**
 * list_transpose2_kernel_b : transpose a list of matrices in parallel
 *
 * n     : the size of the data array 
 * m     : positive integer, the number of matrices to be transposed.
 * idata : the input data, entries of those row major matrices 
 * odata : the output data of length n = m * 2 * h
 * h     : the height of each matrix
 * e     : h = 2^e
 *
 * Multiple blocks handle one matrix; the number of thread blocks needed
 * is nb = 2 * h / N_SHD. We require nb >= 2 or h >= N_SHD.
 *
 */
__global__ void 
list_transpose2_kernel_b(sfixn *odata, const sfixn *idata, sfixn e)
{
    __shared__ sfixn block[N_SHD];
    // block index in the kernel
    int bid = (blockIdx.y << 15) + blockIdx.x;
    // nb = 2 * h / N_SHD; the number of blocks per matrix
    int nb = (1 << (1 + e - E_SHD)); 
    // bq = quo(bid, nb) tells which matrix the thread block is working on.
    int bq = (bid >> (1 + e - E_SHD));
    // br = rem(bid, nb) tells which portion the thread block is working on. 
    int br = (bid & (nb - 1)); 
    
    int i, tid;
    // read matrices into the shared array of size N_SHD.
    const sfixn *din = idata + (bid << E_SHD);  
    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        // virtual thread id in each loop 
        tid = (i << E_THD) + threadIdx.x;
        block[tid] = din[tid];
    }
    // now each block holds N_SHD elements
    __syncthreads();

    // the starting address of this block
    sfixn *dout = odata + (bq << (1 + e)) + (br << (E_SHD - 1)) ; 
    // write out the data in the shared array 

    #pragma unroll
    for (i = 0; i < N_DAT; ++i) {
        // virtual thread id in each loop 
        tid = (i << E_THD) + threadIdx.x;
        // if tid is even, write to dout + tid / 2
        // if tid is odd, write to dout + tid / 2 + h
        dout[(tid >> 1) + ((tid & 1) << e)] = block[tid];
    }
    __syncthreads();
}

/**
 * Perform a list of matrix transpositions  
 * 
 * n     : the size of the data
 * odata : an array of length n
 * idata : an array of length n
 * h     : the height of each matrix
 * e     : h = 2^e
 *
 * NOTE : this is the implementation of I_m @ L_2^h, where
 * @ denotes the tensor product, m and h are a power of 2. 
 * The product n = m * h * 2 is the size of the data, which 
 * is a power of 2. 
 *
 */
void list_transpose2(sfixn n, sfixn *odata, const sfixn *idata, sfixn e)
{
    if (DEBUG) assert(odata != idata);
    if (DEBUG) assert(n >= (2 << e));
    if (DEBUG) assert(n >= N_SHD);

    int nThread = N_THD;
    int nb = (n >> E_SHD);
    dim3 nBlock(nb, 1, 1);
    if (nb > (1 << 15)) { nBlock.x = (1 << 15); nBlock.y = (nb >> 15); }

    // if N_SHD > h calls kernel a, otherwise calls kernel b
    if (E_SHD > e) {
        // printf("Calling kernel a : n = %d, nb = %d, e = %d\n", n, nb, e);
        list_transpose2_kernel_a<<<nBlock, nThread>>>(odata, idata, e);
    } else {
        // printf("Calling kernel b : n = %d, nb = %d, e = %d\n", n, nb, e);
        list_transpose2_kernel_b<<<nBlock, nThread>>>(odata, idata, e);
    }
    cudaThreadSynchronize();
}

void test_list_transpose2() {
    const sfixn p = 257;
    sfixn n = 32;
    sfixn *A = new int[n];
    for (int i = 0; i < n; ++i) A[i] =  i % p;
    sfixn *idata, *odata;
    cudaMalloc((void **)&idata, n*sizeof(sfixn));
    cudaMalloc((void **)&odata, n*sizeof(sfixn));

    printf("Input  A : ");
    print_vector(n, A);

    cudaMemcpy(idata, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) A[i] =  0;
    cudaMemcpy(odata, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);

    list_transpose2(n, odata, idata, 2);

    for (int i = 0; i < n; ++i) A[i] = 0;
    cudaMemcpy(A, odata, sizeof(sfixn)*n, cudaMemcpyDeviceToHost);

    printf("Output A : ");
    print_vector(n, A);

    cudaFree(idata);
    cudaFree(odata);
    delete [] A;
}

/**
 * Perform the data shuffle before calling the DFT bases.
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
 **/
void data_shuffle(sfixn k, sfixn l, sfixn *data) 
{
    // if k equals l, do nothing.
    if (k == l) return;
    sfixn n = ((sfixn)1 << k);
    // device array as work space
    sfixn *W = NULL;
    cudaMalloc((void**)&W, n*sizeof(sfixn));
    if (DEBUG) checkCudaError("Error found in data_shuffle, initial stage");
    
    // perform out-of-place matrix transpositions
    for (sfixn i = 1; i <= k - l; ++i) {
        if (i & (sfixn)1) {
            list_transpose2(n, W, data, (k-i));
        } else {
            list_transpose2(n, data, W, (k-i));
        }
    }
    
    // if k-l is odd then move to data, otherwise do nothing
    if ((k - l) & (sfixn)1) {
        cudaMemcpy(data, W, n*sizeof(sfixn), cudaMemcpyDeviceToDevice);
    }
    cudaFree(W);
    if (DEBUG) checkCudaError("Error found in data_shuffle, ending stage");
}

#undef E_THD
#undef N_THD
#undef E_SHD
#undef N_SHD
#undef N_DAT
////////////////////////////////////////////////////////////////////////////////

/*************** Implementation of the second stage (P1) **********************/

/**
 * Gentleman-Sande FFT : natural input, revsersed output
 *
 * n  : fft size, power of 2
 * A  : input array
 * p  : prime number
 * wn : n-th primitive root of unit
 *
 * 18 registers will be used by this kernel.
 * 
 * 1, wn, wn^2, ..., wn^(n/2) will be used.
 *
 */
__device__ __host__ void fft_ip_inner(sfixn n, sfixn *A, sfixn wn, sfixn p) {
    sfixn num_probs = 1;
    sfixn prob_size = n;
    sfixn wk = wn;
    double pinv = 1 / (double) p;
    while (prob_size > 1) {
        sfixn stride = (prob_size >> 1);
        for (sfixn k = 0; k < num_probs; ++k) {
            sfixn j_first = k * prob_size;
            sfixn j_last = j_first + stride - 1;
            sfixn w = 1;
            for (sfixn j = j_first; j <= j_last; ++j) {
                sfixn t = A[j];
                A[j] = add_mod(t, A[j + stride], p);
                A[j+stride] = mul_mod(w, sub_mod(t, A[j+stride], p), p, pinv);  
                w = mul_mod(w, wk, p, pinv);
            }
        }
        num_probs <<= 1;
        prob_size = stride; 
        wk = mul_mod(wk, wk, p, pinv);
    }
}

/*
 * A should be in the shared memory. Transform A into natural order,
 * taking the bit-reverse ordering bit_reverse takes only 32-bit integers!
 *
 * */
__device__ __host__ 
void fft_ip(sfixn n, sfixn e, sfixn *A, sfixn wn, sfixn p)
{
    fft_ip_inner(n, A, wn, p);
    sfixn v, target;
    for (sfixn i = 0; i < n; ++i) {
        target = bit_reverse(i, e);
        if (target > i) { v = A[target]; A[target] = A[i]; A[i] = v; }
    }
}

/**
 * Perform a list of 1-d FFTs in parallel  
 * 
 * A  : input array of length n
 * n  : power of 2
 * m  : small fft size, power of 2, m <= n, m = 2^l
 * wn : n-th primitive root of unity
 * wm : m-th primitive root of unity
 * p  : prime number
 *
 * NOTE : this is implementing the stage (P1), that is, handling the formula
 *
 * I_{n / m} @ DFT_m
 *
 * The shared memory space is used for each small FFT. 
 * Each thread, nThread in total, performs a 1d FFT in place. 
 * Here m should be a power of 2, and at least 16.
 *
 */
__global__ void 
list_fft_kernel(sfixn n, sfixn *A, sfixn eThd, sfixn l, sfixn wm, sfixn p) 
{
    extern __shared__ sfixn W[];

    // each thread is in charge of a subvector of length m
    // each block is in charge of a subvector of length m * nThd
    sfixn bid = ((blockIdx.y << 15) + blockIdx.x);  
    sfixn *AA = A + (bid << (l + eThd));
    
    // copy data from global memory to shared memory
    // all memory accesses are coalesced if nThread is a multiple of 16
    sfixn nThd = (1 << eThd), m = (1 << l);
    sfixn i, k = threadIdx.x;
    for (i = 0; i < m; ++i) { W[k] = AA[k]; k += nThd; }
    __syncthreads();
    
    // in place DFT computation on the shared memory space
    fft_ip(m, l, W + threadIdx.x * m, wm, p);
    __syncthreads();

    // copy data from shared memory back to global memory 
    // all memory accesses are coalesced if nThd is a multiple of 16
    k = threadIdx.x;
    for (i = 0; i < m; ++i) { AA[k] = W[k]; k += nThd; }
    __syncthreads();
}

/**
 * n   : power of 2, array size of A
 * A   : input array of length n
 * m   : power of 2, small FFT size
 * l   : m = 2^l
 * wn  : n-th primitive root of unity
 * p   : prime number
 *
 */
void list_fft(sfixn k, sfixn *A, sfixn wn, sfixn eThd, sfixn l, sfixn p)
{   
    const sfixn n = (1 << k);
    sfixn wm = pow_mod_pow2(wn, (k - l), p);
    int nb = (n >> (eThd + l));
    dim3 nBlk(nb, 1, 1);
    int nb15 = nb >> 15;
    if (nb15) { nBlk.x = (1 << 15); nBlk.y = nb15; }
    sfixn shared_mem = sizeof(sfixn) << (eThd + l);
    sfixn nThd = (1 << eThd);
    list_fft_kernel<<<nBlk, nThd, shared_mem>>>(n, A, eThd, l, wm, p);
    cudaThreadSynchronize();
    if (DEBUG) checkCudaError("Error found in list_fft, ending stage");
}

void test_list_fft() {
    const sfixn p = 257;
    sfixn e = 5;
    sfixn n = 32;
    sfixn *A = new int[n];
    sfixn w = primitive_root(e, p);
    for (int i = 0; i < n; ++i) A[i] = i % p;

    printf("w = %d\n", w);
    printf("Input  : "); print_vector(n, A);
    fft_ip(n, e, A, w, p);
    printf("Output : "); print_vector(n, A);

    delete [] A;
}
////////////////////////////////////////////////////////////////////////////////

/*************** Implementation of the third stage (P2) ***********************/

///////////////
// CONSTANTS //
///////////////

// each thread block consists of N_THD number of threads
#define E_THD (7)
#define N_THD (1 << E_THD)
// the number of butterflies each thread does
#define E_BUT (0) 
#define N_BUT (1 << E_BUT)
// the total number of elements each block handles
#define E_BLK (E_THD + E_BUT + 1)
#define N_BLK (1 << E_BLK)

/**
 * After the second stage, we combine those base FFTs to the FFT of size n. 
 * A list of operations need to be performed:
 *
 * I_{2^{i-1}} @ DFT_2 @ I_{2^{k-i}} (I_{2^{i-1}} @ D_{2, 2^{k-i}})
 *
 * = I_{2^{i-1}} @ ( (DEF_2 @ I_{2^{k-i}}) D_{2, 2^{k-i}} )
 *
 * for i from k - \ell to 1. Hence, k - i is from ell to k - 1, where n = 2^k 
 * and m = 2^{\ell}.
 *
 * Each application of the above formula is a group of butterfly operations,
 * and each group of butterfly operations 
 *
 * ((DFT_2 @ I_{2^{k-i}}) D_{2, 2^{k-i}})
 *
 * operates on a subarray of length 2*q = 2^{k-i+1}. More precisely the task of 
 * those butterflies is equivalent to
 *
 * for (j = 0; j < q; ++j) {
 *     y[j] = x[j] + x[q+j] * wi^j;
 *     y[q+j] = x[j] - x[q+j] * wi^j;
 * }
 * 
 * where wi is a 2^{k-i+1}-th primitive root of unity, and wi^j = wn^{2^(i-1)*j},
 * for j from 0 to 2^{k-i} - 1 will be accessed for each i. 
 *
 * For example, let k = 5 then the access pattern is 
 * i = 4 : q = 2  : 1, w^8
 * i = 3 : q = 4  : 1, w^4, w^8, w^12,  
 * i = 2 : q = 8  : 1, w^2, w^4, w^6, w^8, w^10, w^12, w^14
 * i = 1 : q = 16 : 1, w, w^2, w^3, w^4, w^5, w^6, w^7, 
 *                  w^8, w^9, w^10, w^11, w^12, w^13, w^14, w^15
 *
 * In total, there are 2^{i-1} groups of butterflies; 
 * each group consists of q = 2^{k-i} butterfly operations and 
 * uses 2^{k-i+1} elements in total.
 *
 */

/**
 * Out of place parallel buttterfly operations
 * 
 * k : exponent of n = 2^k
 * q : the stride size 2^{k-i}
 * odata : array of length n
 * idata : array of length n
 * i  : the i-th step
 * w  : 2^{k-i+1}-th primitive root of unity modulo p
 * Wi : array of [1, w, w^2, ..., w^{q-1}]
 * p  : prime number
 *
 * This kernel implements the formula
 *
 * I_{2^(i-1)} @ ((DFT_2 @ I_{2^{k-i}}) D_{2, 2^{k-i}} )
 *
 * We assume that each thread block can process multiple groups,
 * and at least 1 group. That is, q = 2^{k-i} is small. 
 * The number of blocks required is n / N_BLK and the nnumber of groups 
 * each block handles is N_BLK / (2*q). Hence the preassumption
 * is N_BLK >= 2 * q or E_BLK > k - i.
 *
 * TESTED
 *
 */
__global__ void list_butterfly_kernel_a(sfixn k, sfixn i, sfixn *odata,
           const sfixn *idata, const sfixn *Wi, sfixn p, double pinv)
{
    // block index
    int bid = blockIdx.x + (blockIdx.y << 15);

    sfixn tid, gval, rval, w, offset, x1, x2, t;
    sfixn q = (1 << (k - i));

    #pragma unroll
    for (sfixn j = 0; j < N_BUT; ++j) {
        tid = (j << E_THD) + threadIdx.x;
        gval = tid >> (k - i);
        rval = tid & (q - 1);
        
        // input data x1 and x2
        offset = (bid << E_BLK) + (gval << (1 + k - i)) + rval;
        x1 = idata[offset];
        x2 = idata[offset + q];
        // w = wi^j
        w = Wi[rval]; 
        // y1 = x1 + x2 * w 
        // y2 = x1 - x2 * w;
        t = mul_mod(x2, w, p, pinv);           // t = x2 * w
        odata[offset] = add_mod(x1, t, p);     // first output
        odata[offset + q] = sub_mod(x1, t, p); // second output
    }
}

/**
 * Out of place parallel buttterfly operations
 * 
 * k  : exponent of n = 2^k
 * l  : exponent of m = 2^l
 * odata : array of length n
 * idata : array of length n
 * i  : the i-th step
 * w  : 2^{k-i+1}-th primitive root of unity modulo p
 * Wi : array of [1, w, w^2, ..., w^{q-1}]
 * p  : prime number
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
 * We assume that multiple thread block process a group of butterflies,
 * that is, q = 2^{k-i} is big.  The number of blocks required is n / N_BLK 
 * and the number of thread blocks needed for each group is 2 * q / N_BLK. 
 * Hence the preassumption is N_BLK  < 2 * q or E_BLK <= k - i.
 *  
 */
__global__ void list_butterfly_kernel_b(sfixn k, sfixn i, sfixn *odata,
              const sfixn *idata, const sfixn *Wi, sfixn p, double pinv)
{
    // block index
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn q = 1 << (k - i);
    // the number of thread blocks for each block
    sfixn blocks_per_group = q >> (E_BLK - 1); 
    // gval is the group number
    sfixn gval = bid >> (k - i + 1 - E_BLK);
    // rval is the part number inside a group
    sfixn rval = bid & (blocks_per_group - 1); 

    sfixn j, start, ind, w, x1, x2, tid;
    #pragma unroll
    for (j = 0; j < N_BUT; ++j) {
        tid = (j << E_THD) + threadIdx.x;
        // offset to idata
        start = gval << (1 + k - i);
        // the index increases by N_BLK / 2 for each thread block 
        ind = (rval << (E_BLK - 1)) + tid;
        w = Wi[ind];
        // input data x1 and x2
        x1 = idata[start + ind];
        x2 = idata[start + ind + q];
        // y1 = x1 + x2 * w 
        // y2 = x1 - x2 * w;
        sfixn t = mul_mod(x2, w, p, pinv);          // t = x2 * w
        odata[start + ind] = add_mod(x1, t, p);     // first output
        odata[start + ind + q] = sub_mod(x1, t, p); // second output
    }
}

/**
 * A : the data vector of length n = 2^k
 * k : exponent of n = 2^k
 * l : exponent of m = 2^l
 * W : array of primitive roots of unity 
 * p : prime number
 *
 */
void list_butterfly(sfixn k, sfixn l, sfixn *A, const sfixn *W, sfixn p)
{
    if (k <= l) return; 
    const sfixn n = ((sfixn)1 << k);
    // device array as work space
    sfixn *T = NULL;
    cudaMalloc((void**)&T, n*sizeof(sfixn));
    if (DEBUG) checkCudaError("Error found in list_butterfly, initial stage");
    
    int nThd = N_THD;
    int nb = (n >> E_BLK);
    dim3 nBlk(nb, 1, 1);
    int nb15 = nb >> 15;
    if (nb15) { nBlk.x = ((sfixn)1 << 15); nBlk.y = nb15; }

    double pinv = 1 / (double) p;
    for (sfixn i = k - l; i >= 1; --i) {
        // wi = wn^{2^{i-1}} is a 2^{k-i+1}-th primitive root of unity
        // if k - l - i is odd then data from W to A.
        // if k - l - i is even then data from A to W. 
        // During the first iteration, k - l - i = k - l - k + l = 0 is even
        // T will be initialized by A. Thus not need to initialize T. 
        // Wi = W + n * (1 - 1 / 2^(i-1))
        //    = W + n - n / 2^(i-1)
        const sfixn *Wi = W + n - (1 << (k - i + 1));
        if ((k - l - i) % 2) {
            if (E_BLK > k - i) {
                //printf("Calling kernel a, T->A, k = %d, i = %d\n", k, i);
                list_butterfly_kernel_a<<<nBlk, nThd>>>(k, i, A, T, Wi, p, pinv);
            } else {
                //printf("Calling kernel b, T->A, k = %d, i = %d\n", k, i);
                list_butterfly_kernel_b<<<nBlk, nThd>>>(k, i, A, T, Wi, p, pinv);
            }
        } else {
            if (E_BLK > k - i) {
                //printf("Calling kernel a, A->T, k = %d, i = %d\n", k, i);
                list_butterfly_kernel_a<<<nBlk, nThd>>>(k, i, T, A, Wi, p, pinv);
            } else {
                //printf("Calling kernel b, A->T, k = %d, i = %d\n", k, i);
                list_butterfly_kernel_b<<<nBlk, nThd>>>(k, i, T, A, Wi, p, pinv);
            }
        }
        cudaThreadSynchronize();
    }

    // if k - l is odd then move data from work space to A
    // if k - l is even then do nothing 
    if ((k - l) % 2) {
        cudaMemcpy(A, T, n*sizeof(sfixn), cudaMemcpyDeviceToDevice);
    }
    cudaFree(T);
    if (DEBUG) checkCudaError("Error found in list_butterfly, ending stage");
}

#undef E_THD
#undef N_THD
#undef E_BUT
#undef N_BUT
#undef E_BLK
#undef N_BLK
////////////////////////////////////////////////////////////////////////////////

/**
 * @ powers of primitive roots of unity
 *
 * X = [1, w, w^2, w^3, ..., w^{n-1}]
 *     [1, w^2, w^4, w^6, w^8, ...]
 *     [1, w^4, w^8, ...]
 *     [1, w^8,...]
 *     ...
 *     [1, w^{n/2}]
 *
 * For example, k = 3 and n = 8.
 *
 * X = [1, w, w^2, w^3, w^4, w^5, w^6, w^7,
 *      1, w^2, w^4, w^6,
 *      1, w^4];
 *
 *
 * In toal, the size of X is 2 * n - 2
 *
 */
__device__ __host__ __inline__ 
void setup_roots_of_unity(sfixn w, sfixn *X, sfixn n, sfixn p) {
    double pinv = 1 / (double)p;
    X[0] = 1; X[1] = w;
    for (sfixn i = 2; i < n; ++i) { X[i] = mul_mod(w, X[i-1], p, pinv); }

    sfixn *Y = X; sfixn *Z = X + n;
    for (sfixn m = n / 2; m >= 2; m >>= 1) {
        for (sfixn j = 0; j < m; ++j) { Z[j] = Y[2*j]; }
        Y = Z; Z += m;
    }
}

/**
 * Reorganize the layout of powers of primitive root of unity.
 *
 * Example: given w such that w^8 = -1
 *
 * [1, w, w^2, w^3, w^4, w^5, w^6, w^7]
 *
 * ===>
 *
 * [1, w, w^2, w^3, w^4, w^5, w^6, w^7, 1, w^2, w^4, w^6, 1, w^4]
 *
 * The expanded array has size 2 * n - 2. Or just use indices,
 * the output is
 *
 * 0   1   2   3   4    5   6   7   8   9   10   11   12   13   14   15,  e = 4
 * 
 * t0  t1  t2  t3  t4   t5   t6   t7  
 *  0   2   4   6   8   10   12   14,  i = 3,  tid % 1 == 0,  1 = 2^(e - i - 1)
 *
 * t0  t2  t4  t6 
 *  0   4   8  12,  i = 2, tid % 2 == 0
 *
 * t0  t4
 *  0   8,  i = 1, tid % 4 == 0
 *
 */
__global__ void
expand_roots_of_unity(sfixn n, sfixn e, sfixn *T, const sfixn *W) {
    sfixn tid = threadIdx.x;
    T[tid] = W[tid*2];
    T += n / 2;
    for (sfixn i = e - 2; i >= 1 ; --i) {
        sfixn m = 1 << (e - i - 1);
        sfixn r = tid & (m - 1);
        if (r == 0) { T[tid >> (e - i - 1)] = W[tid * 2]; }
        T += m;
    }
}

/**
 * Compute DFT in parallel using tensor product.
 *
 * n  : DFT size, length of A 
 * k  : exponent of n = 2^k
 * m  : base case size
 * l  : exponent of m = 2^l
 * A  : coefficient vector on device
 * wn : n-th primitive root of unity modulo p
 * p  : prime number
 *
 */
void tensor_ct_fft(sfixn n, sfixn k, sfixn *A, sfixn wn, sfixn p) {
    
    // precompute powers of primitive roots of unity 
    sfixn *W_d;
    sfixn *W_h = new sfixn[n]; 
    cudaMalloc((void **)&W_d, sizeof(sfixn)*n);
    setup_roots_of_unity(wn, W_h, n/2, p);
    cudaMemcpy(W_d, W_h, sizeof(sfixn)*n, cudaMemcpyHostToDevice);    

    // run three steps
    const int ell = 4;
    const int eThd = 6;

    data_shuffle(k, ell, A);   
    list_fft(k, A, wn, eThd, ell, p);
    list_butterfly(k, ell, A, W_d, p);

    // cleanup
    cudaFree(W_d);
    delete [] W_h;
}

void test_tensor_ct_fft() {
    //sfixn p = 3329; // 13 * 2^8 + 1
    sfixn p = 257; // 2^8 + 1
    sfixn k = 6;
    sfixn n = (1 << k);
    sfixn *A_h = new sfixn[n];
    sfixn *A_d;

    for (sfixn i = 0; i != n; ++i) A_h[i] = i;
    print_vector(n, A_h);

    cudaMalloc((void **)&A_d, sizeof(sfixn)*n);
    cudaMemcpy(A_d, A_h, sizeof(sfixn)*n, cudaMemcpyHostToDevice);
    sfixn wn = primitive_root(k, p);
    printf("wn = %d\n", wn);

    tensor_ct_fft(n, k, A_d, wn, p);

    cudaMemcpy(A_h, A_d, sizeof(sfixn)*n, cudaMemcpyDeviceToHost);
    print_vector(n, A_h);

    cudaFree(A_d);
    delete [] A_h;
}

void tensor_ct_fft_host(sfixn n, sfixn k, sfixn *A, sfixn wn, sfixn p) {
    sfixn *A_d;
    cudaMalloc((void **)&A_d, sizeof(sfixn)*n);
    cudaMemcpy(A_d, A, sizeof(sfixn) * n, cudaMemcpyHostToDevice);
    float elapsedTime;
    start_timer(0);

    tensor_ct_fft(n, k, A_d, wn, p);

    stop_timer(0, elapsedTime);
    printf("%2d\tno_transfer\t%8.3f\t", k, elapsedTime);
    cudaMemcpy(A, A_d, sizeof(sfixn) * n, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
}

/***********
 * TESTING *
 ***********/
void test_ct_fft(int argc, char** argv) {
    const sfixn p = 469762049;
    sfixn k = 26; 
    if (argc > 1) { k = atoi(argv[1]); }

    sfixn n = ((sfixn)1 << k);
    sfixn *X = new int[n];
    sfixn w = primitive_root(k, p);
    for (int i = 0; i < n; ++i) X[i] = i;

    float elapsedTime;
    start_timer(0);
    
    tensor_ct_fft_host(n, k, X, w, p);

    stop_timer(0, elapsedTime);
    printf("transfer\t%8.3f\n", elapsedTime);

    delete [] X;
}
