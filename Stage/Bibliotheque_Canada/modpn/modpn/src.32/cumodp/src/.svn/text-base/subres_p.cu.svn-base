///////////////////////////////////////////////////////////////////////////////
#include "../include/subres.h"

/**
 * The input size for subresultant computations is from tens to hundreds.
 * To utilize GPUs fully, each thread block has 128 threads. From the
 * experimentats, it is bad to let one thread compute a subresultant
 * chain independently, due to the unflavored data access pattern. 
 * Threads inside a thread block should be working cooperatively on
 * the same subresultant chain.
 *
 * With the similar sprit to bivariate FFTs, we need to further increase
 * the parallelism of the subresultant chain algorithm, aka parallelizing
 * pseudo-divisions. 
 *
 * For example, without any degenerations, the computation flow is the
 * following. Assuming the degrees of Fi and Gi are 4 and 3, respectively.
 *
 *  F0     F1     F2     F3
 *  G0     G1     G2     G3
 * S02    S12    S22    S32   =  prem( Fi, -Gi) 
 * S01    S11    S21    S31   ~  prem( Gi, -Si2)
 * S00    S10    S20    S30   ~  prem(Si2, -Si1)
 *
 * with deg(Sij) = j. No difficulty to handle.
 *
 * Degenerated case: (1) defective subresutants appear for all i:
 *
 *  F0     F1     F2     F3
 *  G0     G1     G2     G3
 *   0      0      0      0  
 * S01    S11    S21    S31   =  prem(Fi, -Gi) 
 * S00    S10    S20    S30   ~  prem(Gi, -Si1)
 *
 * with deg(Sij) = j. No difficulty to handle.
 *
 * Degenerated case: (2) defective subresutants appear NOT for all i:
 *
 *  F0     F1     F2     F3
 *  G0     G1     G2     G3
 * S02    S12      0    S32  = prem(Fi, -Gi) for i = 0, 1, 3.
 * S01    S11    S21    S31  ~ prem(Gi, -Si2) for i = 0, 1, 3. 
 * S00    S10    S20    S30  ~ prem(Si2, -Si1) for i = 0, 1, 3.
 * 
 * with prem(F2, -G2) = S21, and prem(G2, -S21) ~ S20.
 *
 * How to handle? 
 *
 * In this file, our first implementation of computing subresultants by 
 * evalutation and specialization requires that the shape of subresultant
 * chains is uniform across all the images. The data layout will be 
 * transposed comparing with the usual one. 
 *
 * For example, for the case (1) above, the subresultant chain is
 *
 * |  0   0   0   0  | size is 3B
 * | S01 S11 S21 S31 | size is 2B 
 * | S00 S10 S20 S30 | size is  B
 *
 *               Total size is 6B
 */
///////////////////////////////////////////////////////////////////////////////

/**
 * Given a device array of integers, compute the index of first nonzero
 * entry in the array, from right to left. 
 *
 * For example, opearting on the array
 *
 *  0  1  2  3  4  5  6  index_ptr
 * [0, 5, 2, 4, 0, 0, 0] -1 
 *
 * gets the index 3.
 *
 * Initial value of ind_ptr is -1.
 */
template <sfixn nthds>
__global__ void 
right_nonzero_ind_ker(const sfixn *X, sfixn n, sfixn* ind_ptr) {
    sfixn tid = blockIdx.x * nthds + threadIdx.x;    
    if ((tid < n) && (X[tid] != 0)) { atomicMax(ind_ptr, tid); }
}

void right_nonzero_ind_dev(const sfixn *X, sfixn n, sfixn* ind_ptr) {
    const sfixn nthds = 128;
    sfixn nblks = (n / nthds) + ((n % nthds) ? 1 : 0);
    right_nonzero_ind_ker<nthds><<<nblks, nthds>>>(X, n, ind_ptr);
}

/**
 * Given an integer array of length n, compute the index of the rightmost 
 * nonzero entry, using multiple thread blocks.
 */
template <sfixn nthds>
__global__ void 
right_nonzero_ind_blk_ker(const sfixn *X, sfixn n, sfixn* ind_ptr) 
{
    __shared__ sfixn bind;

    if (threadIdx.x == 0) bind = -1;
    __syncthreads();

    sfixn tid = blockIdx.x * nthds + threadIdx.x;
    if ((tid < n) && (X[tid] != 0)) { atomicMax(&bind, tid); }
    __syncthreads();

    atomicMax(ind_ptr, bind);
}

void right_nonzero_ind_blk_dev(const sfixn *X, sfixn n, sfixn* ind_ptr)
{
    const sfixn nthds = 128;
    sfixn nblks = (n / nthds) + ((n % nthds) ? 1 : 0);
    right_nonzero_ind_blk_ker<nthds><<<nblks, nthds>>>(X, n, ind_ptr);
    cudaThreadSynchronize();
}

/**
 * Set a plain array with value val.
 *
 * @val, value to be set for each slot of X
 * @X, the array to be set
 * @n, the number of slots
 *
 * Assumption: if n is large, then it has a factor of 2^10. 
 *
 */
__global__ void set_array_ker(sfixn val, sfixn n, sfixn *X) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = threadIdx.x + bid * blockDim.x;
    if (tid < n) X[tid] = val;
}

void set_array_dev(sfixn val, sfixn n, sfixn *X) 
{
    const sfixn nthds = 128;
    sfixn nb = (n / nthds) + ((n % nthds) ? 1 : 0);

    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = (sfixn(1) << 10);
        nBlks.y = (nb >> 10);
    }
    set_array_ker<<<nBlks, nthds>>>(val, n, X);
    cudaThreadSynchronize();
}

/**
 * Compute the degrees of a list of univariate polynomials.
 *
 * @B, the size of the list, (assume B is a power of 2) 
 * @n, the size of each polynomial
 * @X, the list of polynomials, its size is B * n
 * @degs, (output) the degrees, its size is B
 *
 * Array [a0, a1, a2, .... ] represents the polynomial a0 + a1 * x + ....
 *
 * For example, let B = 4, n = 4, and 
 *
 * X = 1 3 1 0 | 2 3 3 0 | 3 3 4 1 | 4 3 5 0 
 *
 * Then the degrees are 
 *
 * degs = [2, 2, 3, 2].
 *
 * The total number of threads is n * B.
 *
 * The default value of degs[i] is -1 for each entry, 
 * zero polynomial has degree -1, and const polynomials
 * has degree 0.
 */

__global__ void list_deg_coarse_ker(sfixn *degs, sfixn n, const sfixn *X) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    const sfixn *T = X + tid * n;
    for (sfixn i = n - 1; i >= 0; --i) {
        if (T[i] != 0) { degs[tid] = i; return; }
    }
    degs[tid] = -1;
}

void list_deg_coarse_dev(sfixn B, sfixn *degs, sfixn n, const sfixn *X)
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));

    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = (sfixn(1) << 10);
        nBlks.y = (nb >> 10);
    }

    list_deg_coarse_ker<<<nBlks, nthds>>>(degs, n, X);
    cudaThreadSynchronize();
}

__global__ void list_deg_fine_ker(sfixn *degs, sfixn n, const sfixn *X) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    sfixn qtid = tid / n;
    sfixn rtid = tid % n;
    if (X[tid] != 0) { atomicMax(degs + qtid, rtid); }
}

/* B is the number of images, power of 2. */
void list_deg_fine_dev(sfixn B, sfixn *degs, sfixn n, const sfixn *X) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = n * B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));

    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = (sfixn(1) << 10);
        nBlks.y = (nb >> 10);
    }
    list_deg_fine_ker<<<nBlks, nthds>>>(degs, n, X);
    cudaThreadSynchronize();
}

/**
 * When n is small, coarse-grained list degree computations is better
 * than fine-grained one. 
 *
 * By default, the coarse version will be used.
 */
inline void list_deg_dev(sfixn B, sfixn *degs, sfixn n, const sfixn *X) {
    list_deg_coarse_dev(B, degs, n, X);
}

/**
 * Check if an array consists of the same value. 
 * 
 * @res, the address for the return value
 * @X, the array to be checked
 * @B, the size of the array 
 *
 * Assumption: B is a power of 2, at least 128.
 */
__global__ void has_same_val_ker(sfixn *res, sfixn B, const sfixn *X)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    sfixn v = X[0];
    if (X[tid] != v) atomicExch(res, 0);
}

void has_same_val_dev(sfixn *res, sfixn B, const sfixn *X) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));

    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 15) == 0) {
        nBlks.x = (sfixn(1) << 15);
        nBlks.y = (nb >> 15);
    }
    has_same_val_ker<<<nBlks, nthds>>>(res, B, X);
    cudaThreadSynchronize();
}

////////////////////////////////////////////////////////////////////////////////
// The following is used to compute prem(F, -G, x) in parallel.
//
// Given F = sum(ai * x^i, i = 0..m) and G = sum(bi * x^i, i=0..n) with m >= n,
// prem(F, -G, x) can be computed by a sequence of cancellations.
//
// Let
//
// F := a3 * x^3 + a2 * x^2 + a1 * x + a0 and G := b2 * x^2 + b1 * x + b0,
//
// then
//
// (1) H2 := -b2 * F + a3 * x * G = c2 * x^2 + c1 * x + c0;
//
// (2) H1 := -b2 * H2 + c2 * G = d1 * x + b0,
//
// where
//
// c2 = | a3 a2 |, c1 = | a3 a1 | and c0 = | a3 a0 |  (degree gap = 1)
//      | b2 b1 |       | b2 b0 |          | b2  0 |
//
// d1 = | c2 c1 |, and d0 = | c2 c0 |, (degree gap = 0)  
//      | b2 b1 |           | b2 b0 |
//
// Example, given
//
// F = [0, 1, 2, 3, 4, 5, 6, 0] ===> f = x+2*x^2+3*x^3+4*x^4+5*x^5+6*x^6
//
// G = [1, 2, 3, 4, 5, 0, 0, 0] ===> g = 1+2*x+3*x^2+4*x^3+5*x^4
//
// (1) degree gap = 2,  H5 = -5*F + 6*x^2*G
//
// H5 = -5*x-4*x^2-3*x^3-2*x^4-x^5
// 
// (2) degree gap = 1, 
//
// H4 = (-5) * h5 + (-1) * x * G = 24*x+18*x^2+12*x^3+6*x^4
//
// (3) degree gap = 0,
//
// H3 = (-5) * H4 + 6 * G = -108*x-72*x^2-36*x^3+6
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @B,  the number of segments 
 * @LH, the starting address of H[0]
 * @LF, the starting address of F[0]
 * @LG, the starting address of G[0]
 * @dF, the degree of F[i] 
 * @dG, the degree of G[i]
 * @p,  prime number
 *
 * The data layout is the following
 *
 * LF => F[0] F[1] F[2] F[3] ... F[B-1]
 * LG => G[0] G[1] G[2] G[3] ... G[B-1]
 * LH => H[0] H[1] H[2] H[3] ... H[B-1]
 * 
 * The length of F[i] is dF + 1, the length of G[i] is dG + 1 and the length
 * of H[i] is dF.
 *
 * For example, given
 *
 * F0 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4,  
 * G0 = c0 + c1*x + c2*x^2,
 * F1 = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4,
 * G1 = d0 + d1*x + d2*x^2,
 *
 * with c2 * d2 != 0.
 *
 * This function computes 
 *
 * H0 = c2 * F0 - a4 * x^2 * G0,  
 *    = c2 * a0 + c2 * a1 * x + c2 * a2 * x^2 + c2 * a3 * x^3
 *                            - a4 * c0 * x^2 - a4 * c1 * x^3
 *
 *    = |a0 a4| + |a1 a4| * x + |a2  a4| * x^2 + |a3 a4| * x^3
 *      |0  c2|   |0  c2|       |c0  c2|         |c1 c2| 
 * 
 *      THD 0      THD 1         THD 2            THD 3   
 *
 * H1 = d2 * F1 - b4 * x * G1. 
 *
 *    = |b0 b4| + |b1 b4| * x + |b2  b4| * x^2 + |b3 b4| * x^3
 *      |0  d2|   |0  d2|       |d0  d2|         |d1 d2| 
 *
 *      THD 4      THD 5         THD 6            THD 7   
 *
 * The total number of cancellations is dF * B, which is also the total number
 * of threads used in this kernel. Let tid be the thread index and let
 *
 * qtid = tid / dF and rtid = tid % dF.
 *
 * Then qtid tells which polynomial it works on and rtid tells which 
 * cancellations it works on.
 */

__global__ void 
list_poly_reduce_ker(sfixn *LH, sfixn dF, const sfixn *LF, 
    sfixn dG, const sfixn *LG, const sfixn p, double pinv)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;

    sfixn qtid = tid / dF;
    sfixn rtid = tid % dF;
    
    const sfixn *F = LF + qtid * (dF + 1);
    const sfixn *G = LG + qtid * (dG + 1);
    sfixn *H = LH + qtid * dF;

    sfixn dgap = dF - dG;
    sfixn a = F[dF]; // a is the leading coefficient of F
    sfixn b = G[dG]; // b is the leading coefficient of G, nonzero
    sfixn u = F[rtid];
    sfixn v = ((rtid >= dgap) ? G[rtid - dgap] : 0);
    // The configuration is the following
    //         u ...... a
    //   v ...... b  
    // where a is the current leading coefficient to be eliminated
    // (possibly 0), b is the current leading coefficient (nonzero),
    // u and v are cofficients to be adjusted. For each pair
    // (u, v), compute a * v - u * b mod p and store it to H[rtid];
    //sfixn t1 = mul_mod(a, v, p, pinv);
    //sfixn t2 = mul_mod(b, u, p, pinv);
    sfixn t1 = mul_mod(a, v, p);
    sfixn t2 = mul_mod(b, u, p);
    H[rtid] = sub_mod(t1, t2, p);
}

/** 
 * Assumptions: 
 * - B is the number of images, power of 2,
 * - dF cannot be huge, that is,
 *  
 * if nb = dF * B / nthds > 2^16 then, 2^10 should be a factor of nb. 
 *
 **/
void list_poly_reduce_dev(sfixn B, sfixn *LH, sfixn dF, const sfixn *LF,
    sfixn dG, const sfixn *LG, const sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = dF * B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));

    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = (sfixn(1) << 10);
        nBlks.y = (nb >> 10);
    }
    list_poly_reduce_ker<<<nBlks, nthds>>>(LH, dF, LF, dG, LG, p, pinv);
    cudaThreadSynchronize();
}

/**
 * Extended version of list_poly_reduce_dev, in which zeros may be padded
 * in each slot of LG. 
 *
 * @szG, the size of each LG[i] and dG + 1 <= szG. 
 *
 * For example, B = 2, dG = 3, szG = 5
 *
 * LG = [0 1 2 3 0 0 
 *       4 5 6 7 0 0]
 *
 */
__global__ void 
list_poly_reduce_defective_ker(sfixn *LH, sfixn dF, const sfixn *LF,
    sfixn szG, sfixn dG, const sfixn *LG, const sfixn p, double pinv)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;

    sfixn qtid = tid / dF;
    sfixn rtid = tid % dF;
    
    const sfixn *F = LF + qtid * (dF + 1);
    const sfixn *G = LG + qtid * szG;
    sfixn *H = LH + qtid * dF;
    
    sfixn dgap = dF - dG;
    sfixn a = F[dF]; // a is the leading coefficient of F
    sfixn b = G[dG]; // b is the leading coefficient of G, nonzero
    sfixn u = F[rtid];
    sfixn v = ((rtid >= dgap) ? G[rtid - dgap] : 0);
    // The configuration is the following
    //         u ...... a
    //   v ...... b  
    // where a is the current leading coefficient to be eliminated
    // (possibly 0), b is the current leading coefficient (nonzero),
    // u and v are cofficients to be adjusted. For each pair
    // (u, v), compute a * v - u * b mod p and store it to H[rtid];
    //sfixn t1 = mul_mod(a, v, p, pinv);
    //sfixn t2 = mul_mod(b, u, p, pinv);
    sfixn t1 = mul_mod(a, v, p);
    sfixn t2 = mul_mod(b, u, p);
    H[rtid] = sub_mod(t1, t2, p);
}

/**
 * Assumptions: 
 * - B is the number of images, power of 2,
 * - dF cannot be huge, that is,
 *  
 * if nb = dF * B / nthds > 2^16 then, 2^10 should be a factor of nb. 
 */
void list_poly_reduce_defective_dev(sfixn B, sfixn *LH, sfixn dF,
    const sfixn *LF, sfixn szG, sfixn dG, const sfixn *LG, 
    const sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = dF * B / nthds;

    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));
    if (DEBUG) assert(szG >= dG + 1);

    dim3 nBlks(nb, 1, 1);

    if (rem2e(nb, 10) == 0) {
        nBlks.x = (sfixn(1) << 10);
        nBlks.y = (nb >> 10);
    }

    list_poly_reduce_defective_ker<<<nBlks, nthds>>>
        (LH, dF, LF, szG, dG, LG, p, pinv);
    cudaThreadSynchronize();
}

/**
 * Compute H[i] = prem(F[i], -G[i], y) for i from 0 to B - 1.
 *
 * @B, the size of the list
 * @LH, the result of length B * dG
 * @LF, list of univariate polynomials
 * @dF, the common degree of polynomials in LF
 * @LF, list of univariate polynomials
 * @dG, the common TRUE degree of polynomials in LG 
 * @LW, the work space of size at least 2 * B * dF.
 *
 * Work space LW serves as a double buffer.
 *
 */
void list_neg_prem_dev(sfixn B, sfixn *LH, sfixn dF, const sfixn *LF, 
    sfixn dG, const sfixn *LG, sfixn p, double pinv, sfixn *LW)
{
    // double buffer method
    sfixn *LX = LW, *LY = LW + B * dF, *T; 

    // reduce once and store the result into LX
    list_poly_reduce_dev(B, LX, dF, LF, dG, LG, p, pinv);
    for (sfixn d = dF - 1; d >= dG; --d) {
        // LX --> LY
        list_poly_reduce_dev(B, LY, d, LX, dG, LG, p, pinv);
        // switch the role of LX and LY
        T = LX, LX = LY, LY = T;
    }
    cudaMemcpy(LH, LX, sizeof(sfixn) * B * dG, cudaMemcpyDeviceToDevice);
    cudaThreadSynchronize();
}

/* The version without passing work space */
void list_neg_prem_dev(sfixn B, sfixn *LH, sfixn dF, const sfixn *LF, 
    sfixn dG, const sfixn *LG, sfixn p, double pinv)
{
    sfixn *LW;
    cudaMalloc((void**)&LW, dF*B*2*sizeof(sfixn));

    // double buffer method
    sfixn *LX = LW, *LY = LW + B * dF, *T; 

    // reduce once and store the result into LX
    list_poly_reduce_dev(B, LX, dF, LF, dG, LG, p, pinv);
    for (sfixn d = dF - 1; d >= dG; --d) {
        // LX --> LY
        list_poly_reduce_dev(B, LY, d, LX, dG, LG, p, pinv);
        // switch the role of LX and LY
        // effective data is in LX again
        T = LX, LX = LY, LY = T;
    }
    cudaMemcpy(LH, LX, sizeof(sfixn) * B * dG, cudaMemcpyDeviceToDevice);
    cudaFree(LW);
    cudaThreadSynchronize();
}

/**
 * Compute the inverse of the leading coefficients in a polynomial list. 
 *
 * @LA, a list of univariate polynomials of degree d (true degree), each poly
 *      is stored in a vector of length n >= d + 1. The size is n * B.
 * @LC, (output) the list of inversed leading coefficients of LA
 * @B,  the length of LC
 * @d,  the common degree of polynomials in LA
 *
 * For example, n = 4, d = 2, B = 4, and
 *
 * LA = [ 0 1 0 0, 2 3 0 0, 4 5 0 0, 6 7 0 0 ]
 * LC = [ 1/1 1/3 1/5 1/7 ] mod p
 *
 * The total number of threads is B.
 *
 */
__global__ void 
list_inv_lcoeff_ker(sfixn n, sfixn d, sfixn *LC, const sfixn *LA,
    sfixn p, double pinv) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn tid = threadIdx.x + bid * blockDim.x;
    sfixn a = LA[tid * n + d];
    // LC[tid] = inv_mod(a, p, pinv);
    LC[tid] = inv_mod(a, p);
}

void list_inv_lcoeff_dev(sfixn B, sfixn n, sfixn d, sfixn *LC, 
    const sfixn *LA, sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = B / nthds;
    if (DEBUG) assert(nb >= 1);
    
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 15) == 0) {
        nBlks.x = ((sfixn)1 << 15);
        nBlks.y = (nb >> 15);
    }

    list_inv_lcoeff_ker<<<nBlks, nthds>>>(n, d, LC, LA, p, pinv);
    cudaThreadSynchronize();
}

/**
 * For each 0 <= i < B, compute 
 *
 * LC[i] = LB[i] * (lcoeff(LB[i])/lcoeff(LA[i])^n)^(d - e - 1)
 *
 * @B, the number of images, power of 2
 * @n, the power to be rasied for lcoeff(LA[i])
 * @LA, the  first list of subresultants, the size is B * (d + 1) 
 * @LB, the second list of subresultants, the size is B * d
 * @LC, the output list of subresultants, the size is B * (e + 1)
 * @d, subresultant index
 * @e, subresultant index
 *
 * The total number of threads is B * (e + 1)
 * 
 * For example, B = 1, n = 1, d = 5, e = 2, 
 *
 * LA : a0 a1 a2 a3 a4 a5 
 *
 * LB : b0 b1 b2  0  0
 *
 * LC : c0 c1 c2
 * -------------------------------------
 *
 * c[i] = (b2 / a5)^2 * b[i]
 */

__global__ void 
list_def_to_reg_subres_ker(sfixn n, sfixn *LC, const sfixn *LB,
    const sfixn *LA, sfixn e, sfixn d, sfixn p, double pinv)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    sfixn qtid = tid / (e + 1);
    sfixn rtid = tid % (e + 1);
    
    // ai = (lcoeff(LA[i]))^n 
    // bi = lcoeff(LB[i]) 
    // ui = (bi/ai)^(d - e - 1)
    //sfixn ai = pow_mod(LA[qtid * (d + 1) + d], n, p, pinv);
    sfixn ai = pow_mod(LA[qtid * (d + 1) + d], n, p);
    sfixn bi = LB[qtid * d + e];  
    //sfixn ui = pow_mod(quo_mod(bi, ai, p, pinv), d - e - 1, p, pinv);
    sfixn ui = pow_mod(quo_mod(bi, ai, p), d - e - 1, p);
    //LC[tid] = mul_mod(LB[qtid * d + rtid], ui, p, pinv);
    LC[tid] = mul_mod(LB[qtid * d + rtid], ui, p);
}

void list_def_to_reg_subres_dev(sfixn n, sfixn B, sfixn *LC, const sfixn *LB, 
    const sfixn *LA, sfixn e, sfixn d, sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = (e + 1) * B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));
    if (DEBUG) assert((n >= 0) && (d - e > 1));
    
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = ((sfixn)1 << 10);
        nBlks.y = (nb >> 10);
    }

    list_def_to_reg_subres_ker<<<nBlks, nthds>>>(n, LC, LB, LA, e, d, p, pinv);
    cudaThreadSynchronize();
}

/**
 * Compute LC = LX / lcoeff(LA)^m  
 *
 * The size of LC is B * e;
 * the size of LX is B * e;
 * the size of LA is B * (d + 1).
 */
__global__ void 
list_next_subres_scale_ker(sfixn m, sfixn *LC, sfixn e, const sfixn *LX, 
    sfixn d, const sfixn *LA, sfixn p, double pinv) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    sfixn qtid = tid / e;

    sfixn a = LA[qtid * (d + 1) + d];
    //a = pow_mod(a, m, p, pinv);
    a = pow_mod(a, m, p);
    //LC[tid] = quo_mod(LX[tid], a, p, pinv);
    LC[tid] = quo_mod(LX[tid], a, p);
}

void list_next_subres_scale_dev(sfixn n, sfixn B, sfixn *LC, sfixn e,
    const sfixn *LX, sfixn d, const sfixn *LA, sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = e * B / nthds;
    if (DEBUG) assert(nb >= 1);
    sfixn m = n * (d - e) + 1;
    
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = ((sfixn)1 << 10);
        nBlks.y = (nb >> 10);
    }

    list_next_subres_scale_ker<<<nBlks, nthds>>>(m, LC, e, LX, d, LA, p, pinv);
    cudaThreadSynchronize();
}

/**
 * The second implementation of computing a list of subresultant chains. 
 * The bottleneck is list_next_subres_scale_dev, which accesses the leading 
 * coefficient of LA in a jumpped manner. The new implementation tries to 
 * alleviate this by precomputing coeff(LA)^m.
 *
 * WP, Tue Oct 19 16:21:31 EDT 2010
 *
 * The second bottleneck is list_def_to_reg_subres_dev, 
 * which accesses the leading coefficient of LA[i] in a jumpped manner. 
 *
 * WP, Tue Oct 19 18:28:48 EDT 2010
 **/

/**
 * Compute the leading coefficient of a list polynomials.
 *
 * LCA[i] = lcoeff(LA[i]) for i = 0 .. B - 1.
 *
 * @LCA, the list coefficient of length B
 * @d, the common degree of polynomials in LA
 */
__global__ void 
list_lcoeff_ker(sfixn *LCA, sfixn d, const sfixn *LA) {
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    LCA[tid] = LA[tid * (d + 1) + d];
}

/**
 * Raise to a negative power for each element in an array.
 *
 * LCM[i] = 1 / LCM[i]^m
 *
 * @LCM, a list of length B
 * @m, positive integer
 */
__global__ void
list_inv_power_ker(sfixn m, sfixn *LCM, sfixn p, double pinv) {
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    sfixn inv_a = inv_mod(LCM[tid], p);
    //LCM[tid] = pow_mod(inv_a, m, p, pinv);
    LCM[tid] = pow_mod(inv_a, m, p);
}

/**
 * Compute a negative power for each leading coefficient in a list of 
 * polynomials of degree d:
 *
 * LCM[i] = lcoeff(LA[i])^{-m}.
 *
 * @B, the list size
 * @LCM, the list coefficient of length B (output)
 * @d, the common degree of polynomials in LA
 *
 */
void list_inv_pow_lcoeff_dev(sfixn m, sfixn B, sfixn *LCM, sfixn d, 
    const sfixn *LA, sfixn p) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = B / nthds;
    if (DEBUG) assert(nb >= 1);
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = ((sfixn)1 << 10);
        nBlks.y = (nb >> 10);
    }

    // store the leading coefficient into LCM
    list_lcoeff_ker<<<nBlks, nthds>>>(LCM, d, LA);
    cudaThreadSynchronize();

    // raise to a negative power for each leading coefficient
    if (DEBUG) assert(m >= 0);

    double pinv = 1 / double(p);
    list_inv_power_ker<<<nBlks, nthds>>>(m, LCM, p, pinv);
    cudaThreadSynchronize();

    if (DEBUG) checkCudaError("list_inv_pow_lcoeff_dev");
}

/**
 * Compute LC[i] = LX[i] / LCM[i] 
 *
 * LCM[i] = 1 / lcoeff(LA[i])^m 
 * 
 * @m, the power to raise 
 * @e, the common size of polynomials in LC[i] and LX[i]
 * @LC, a list of polynomials (output)
 * @LX, a list of polynomials
 * @LCM, a list of preprocessed leading cofficients of LA
 *
 */
__global__ void 
list_next_subres_scale2_ker(sfixn *LC, sfixn e, const sfixn *LX, 
    const sfixn *LCM, sfixn p, double pinv) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    // qtid indicates which polynomial this thread is working on
    sfixn qtid = tid / e; 
    sfixn a = LCM[qtid];
    //LC[tid] = mul_mod(LX[tid], a, p, pinv);
    LC[tid] = mul_mod(LX[tid], a, p);
}

/**
 * Compute LC = LX / lcoeff(LA)^m, where m = n * (d - e) + 1  
 *
 * The size of LC is B * e;
 * the size of LX is B * e;
 * the size of LA is B * (d + 1).
 */
void list_next_subres_scale2_dev(sfixn n, sfixn B, sfixn *LC, sfixn e,
    const sfixn *LX, sfixn d, const sfixn *LA, sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = e * B / nthds;
    if (DEBUG) assert(nb >= 1);
    sfixn m = n * (d - e) + 1;
    
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = ((sfixn)1 << 10);
        nBlks.y = (nb >> 10);
    }

    // precompute LCM from LA
    sfixn *LCM;
    cudaMalloc((void**)&LCM, sizeof(sfixn)*B);
    list_inv_pow_lcoeff_dev(m, B, LCM, d, LA, p); 

    // scaling
    list_next_subres_scale2_ker<<<nBlks, nthds>>>(LC, e, LX, LCM, p, pinv);
    cudaThreadSynchronize();
    
    cudaFree(LCM);
    if (DEBUG) checkCudaError("list_next_subres_scale2_dev");
}

/**
 * For each 0 <= i < B, compute 
 *
 * LC[i] = LB[i] * LCM[i] with
 *
 * LCM[i] = (lcoeff(LB[i])/lcoeff(LA[i])^n)^(d - e - 1)
 *
 * @B, the number of images, power of 2
 * @n, the power to be rasied for lcoeff(LA[i])
 * @LA, the  first list of subresultants, the size is B * (d + 1) 
 * @LB, the second list of subresultants, the size is B * d
 * @LC, the output list of subresultants, the size is B * (e + 1)
 * @LCM, the precomputed data 
 * @d, subresultant index
 * @e, subresultant index
 *
 * The total number of threads is B * (e + 1)
 * 
 * For example, B = 1, n = 1, d = 5, e = 2, 
 *
 * LA : a0 a1 a2 a3 a4 a5 
 *
 * LB : b0 b1 b2  0  0
 *
 * LC : c0 c1 c2
 *
 * LCM : (b2 / a5)^2
 * -------------------------------------
 *
 * c[i] = (b2 / a5)^2 * b[i]
 */

/**
 * Compute LCM[i] = (lc(LB[i]) * LCM[i])^(d - e - 1)
 * 
 * The size of LB[i] is d, but the degree of LB[i] is e.
 *
 * The total number of threads is B.
 */
__global__ void
list_mul_pow_ker(sfixn *LCM, sfixn d, sfixn e, const sfixn *LB,
    sfixn p, double pinv) 
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    sfixn ai = LCM[tid];
    sfixn bi = LB[tid * d + e];  
    sfixn ui = mul_mod(bi, ai, p, pinv);
    LCM[tid] = pow_mod(ui, d - e - 1, p);
}

void list_mul_pow_dev(sfixn B, sfixn *LCM, sfixn d, sfixn e, const sfixn *LB,
    sfixn p, double pinv) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = B / nthds;
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = ((sfixn)1 << 10);
        nBlks.y = (nb >> 10);
    }
    list_mul_pow_ker<<<nBlks, nthds>>>(LCM, d, e, LB, p, pinv);
    cudaThreadSynchronize();
}

__global__ void 
list_def_to_reg_subres2_ker(sfixn *LC, const sfixn *LB, const sfixn *LCM, 
    sfixn e, sfixn d, sfixn p, double pinv)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 10);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    // qtid indicates which polynomial the thread is working on
    // rtid indicates which coefficient the thread is working on
    sfixn qtid = tid / (e + 1);
    sfixn rtid = tid % (e + 1);
    
    //LC[tid] = mul_mod(LB[qtid * d + rtid], LCM[qtid], p, pinv);
    LC[tid] = mul_mod(LB[qtid * d + rtid], LCM[qtid], p);
}

void list_def_to_reg_subres2_dev(sfixn n, sfixn B, sfixn *LC, const sfixn *LB,
    const sfixn *LA, sfixn e, sfixn d, sfixn p, double pinv) 
{
    if (DEBUG) assert((n >= 0) && (d - e > 1));

    sfixn *LCM;
    cudaMalloc((void**)&LCM, B*sizeof(sfixn));

    // compute LCM[i] = lc(LA[i])^{-n}
    list_inv_pow_lcoeff_dev(n, B, LCM, d, LA, p);

    // compute LCM[i] = (lc(LB[i])/lc(LA[i]^n))^{d-e-1}
    list_mul_pow_dev(B, LCM, d, e, LB, p, pinv);

#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = (e + 1) * B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));
    
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 10) == 0) {
        nBlks.x = ((sfixn)1 << 10);
        nBlks.y = (nb >> 10);
    }

    list_def_to_reg_subres2_ker<<<nBlks, nthds>>>(LC, LB, LCM, e, d, p, pinv);
    cudaThreadSynchronize();

    cudaFree(LCM);
}

/********************** END OF OPTIMIZATION ***********************************/

/**
 * Compute LC = prem(LA, -LB, x) / lcoeff(LA)^(n*(d-e) + 1)  
 *
 * @LA, list of subres,        size B*(d+1), S_d
 * @LB, list of subres,        size B*d,     S_{d-1}, degree(S_{d-1}) = e
 * @LC, list of output subres, size B*e,     S_{e-1}
 * @LW, the workspace of size B * (2 * d + 2), double buffer
 *
 * Note that n = dgP - dgQ for the first iteration and n = 1 for all others.
 */
void list_next_subres_dev(sfixn n, sfixn B, sfixn *LC, sfixn d, const sfixn *LA,
    sfixn e, const sfixn *LB, sfixn p, double pinv, sfixn *LW) 
{
    // double buffer method
    sfixn *LX = LW, *LY = LW + B * d, *T; 
    // reduce once and store the result into LX
    list_poly_reduce_defective_dev(B, LX, d, LA, d, e, LB, p, pinv);

    for (sfixn i = d - 1; i >= e; --i) {
        // LX --> LY
        list_poly_reduce_defective_dev(B, LY, i, LX, d, e, LB, p, pinv);
        // switch the role of LX and LY
        T = LX, LX = LY, LY = T;
    }
    //list_next_subres_scale_dev(n, B, LC, e, LX, d, LA, p, pinv);
    list_next_subres_scale2_dev(n, B, LC, e, LX, d, LA, p, pinv);
    cudaThreadSynchronize();
    if (DEBUG) checkCudaError("list_next_subres_dev");
}

/**
 * Compute the subresultant chains of two lists of polynomials, with
 * the assumption that degree sequences are uniform across the list.
 * If this condition fails then returns CUMODP_SUBRES_NON_REGULAR
 * otherwise returns CUMODP_SUCCESS and fills the output S.
 *
 * @S,   subresultant chain data
 * @Bd,  the total number of evaluation images, power of 2
 * @dgP, the degree of P in y
 * @dgQ, the degree of Q in y, dgP >= dgQ
 * @p,   prime number
 *
 * The length of S is Bd * (1 + dgQ) * dgQ / 2. 
 **/
cumodp_err fine_subres_uni_dev(sfixn *S, sfixn Bd, sfixn dgP, const sfixn *LP,
    sfixn dgQ, const sfixn *LQ, sfixn p) 
{
    /////////////////////
    //   Settings
    /////////////////////
    sfixn Ssz = Bd * dgQ * (dgQ + 1) / 2;
    double pinv = 1 / (double)p;
    
    // allocate extra work spaces 
    sfixn *buffer;
    sfixn nbuff = 2 * Bd * dgP + Bd + 1;
    // printf("Bd = 2^%d, nbuff = %-8.3f(MB)\n", ceiling_log2(Bd), 
    //    sizeof(sfixn) * double(nbuff) / (1L << 20));
    cudaError_t err = cudaMalloc((void **)&buffer, nbuff*sizeof(sfixn));
    if (err != cudaSuccess) {
        if (DEBUG) fprintf(stderr, "fine_subres, out of global memory\n");
        return CUMODP_OUT_OF_MEMORY;
    }

    // double buffer for pseudo-divisions, size = 2 * dgQ * Bd
    sfixn *LW = buffer;  
    // degree-vector, size Bd + 1
    // the last element ldeg[Bd] is a flag indicating 
    // if degrees are different.
    // 0 means not the same, otherwise the same. 
    sfixn *ldeg = buffer + 2 * Bd * dgP; 
    
    ////////////////////////////////////////////////
    // Compute the first subresultant: S_{dgQ - 1}
    ////////////////////////////////////////////////
    sfixn *LB = S;
    list_neg_prem_dev(Bd, LB, dgP, LP, dgQ, LQ, p, pinv, LW);
    
    set_array_dev(-1, Bd + 1, ldeg);
    list_deg_dev(Bd, ldeg, dgQ, LB);
    has_same_val_dev(ldeg + Bd, Bd, ldeg);

    //printDeviceArray(ldeg, Bd+1);
    if (0 == get_dev_val(ldeg + Bd)) {
        cudaFree(buffer);
        // clean corrupted data
        reset_vector_dev(Ssz, S);
        return CUMODP_SUBRES_NON_REGULAR;
    }

    // e = 0, then all remainders are nonzero contants.
    // e = -1, then all remainders are zero.
    // e > 0, the common degree of the remainders.
    sfixn e = get_dev_val(ldeg);
    sfixn d = dgQ;

    if (e == -1) { 
        cudaFree(buffer); 
        return CUMODP_SUCCESS; 
    }

    const sfixn *LA = LQ;
    sfixn *LC; 

    //////////////////////////////////////
    // Compute possible subresultant, S_e
    //////////////////////////////////////
    if (d > e + 1) {
        LC = S + Ssz - Bd * (e + 2) * (e + 1) / 2;
        //list_def_to_reg_subres_dev(dgP - dgQ, Bd, LC, LB, LA, e, d, p, pinv);
        list_def_to_reg_subres2_dev(dgP - dgQ, Bd, LC, LB, LA, e, d, p, pinv);
    } else {
        LC = LB;
    }

    // all subresultants are computed
    if (e == 0) { cudaFree(buffer); return CUMODP_SUCCESS; }

    //////////////////////////////////
    // Compute subresultant, S_{e-1}
    //////////////////////////////////
    sfixn *LT = S + Ssz - Bd * (e + 1) * e / 2;
    list_next_subres_dev(dgP - dgQ, Bd, LT, d, LA, e, LB, p, pinv, LW); 

    /////////////////////////////
    // Prepare for the next pair
    /////////////////////////////
    set_array_dev(-1, Bd + 1, ldeg);
    list_deg_dev(Bd, ldeg, e, LT);
    has_same_val_dev(ldeg + Bd, Bd, ldeg);

    //printDeviceArray(ldeg, Bd+1);
    if (0 == get_dev_val(ldeg + Bd)) {
        cudaFree(buffer);
        // clean corrupted data
        reset_vector_dev(Ssz, S);
        return CUMODP_SUBRES_NON_REGULAR;
    }
    
    LA = LC;
    LB = LT;
    d = e;
    e = get_dev_val(ldeg);

    while (e != -1) {

        if (d > e + 1) {
            LC = S + Ssz - Bd * (e + 2) * (e + 1) / 2;
            //list_def_to_reg_subres_dev(1, Bd, LC, LB, LA, e, d, p, pinv);
            list_def_to_reg_subres2_dev(1, Bd, LC, LB, LA, e, d, p, pinv);
        } else { 
            LC = LB; 
        }

        // all subresultant computed.  
        if (e == 0) { break; }

        //////////////////////////////////
        // Compute subresultant, S_{e-1}
        //////////////////////////////////
        LT = S + Ssz - Bd * (e + 1) * e / 2;
        list_next_subres_dev(1, Bd, LT, d, LA, e, LB, p, pinv, LW); 
    
        //////////////////////////////
        // Prepare for the next pair
        //////////////////////////////
        set_array_dev(-1, Bd + 1, ldeg);
        list_deg_dev(Bd, ldeg, e, LT);
        has_same_val_dev(ldeg + Bd, Bd, ldeg);

        if (0 == get_dev_val(ldeg + Bd)) {
            cudaFree(buffer);
            // clean corrupted data
            reset_vector_dev(Ssz, S);
            return CUMODP_SUBRES_NON_REGULAR;
        }

        LA = LC;
        LB = LT;
        d = e;
        e = get_dev_val(ldeg);
    }

    cudaFree(buffer);
    if (DEBUG) checkCudaError("fine_subres_uni_dev");
    return CUMODP_SUCCESS;
}

/**
 * Fine-grained algorithm to subresultant chain cubes by the evaluation and 
 * interpolation.
 *
 * @P   : bivariate rdr-poly over Z_p
 * @Q   : bivaraite rdr-poly over Z_p
 *
 * Assume that deg(P, y) >= deg(Q, y) > 0. The data layout is different from
 * the modpn, and one can be converted into the other by a rectangle matrix 
 * transposition. 
 *
 * For example, let B be the number of images (or evaluation points), let
 * w be a B-th primitive root unity selected and let S_{n-1}, ... S_0 be 
 * the subresultant chain of P and Q. 
 *
 * The data layout of this scube is
 *
 * S_{n-1}(1)  S_{n-1}(w)  S_{n-1}(w^2)  ...   S_{n-1}(w^{B-1}) 
 * S_{n-2}(1)  S_{n-2}(w)  S_{n-2}(w^2)  ...   S_{n-2}(w^{B-1}) 
 * .....
 * S_{0}(1)    S_{0}(w)    S_{0}(w^2)    ...   S_{0}(w^{B-1}) 
 *
 * The data layout of modpn scube is
 *
 * S_{n-1}(1)        S_{n-2}(1)        S_{n-3}(1)        ...   S_{0}(1) 
 * S_{n-1}(w)        S_{n-2}(w)        S_{n-3}(w)        ...   S_{0}(w) 
 * .....
 * S_{n-1}(w^{B-1})  S_{n-2}(w^{B-1})  S_{n-3}(w^{B-1})  ...   S_{0}(w^{B-1})
 *
 */

///////////////////////////////////
// bivariate with raw input data
///////////////////////////////////
sfixn subres_chain2_fine_host(sfixn *S, sfixn B, sfixn w,
    sfixn npx, sfixn npy, const sfixn *P, 
    sfixn nqx, sfixn nqy, const sfixn *Q, sfixn p) 
{
    sfixn bpow = ceiling_log2(B);
    sfixn *W;
    cudaMalloc((void **)&W, sizeof(sfixn)<<(bpow - 1));
    get_powers_binary(bpow - 1, W, w, p); 

    // Step (2) Evaluate P at powers of w.
    sfixn *P_d, *P2_d;
    cudaMalloc((void**)&P_d, sizeof(sfixn) * (npy << bpow));
    cudaMemcpy(P_d, P, sizeof(sfixn) * npx * npy, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&P2_d, sizeof(sfixn) * (npy << bpow));
    expand_to_list_fft_dev(npy, B, bpow, P2_d, npx, P_d);

    // now P2_d holds the evaluated data, layout (I) at subres.cu
    list_stockham_dev(P2_d, npy, bpow, W, p);
    // now P_d holds the transposed data, layout (II) at subres.cu
    transpose_dev(P_d, P2_d, B, npy);
    // free P2_d
    cudaFree(P2_d);
             
    // Step (3) Evaluate Q at powers of w.
    sfixn *Q_d, *Q2_d;
    cudaMalloc((void**)&Q_d, sizeof(sfixn) * (nqy << bpow));
    cudaMemcpy(Q_d, Q, sizeof(sfixn) * nqx * nqy, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Q2_d, sizeof(sfixn) * (nqy << bpow));
    expand_to_list_fft_dev(nqy, B, bpow, Q2_d, nqx, Q_d);
    // Q2_d holds the evaluated data, layout (I) at subres.cu
    list_stockham_dev(Q2_d, nqy, bpow, W, p);
    // Q_d holds the transposed data, layout (II) at subres.cu
    transpose_dev(Q_d, Q2_d, B, nqy);
    // free Q2_d and W
    cudaFree(Q2_d);
    cudaFree(W);

    // Step (4) Compute subres chain for each evaluation point
    sfixn *Sd;
    sfixn sz = (nqy * nqy - nqy) / 2;
    cudaError_t err = cudaMalloc((void **)&Sd, sizeof(sfixn) * sz * B);
    if (err != cudaSuccess) {
        cudaFree(P_d);
        cudaFree(Q_d);
        return 1;
    }

    reset_vector_dev(sz * B, Sd);
    bool ret = fine_subres_uni_dev(Sd, B, npy - 1, P_d, nqy - 1, Q_d, p);
    cudaFree(P_d);
    cudaFree(Q_d);
    if (!ret) { cudaFree(Sd); return 1; }
    cudaMemcpy(S, Sd, sizeof(sfixn) * sz * B, cudaMemcpyDeviceToHost);
    cudaFree(Sd);

    if (DEBUG) checkCudaError("fine-grained parallel subres_chain2");
    return 0;
}

////////////////////
// Trivariate case
////////////////////

/**
 * Given P(x, y, Z) and Q(x, y, z) with partial degrees in z, m and n, 
 * we compute the subresultant chains of P and Q in z by FFT evaluations.
 *
 * The data layout is the following. Assume that m >= n = 1 and
 * that the degree bound in x and y are both equal to 4. Let wx and
 * wy are 4-th primitive roots of unity. Let 
 *
 * P = p0 + p1 * z + p2 * z^2
 *
 * and let 
 *
 * Q = q0 + q1 * z
 *
 * (1) Evaluate P, by a list of 2d FFT computations
 *
 *  p0(1, 1)     p0(wx, 1)     p0(wx^2, 1)     p0(wx^3, 1)
 *  p0(1, wy)    p0(wx, wy)    p0(wx^2, wy)    p0(wx^3, wy)
 *  p0(1, wy^2)  p0(wx, wy^2)  p0(wx^2, wy^2)  p0(wx^3, wy^2)
 *  p0(1, wy^3)  p0(wx, wy^3)  p0(wx^2, wy^3)  p0(wx^3, wy^3)
 *
 *  p1(1, 1)     p1(wx, 1)     p1(wx^2, 1)     p1(wx^3, 1)
 *  p1(1, wy)    p1(wx, wy)    p1(wx^2, wy)    p1(wx^3, wy)
 *  p1(1, wy^2)  p1(wx, wy^2)  p1(wx^2, wy^2)  p1(wx^3, wy^2)
 *  p1(1, wy^3)  p1(wx, wy^3)  p1(wx^2, wy^3)  p1(wx^3, wy^3)
 *
 *  p2(1, 1)     p2(wx, 1)     p2(wx^2, 1)     p2(wx^3, 1)
 *  p2(1, wy)    p2(wx, wy)    p2(wx^2, wy)    p2(wx^3, wy)
 *  p2(1, wy^2)  p2(wx, wy^2)  p2(wx^2, wy^2)  p2(wx^3, wy^2)
 *  p2(1, wy^3)  p2(wx, wy^3)  p2(wx^2, wy^3)  p2(wx^3, wy^3)
 *
 * (2) Transpose the evalations
 * 
 * (00)    p0(1,       1)    p1(1,       1)    p2(1,       1)
 * (01)    p0(wx,      1)    p1(wx,      1)    p2(wx,      1)
 * (02)    p0(wx^2,    1)    p1(wx^2,    1)    p2(wx^2,    1)
 * (03)    p0(wx^3,    1)    p1(wx^3,    1)    p2(wx^3,    1)
 * (04)    p0(1,      wy)    p1(1,      wy)    p2(1,      wy)
 * (05)    p0(wx,     wy)    p1(wx,     wy)    p2(wx,     wy)
 * (06)    p0(wx^2,   wy)    p1(wx^2,   wy)    p2(wx^2,   wy)
 * (07)    p0(wx^3,   wy)    p1(wx^3,   wy)    p2(wx^3,   wy)
 * (08)    p0(1,    wy^2)    p1(1,    wy^2)    p2(1,    wy^2)
 * (09)    p0(wx,   wy^2)    p1(wx,   wy^2)    p2(wx,   wy^2)
 * (10)    p0(wx^2, wy^2)    p1(wx^2, wy^2)    p2(wx^2, wy^2)
 * (11)    p0(wx^3, wy^2)    p1(wx^3, wy^2)    p2(wx^3, wy^2)
 * (12)    p0(1,    wy^3)    p1(1,    wy^3)    p2(1,    wy^3)
 * (13)    p0(wx,   wy^3)    p1(wx,   wy^3)    p2(wx,   wy^3)
 * (14)    p0(wx^2, wy^3)    p1(wx^2, wy^3)    p2(wx^2, wy^3)
 * (15)    p0(wx^3, wy^3)    p1(wx^3, wy^3)    p2(wx^3, wy^3)
 *
 * (3) Evaluate Q, by a list of 2d FFT computations
 *
 *  q0(1, 1)     q0(wx, 1)     q0(wx^2, 1)     q0(wx^3, 1)
 *  q0(1, wy)    q0(wx, wy)    q0(wx^2, wy)    q0(wx^3, wy)
 *  q0(1, wy^2)  q0(wx, wy^2)  q0(wx^2, wy^2)  q0(wx^3, wy^2)
 *  q0(1, wy^3)  q0(wx, wy^3)  q0(wx^2, wy^3)  q0(wx^3, wy^3)
 *
 *  q1(1, 1)     q1(wx, 1)     q1(wx^2, 1)     q1(wx^3, 1)
 *  q1(1, wy)    q1(wx, wy)    q1(wx^2, wy)    q1(wx^3, wy)
 *  q1(1, wy^2)  q1(wx, wy^2)  q1(wx^2, wy^2)  q1(wx^3, wy^2)
 *  q1(1, wy^3)  q1(wx, wy^3)  q1(wx^2, wy^3)  q1(wx^3, wy^3)
 *
 * (4) Transpose the evalations
 *
 * (00)    q0(1,       1)    q1(1,       1) 
 * (01)    q0(wx,      1)    q1(wx,      1) 
 * (02)    q0(wx^2,    1)    q1(wx^2,    1) 
 * (03)    q0(wx^3,    1)    q1(wx^3,    1) 
 * (04)    q0(1,      wy)    q1(1,      wy) 
 * (05)    q0(wx,     wy)    q1(wx,     wy) 
 * (06)    q0(wx^2,   wy)    q1(wx^2,   wy) 
 * (07)    q0(wx^3,   wy)    q1(wx^3,   wy) 
 * (08)    q0(1,    wy^2)    q1(1,    wy^2) 
 * (09)    q0(wx,   wy^2)    q1(wx,   wy^2) 
 * (10)    q0(wx^2, wy^2)    q1(wx^2, wy^2) 
 * (11)    q0(wx^3, wy^2)    q1(wx^3, wy^2) 
 * (12)    q0(1,    wy^3)    q1(1,    wy^3) 
 * (13)    q0(wx,   wy^3)    q1(wx,   wy^3) 
 * (14)    q0(wx^2, wy^3)    q1(wx^2, wy^3) 
 * (15)    q0(wx^3, wy^2)    q1(wx^3, wy^3) 
 *
 * (5) Compute a list of subresultant chains (transposed layout to modpn)
 *
 *  S_{0, n-1}  S_{1, n-1}  S_{2, n-1} ....  S_{Bx*By - 1, n-1}
 *  S_{0, n-2}  S_{1, n-2}  S_{2, n-2} ....  S_{Bx*By - 1, n-2}
 *  ...
 *  S_{0, 1}    S_{1, 1}    S_{2, 1}   ....  S_{Bx*By - 1, 1}
 *  S_{0, 0}    S_{1, 0}    S_{2, 0}   ....  S_{Bx*By - 1, 0}
 * 
 *  Here n = 1, and it is trivial, only the last row.
 */ 

/**
 * Memory usage notes:
 * 
 * (a) Setting, the partial degrees of P and Q satisfying
 *
 *     deg(P, x) <= dx, deg(Q, x) <= dx
 *     deg(P, y) <= dy, deg(Q, y) <= dy
 *     deg(P, z)  = d1, deg(Q, z)  = d2, dz1 >= d2
 *
 * (b) The evaluation bound 
 *
 *     Bx = dx * (d2 + d1) + 1
 *     By = dy * (d2 + d1) + 1
 *   
 *     Bx2 = 2^ceiling_log2(Bx);
 *     By2 = 2^ceiling_log2(By);
 *
 *     B = Bx2 * By2
 * 
 * (c) The sizes of the precomputed primitive roots of unity
 *
 *     x-direction: Bx2 / 2                             (1)
 *     y-direction: By2 / 2                             (2)
 *
 * (d) Evaluation of P and Q
 *
 *     P: B * (1 + d1)                                  (3) 
 *     Q: B * (1 + d2)                                  (4) 
 *
 *     (1) and (2) can be freed now.
 *
 * (e) The size of the scube,
 *
 *     Ssz = B * d2 * (d2 + 1) / 2.                     (5)   
 * 
 * (f) The size of the buffer,
 *
 *     nbuff = 2 * B * d1 + B + 1.                      (6)
 *
 * The maximal amount of memory in use is
 *
 *  (3) + (4) + (5) + (6)
 *
 *  = B * (1 + d1 + 1 + d2 + 2 * d1 + 1) 
 *    + B * d2 * (d2 + 1) / 2 + 1 
 *
 *  = B * (3 * d1 + d2 + 3) + B * (d2 + d2^2) / 2 + 1    (7) 
 *
 * Example, let dx = dy = d1 = d2 = 20, then 
 *
 * B = Bx * By = 2^20
 *
 * (7) = B * (3 * 20 + 20 + 3) + B * (20 + 400) / 2 + 1
 *     = B * 83 + B * 210 + 1
 *     = B * 293 + 1
 *
 * This is larger than 1GB, since
 *
 *     293 * 2^20 * sizeof(sfixn) ~ 1172 MB
 */

sfixn subres_chain3_fine_host(sfixn *S, sfixn Bx, sfixn By, sfixn wx, sfixn wy,
    sfixn npx, sfixn npy, sfixn npz, const sfixn *P, 
    sfixn nqx, sfixn nqy, sfixn nqz, const sfixn *Q, sfixn p) 
{
    sfixn bxpow = ceiling_log2(Bx);
    sfixn bypow = ceiling_log2(By);
    
    sfixn *Wx;
    cudaMalloc((void **)&Wx, sizeof(sfixn)<<(bxpow - 1));
    get_powers_binary(bxpow - 1, Wx, wx, p); 
    //printf("wx = %d\n", wx);

    sfixn *Wy;
    cudaMalloc((void **)&Wy, sizeof(sfixn)<<(bypow - 1));
    get_powers_binary(bypow - 1, Wy, wy, p); 
    //printf("wy = %d\n", wy);
    
    // Step (2) Evaluate P at powers of wx and wy
    sfixn *P_d, *P2_d;
    cudaMalloc((void**)&P_d, sizeof(sfixn) * (npz << (bxpow + bypow)));
    cudaMemcpy(P_d, P, sizeof(sfixn) * npx * npy * npz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&P2_d, sizeof(sfixn) * (npz << (bxpow + bypow)));
    expand_to_list_fft2_dev(npz, bxpow, bypow, P2_d, npx, npy, P_d);
    
    // now P2_d holds the evaluated data
    list_bivariate_stockham_dev(P2_d, npz, bxpow, Wx, bypow, Wy, p);
    // now P_d holds the transposed data
    transpose_dev(P_d, P2_d, Bx * By, npz);
    // P2_d freed
    cudaFree(P2_d);

    // Step (3) Evaluate Q at powers of wx and wy.
    sfixn *Q_d, *Q2_d;
    cudaMalloc((void**)&Q_d, sizeof(sfixn) * (nqz << (bxpow + bypow)));
    cudaMemcpy(Q_d, Q, sizeof(sfixn) * nqx * nqy * nqz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Q2_d, sizeof(sfixn) * (nqz << (bxpow + bypow)));
    expand_to_list_fft2_dev(nqz, bxpow, bypow, Q2_d, nqx, nqy, Q_d);

    // now Q2_d holds the evaluated data
    list_bivariate_stockham_dev(Q2_d, nqz, bxpow, Wx, bypow, Wy, p);
    // now P_d holds the transposed data
    transpose_dev(Q_d, Q2_d, Bx * By, nqz);
    // Q2_d, Wx, Wy, freed
    cudaFree(Q2_d);
    cudaFree(Wx);
    cudaFree(Wy);
    
    // Step (4) Compute subres chain for each evaluation point
    sfixn *Sd;
    sfixn sz = (nqz * nqz - nqz) / 2;
    
    cudaError_t err = cudaMalloc((void **)&Sd, sizeof(sfixn) * sz * Bx * By);
    if (err != cudaSuccess) {
        cudaFree(P_d);
        cudaFree(Q_d);
        return 1;
    }

    reset_vector_dev(sz * Bx * By, Sd);
    bool ret = fine_subres_uni_dev(Sd, Bx * By, npz - 1, P_d, nqz - 1, Q_d, p);
    cudaFree(P_d);
    cudaFree(Q_d);
    if (!ret) { cudaFree(Sd); return 1; }
     
    cudaMemcpy(S, Sd, sizeof(sfixn) * sz * Bx * By, cudaMemcpyDeviceToHost);
    cudaFree(Sd);
    if (DEBUG) checkCudaError("fine-grained parallel subres_chain3");
    return 0;
}

///////////  END OF THE FILE ///////////
