#include <cassert>
#include "../include/fft_aux.h"
#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/cudautils.h"
#include "../include/cumodp.h"
#include "../include/stockham.h"

///////////////////////////////////////////////////////////////////////////////
//              filling powers of primitive roots of unity                   // 
///////////////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * Compute 1, w, w^2, ... w^{n-1} naively, for host array W
 */
inline void get_powers_naive(sfixn n, sfixn *W, sfixn w, sfixn p, double pinv) 
{
    W[0] = 1; W[1] = w;
    for (sfixn i = 2; i < n; ++i) { W[i] = mul_mod(W[i-1], w, p, pinv); }
}

inline void get_powers_naive(sfixn n, sfixn *W, sfixn w, sfixn p) {
    double pinv = 1 / (double)p;
    W[0] = 1; W[1] = w;
    for (sfixn i = 2; i < n; ++i) { W[i] = mul_mod(W[i-1], w, p, pinv); }
}

/**
 * @n, the size of idata and odata 
 * @idata, input array
 * @odata, output array
 * @wn, value to be scaled
 * @p, prime number
 *
 * Compute idata[i] = odata[i] * wn mod p
 */
__global__ void
double_expand_ker(sfixn n, sfixn *odata, const sfixn *idata, sfixn wn,
                  sfixn p, double pinv)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn tid = threadIdx.x + (bid << E_THD);
    odata[tid] = mul_mod(idata[tid], wn, p, pinv);
}

/**
 * Compute 1, w, w^2, ... w^{n-1}, n = 2^e, for device array W
 * 
 * @W, device array of length n = 2^e   
 * @w, integer in the finite field Zp
 * @p, prime number
 *
 */
void get_powers_binary (sfixn e, sfixn *W, sfixn w, sfixn p) {

    sfixn b = 10, m = (1L << b), *Y;
    double pinv = 1/(double)p;

    if (e <= b) {
        sfixn n = (1 << e);
        Y = new sfixn[n];
        get_powers_naive(n, Y, w, p, pinv);
        cudaMemcpy(W, Y, sizeof(sfixn)*n, cudaMemcpyHostToDevice);
        delete [] Y;
        return;
    }

    Y = new sfixn[m];
    get_powers_naive(m, Y, w, p, pinv);

    cudaMemcpy(W, Y, sizeof(sfixn)*m, cudaMemcpyHostToDevice);

    // now expand W to the size n 
    sfixn *Z = W + m;
    sfixn ws = mul_mod(Y[m-1], w, p, pinv);
    delete [] Y;
    // the dimension of nBlk to be decided.
    dim3 nBlk(1, 1, 1);
    sfixn nThd = N_THD, s, nb;

    for (sfixn i = b; i < e; ++i) {
        // the size of the array has been filled.
        s = (1 << i);
        nb = (s >> E_THD);

        // set the grid size, handle large arrays
        if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }
        else { nBlk.x = nb; }
 
        double_expand_ker<<<nBlk, nThd>>>(s, Z, W, ws, p, pinv);

        // pivot move to the next power of 2
        ws = mul_mod(ws, ws, p, pinv);
        // the starting address moves to the next one
        Z += s;
    }

    if (DEBUG) checkCudaError("error found in get_powers_binary");
}
#undef E_THD
#undef N_THD

/**
 * Precompute powers of primitive roots of unity.
 *
 * @W, the output 
 * @m, the number of roots
 * @es, the exponent array of size m
 * @p, the fourier prime
 *
 * Example. Let es[0] = 3, es[1] = 4, s be a 8-th primitive root of unity
 * and t be a 16-th primitive root of unity. 
 * Then W consists of two segments
 *
 * [1, s, s^2, s^3, 1, t, t^2, t^3, t^4, t^5, t^6, t^7]
 *
 */
void get_powers_of_roots(int m, sfixn *W, const sfixn *es, const sfixn p) 
{
    for (int i = 0; i < m; ++i) {
        sfixn w = primitive_root(es[i], p);
        //printf("w[%d] = %d\n", i, w);
        get_powers_binary(es[i] - 1, W, w, p); 
        W += (sfixn(1) << (es[i] - 1));
    }
}

/**
 * Inverse  
 *
 * [1, s, s^2, s^3, 1, t, t^2, t^3, t^4, t^5, t^6, t^7]
 *
 * to
 *
 * [1, s^{-1}, s^{-2}, s^{-3}, 
 *  1, t^{-1}, t^{-2}, t^{-3}, t^{-4}, t^{-5}, t^{-6}, t^{-7}]
 * 
 * This can be used in the interpolation stage
 */
void inv_powers_of_roots(int m, sfixn *W, const sfixn *es, const sfixn p) {
    for (int i = 0; i < m; ++i) {
        sfixn w;
        cudaMemcpy(&w, W + 1, sizeof(sfixn), cudaMemcpyDeviceToHost);
        //printf("p = %d, w = %d \n", p, w);
        w = inv_mod(w, p);
        get_powers_binary(es[i] - 1, W, w, p); 
        W += (sfixn(1) << (es[i] - 1));
    }
}

//////////////////////////////////////////////////////////////////////
// pointwise modular multiplication of two vectors of the same size //
//////////////////////////////////////////////////////////////////////
#if DEBUG > 0
#define E_THD (2)
#else
#define E_THD (7)
#endif

#define N_THD (1 << E_THD)

/**
 * @n    : the length of the vector, n = 2^k 
 * @Y    : device vector of length n
 * @X    : device vector of length n
 * @p    : prime number
 * @pinv : inverse of p
 *
 * Y[i] = Y[i] * X[i] mod p
 *
 */
__global__ 
void pointwise_mul_ker(sfixn k, sfixn *Y, const sfixn *X, sfixn p, double pinv)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn *Yb = Y + (bid << E_THD); 
    const sfixn *Xb = X + (bid << E_THD); 
    Yb[threadIdx.x] = mul_mod(Yb[threadIdx.x], Xb[threadIdx.x], p, pinv);
}

void pointwise_mul_dev(sfixn n, sfixn k, sfixn *Y, const sfixn *X, sfixn p) 
{
    if (DEBUG) assert(n >= N_THD);
    if (DEBUG) assert((n & (n - 1)) == 0);

    sfixn nb = (n >> E_THD);
    dim3 nBlk(nb, 1, 1);
    if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }

    double pinv = 1 / (double)p;
    pointwise_mul_ker<<<nBlk, N_THD>>>(k, Y, X, p, pinv);
}

/**
 * Compute X[i] = X[i] * a mod p
 */
__global__ 
void scale_vector_ker(sfixn a, sfixn *X, sfixn p, double pinv) {
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn *Xb = X + (bid << E_THD); 
    sfixn b = Xb[threadIdx.x];
    Xb[threadIdx.x] = mul_mod(a, b, p, pinv);
}

void scale_vector_dev(sfixn a, sfixn n, sfixn *X, sfixn p) {
    if (DEBUG) assert(n >= N_THD);
    if (DEBUG) assert((n & (N_THD - 1)) == 0);

    sfixn nb = (n >> E_THD);
    dim3 nBlk(nb, 1, 1);
    if (nb > (1L << 15)) { nBlk.x = (1 << 15); nBlk.y = (nb >> 15); }

    double pinv = 1 / (double)p;
    scale_vector_ker<<<nBlk, N_THD>>>(a, X, p, pinv);
}

/**
 * Compute X[i] = 1 / X[i] mod p 
 */
__global__ void inverse_vector_ker(sfixn *X, sfixn p) {
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn *Xb = X + (bid << E_THD); 
    sfixn b = Xb[threadIdx.x];
    Xb[threadIdx.x] = inv_mod(b, p);
}

void inverse_vector_dev(sfixn n, sfixn *X, sfixn p) {
    if (DEBUG) assert(n >= N_THD);
    if (DEBUG) assert((n & (N_THD - 1)) == 0);

    sfixn nb = (n >> E_THD);
    dim3 nBlk(nb, 1, 1);
    if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }

    inverse_vector_ker<<<nBlk, N_THD>>>(X, p);
}

/**
 * reset a device array 
 *
 * @X, input/output coefficient vector of length s, device array
 * @s, positive integer
 * 
 **/
__global__ void reset_ker(sfixn s, sfixn *X) {
    sfixn bid = blockIdx.x + blockIdx.y * gridDim.x;
    sfixn tid = threadIdx.x + (bid << E_THD);
    if (tid < s) X[tid] = 0;
}

/**
 * @X, input and output coefficient vector 
 * @m, power of 2
 * @n, m >= n
 *
 * Example, let n = 7, m = 16, given
 *
 * X = [0, 1, 2, 3, 4, 5, 6, *, *, *, *, *, *, *, *, *],  
 *
 * Only reset the last m - n = 9 elements.
 */ 

void expand_to_fft_dev(sfixn m, sfixn n, sfixn *X) {
    sfixn s = m - n;
    // the number of block needed is
    // s / N_THD if N_THD divides s, 
    // otherwise, floor(s / N_THD) + 1 
    sfixn nb = (s / N_THD) + ((s % N_THD) ? 1 : 0);
    // Note that nb might not be a power of 2!
    // We factor nb approximately.
    sfixn nx, ny, found;
    found = approx_factor(nb, &nx, &ny, (1L << 16));
    if (DEBUG) assert(found);
    dim3 nBlk(nx, ny);
    
    reset_ker<<<nBlk, N_THD>>>(s, X + n);
    cudaThreadSynchronize();
}

/**
 * Assume that a power of 2 can be factored if nb if big.
 */
void reset_vector_dev(sfixn n, sfixn *X) {
    sfixn nb = (n / N_THD) + ((n % N_THD) ? 1 : 0);
    dim3 nBlk(nb, 1, 1);
    if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }
    reset_ker<<<nBlk, N_THD>>>(n,  X);
    cudaThreadSynchronize();
}

/**
 * Expand a rdr-poly into a rdr-poly with FFT size coefficients.
 *
 * @q,    the number of coefficients (q >= 1)
 * @m,    FFT size m = 2^k
 * @n,    cofficient size, n <= m
 * @din,  device array of size q * n filled with the input data
 * @dout, (output) device array of size q * m 
 *
 * The total number of thread blocks is q * m / N_THD.
 *
 * Example, let 
 *
 * F = (1 + 2*x + 3*x^2) + (4 + 5*x + 6*x^2)*y + (7 + 8*x + 9*x^2)*y^2
 *
 * Expand F into
 *
 * F = (1 + 2*x + 3*x^2 + 0*x^3) + 
 *     (4 + 5*x + 6*x^2 + 0*x^3)*y + 
 *     (7 + 8*x + 9*x^2 + 0*x^3)*y^2
 *
 * That is, 
 *
 * {1, 2, 3, 4, 5, 6, 7, 8, 9} ==> {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0}
 *
 * */
__global__ void expand_to_list_fft_ker(sfixn m, sfixn k, sfixn *dout, 
                                       sfixn n, const sfixn *din)
{
    sfixn bid = (blockIdx.y << 15) + blockIdx.x;
    sfixn tid = (bid << E_THD) + threadIdx.x;
    // the number of threads for one coefficient is m.
    sfixn tq = (tid >> k);
    sfixn tr = (tid & (m - 1));
    dout[tid] = ((tr < n) ? din[tr + tq * n] : 0);
}

void expand_to_list_fft_dev(sfixn q, sfixn m, sfixn k, sfixn *dout, sfixn n,
                            const sfixn *din)
{
    sfixn nb = ((q << k) >> E_THD);
    assert(nb > 0);

    dim3 nBlk(nb, 1, 1);
    if (nb > (1L << 15)) { nBlk.x = (1L << 15); nBlk.y = (nb >> 15); }
    
    expand_to_list_fft_ker<<<nBlk, N_THD>>>(m, k, dout, n, din);
    cudaThreadSynchronize();
}

#undef E_THD
#undef N_THD

/****************** Matrix transposition *************************************/
#define E_BLK_DIM 4
#define N_BLK_DIM (1 << E_BLK_DIM)

// This kernel is optimized to ensure all global reads and writes are
// coalesced, and to avoid bank conflicts in shared memory.  This kernel
// is up to 11x faster than the naive kernel below.  Note that the shared
// memory array is sized to (BLOCK_DIM+1)*BLOCK_DIM. This pads each row of
// the 2D block in shared memory so that bank conflicts do not occur when
// threads address the array column-wise.

// Principle: We are transposing the input matrix block-wise.
// Each of these blocks are moraly square of order N_BLK_DIM
// However, at the end of each row (on the shared memory copy)
// we add a padding element to deal with bank conflicts.

// Each thread block is in chrage of one matrix block.
// Each thread in a thread block is in charge of one matrix entry.

/**
 *  Example, let M be the matrix to be transposed:
 *
 *      [  0   1   2   3 ]        [ 0  4   8  12  16  20 ]  
 *  M = [  4   5   6   7 ]   Mt = [ 1  5   9  13  17  21 ]
 *      [  8   9  10  11 ]        [ 2  6  10  14  18  22 ]
 *      [ 12  13  14  15 ]        [ 3  7  11  15  19  23 ]
 *      [ 16  17  18  19 ]         
 *      [ 20  21  22  23 ]          
 * 
 *  We assume each thread block is of size (2, 2), and thus the grid is 
 *  of size (2, 3). The work done by the first thread block is following:
 *
 *  Block(0, 0) has four threads T00, T10, T01, and T11, and
 * 
 *  T00 reads 0 = M[0, 0] to block[0, 0]
 *  T10 reads 1 = M[0, 1] to block[0, 1]
 *  T01 reads 4 = M[1, 0] to block[1, 0]
 *  T11 reads 5 = M[1, 1] to block[1, 1]
 *
 *  That is, it is a direct copy, coalesced accesses. After reading
 *  in data, the first thread block writes out data as follows:
 *
 *  T00 writes 0 = block[0, 0] to Mt[0, 0]
 *  T10 writes 4 = block[1, 0] to Mt[0, 1]
 *  T01 writes 1 = block[0, 1] to Mt[1, 0]
 *  T11 writes 5 = block[1, 1] to Mt[1, 1]
 * 
 *  Reading to the shared memory block is transposed in the latter step.
 *
 * @idata, input matrix
 * @odata, output matrix
 *
 */
__global__ void
transpose_ker(sfixn *odata, const sfixn *idata, sfixn width, sfixn height)
{
    __shared__ sfixn block[N_BLK_DIM][N_BLK_DIM+1];

    sfixn i;

    // read the matrix tile into shared memory
    sfixn xIndex = (blockIdx.x << E_BLK_DIM) + threadIdx.x;
    sfixn yIndex = (blockIdx.y << E_BLK_DIM) + threadIdx.y;
    if ((xIndex < width) && (yIndex < height)) {
        i = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[i];
    }
    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = (blockIdx.y << E_BLK_DIM) + threadIdx.x;
    yIndex = (blockIdx.x << E_BLK_DIM) + threadIdx.y;
    if ((xIndex < height) && (yIndex < width)) {
        i = yIndex * height + xIndex;
        odata[i] = block[threadIdx.x][threadIdx.y];
    }
}

/**
 * Transpose a matrix with unbalanced dimensions. In this kernel, we assume that
 * the width of the matrix (the number of columns) is large, but the height of 
 * the matrix (the numer of rows) is relatively small. The kernel configuration 
 * is slightly different from the standard one, since blockIdx.x or blockIdx.y 
 * is at most 65535 = 2^16 - 1. 
 *
 * We assume that a power of 2 can be factored out from x component, and 
 * the method is to encode the pair (n * 2^e, m) as (n, 2^e*m), the offset 
 * exponetnt e will be passed as a parameter. To recover the true block indices,
 * we calculate
 *
 * bkIdx_x = blockIdx.x + 2^e * rem(blockIdx.y, 2^e)
 * bkIdx_y = quo(blockIdx.y, 2^e)
 *
 * The execution ordering of thread blocks is still preserved.
 *
 */
__global__ void 
transpose_large_width_ker(sfixn *odata, const sfixn *idata, sfixn width, 
    sfixn height, sfixn e)
{
    __shared__ sfixn block[N_BLK_DIM][N_BLK_DIM+1];

    // compute the true block indices
    // sfixn bkIdx_x = blockIdx.x + ((blockIdx.y & ((sfixn(1) << e) - 1)) << e);
    sfixn bkIdx_x = blockIdx.x + (rem2e(blockIdx.y, e) << e);
    sfixn bkIdx_y = blockIdx.y >> e;

    // read the matrix tile into shared memory
    sfixn xIndex = (bkIdx_x << E_BLK_DIM) + threadIdx.x;
    sfixn yIndex = (bkIdx_y << E_BLK_DIM) + threadIdx.y;
    if ((xIndex < width) && (yIndex < height)) {
        sfixn i = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[i];
    }
    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = (bkIdx_y << E_BLK_DIM) + threadIdx.x;
    yIndex = (bkIdx_x << E_BLK_DIM) + threadIdx.y;
    if ((xIndex < height) && (yIndex < width)) {
        sfixn i = yIndex * height + xIndex;
        odata[i] = block[threadIdx.x][threadIdx.y];
    }
}

/**
 * We assume that a power of 2 can be factored out from y component, and 
 * the method is to encode the pair (n, m * 2^e) as (2^e * n, m), the offset 
 * exponetnt e will be passed as a parameter. To recover the true block indices,
 * we calculate
 *
 * bkIdx_x = rem(blockIdx.x, 2^e) 
 * bkIdx_y = quo(blockIdx.x, 2^e) + 2^e * blockIdx.y
 *
 * The execution ordering of thread blocks is still preserved.
 */
__global__ void 
transpose_large_height_ker(sfixn *odata, const sfixn *idata, sfixn width, 
    sfixn height, sfixn e)
{
    __shared__ sfixn block[N_BLK_DIM][N_BLK_DIM+1];

    // compute the true block indices
    sfixn bkIdx_x = rem2e(blockIdx.x, e);
    sfixn bkIdx_y = (blockIdx.x >> e) + (blockIdx.y << e);

    // read the matrix tile into shared memory
    sfixn xIndex = (bkIdx_x << E_BLK_DIM) + threadIdx.x;
    sfixn yIndex = (bkIdx_y << E_BLK_DIM) + threadIdx.y;
    if ((xIndex < width) && (yIndex < height)) {
        sfixn i = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[i];
    }
    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = (bkIdx_y << E_BLK_DIM) + threadIdx.x;
    yIndex = (bkIdx_x << E_BLK_DIM) + threadIdx.y;
    if ((xIndex < height) && (yIndex < width)) {
        sfixn i = yIndex * height + xIndex;
        odata[i] = block[threadIdx.x][threadIdx.y];
    }
}

/**
 * Transpose a matrix inside device.
 *
 * @odata  : output matrix on device, array of length w * h
 * @idata  : input matrix on device, array of length h * w
 * @w      : the width of the matrix
 * @h      : the height of the matrix
 *
 * Output  :  odata is the transposed matrix with width = h and height = w.
 **/

void transpose_dev(sfixn *odata, const sfixn *idata, sfixn w, sfixn h) {
    sfixn gx = (w >> E_BLK_DIM) + ((w & (N_BLK_DIM - 1)) == 0 ? 0 : 1);
    sfixn gy = (h >> E_BLK_DIM) + ((h & (N_BLK_DIM - 1)) == 0 ? 0 : 1);
    dim3 ngrid(gx, gy, 1);
    dim3 nthread(N_BLK_DIM, N_BLK_DIM, 1);

    //printf("w = %d, h = %d\n", w, h);
    //printf("gx = %d, gy = %d\n", ngrid.x, ngrid.y);

    if (gx > (1L << 15)) {
        //  width up to 2^(15 + e) * N_BLK_DIM = 2^26
        // height up to 2^(15 - e) * N_BLK_DIM = 2^12
        const sfixn e = 7;
        if ((rem2e(gx, e) == 0) && ((gy << e) <= (1L << 15))) {
            ngrid.x >>= e;
            ngrid.y <<= e;
            transpose_large_width_ker<<<ngrid, nthread>>>(odata, idata, w, h, e);
            cudaThreadSynchronize();
        } else {
            printf("Cannot handle this transposition\n"); 
        }
        return;
    } else if (gy > (1L << 15)) {
        //  width up to 2^(15 - e) * N_BLK_DIM = 2^12
        // height up to 2^(15 + e) * N_BLK_DIM = 2^26
        const sfixn e = 7;
        if ((rem2e(gy, e) == 0) && ((gx << e) <= (1L << 15))) {
            ngrid.x <<= e;
            ngrid.y >>= e;
            transpose_large_height_ker<<<ngrid, nthread>>>(odata, idata, w, h, e);
            cudaThreadSynchronize();
        } else {
            printf("Cannot handle this transposition\n"); 
        }
        return;
    }

    transpose_ker<<<ngrid, nthread>>>(odata, idata, w, h);
    cudaThreadSynchronize();
}

#undef E_BLK_DIM
#undef N_BLK_DIM

////////////////////////////////////////////////////////////////////////////////
#define E_BLK_DIM 4
#define N_BLK_DIM (1 << E_BLK_DIM)

/**
 * @coeffs1: device array of size nx1 * ny1
 * @coeffs2: device array of size nx2 * ny2
 * @nx2, nx2 = 2^ex2
 * @ny2, ny2 = 2^ey2
 *
 * Given a (nx1, ny1)-rdr bivariate polynomial F, expand it coefficient vector
 * into (nx2, ny2)-rdr such that nx2 >= nx1 and ny2 >= ny1
 *
 * Require nx2, ny2 >= N_BLK_DIM.
 *
 * Example:
 *
 * f = (1 + 2x) + (3 + 4x + 5x^2)y + (6 + 7x^2) y^2
 *
 * (3, 3)-coeffs is {1, 2, 0, 
 *                   3, 4, 5, 
 *                   6, 0, 7}   
 *
 * (4, 4)-coeffs is {1, 2, 0, 0, 
 *                   3, 4, 5, 0, 
 *                   6, 0, 7, 0, 
 *                   0, 0, 0, 0}   
 *
 */
__global__ void expand_to_fft2_ker(sfixn ex2, sfixn ey2, sfixn *coeffs2, 
                                   sfixn nx1, sfixn ny1, const sfixn *coeffs1)
{
    // assume that nx and ny are kind of balanced
    sfixn x = (blockIdx.x << E_BLK_DIM) + threadIdx.x;
    sfixn y = (blockIdx.y << E_BLK_DIM) + threadIdx.y;

    if ((x < nx1) && (y < ny1)) {
        coeffs2[(y << ex2) + x] = coeffs1[y * nx1 + x];
    } else {
        coeffs2[(y << ex2) + x] = 0;
    }
}

void expand_to_fft2_dev(sfixn ex2, sfixn ey2, sfixn *coeffs2, 
                        sfixn nx1, sfixn ny1, const sfixn *coeffs1)
{
    if (DEBUG) assert((1 << ex2) >= nx1);
    if (DEBUG) assert((1 << ey2) >= ny1);
    if (DEBUG) assert((ex2 >= E_BLK_DIM) && (ey2 >= E_BLK_DIM));

    sfixn gx = ((sfixn)1 << (ex2 - E_BLK_DIM));
    sfixn gy = ((sfixn)1 << (ey2 - E_BLK_DIM));

    dim3 nblk(gx, gy, 1);
    dim3 nthd(N_BLK_DIM, N_BLK_DIM, 1);
    expand_to_fft2_ker<<<nblk, nthd>>>(ex2, ey2, coeffs2, nx1, ny1, coeffs1);
    cudaThreadSynchronize();
}

/* 
 * Extract a submatrix from a matrix, the inverse of expand_to_fft2_ker.
 * 
 **/
__global__ void extract_from_fft2_ker(sfixn nx1, sfixn ny1, sfixn *coeff1, 
                                      sfixn ex2, const sfixn *coeff2)
{
    sfixn x = (blockIdx.x << E_BLK_DIM) + threadIdx.x;
    sfixn y = (blockIdx.y << E_BLK_DIM) + threadIdx.y;
    if (x < nx1 && y < ny1) { coeff1[y * nx1 + x] = coeff2[(y << ex2) + x]; }
}

void extract_from_fft2_dev(sfixn nx1, sfixn ny1, sfixn *coeff1,
                           sfixn ex2, const sfixn *coeff2)
{
    sfixn gx =  nx1 / N_BLK_DIM + (nx1 % N_BLK_DIM == 0 ? 0 : 1);
    sfixn gy =  ny1 / N_BLK_DIM + (ny1 % N_BLK_DIM == 0 ? 0 : 1);
    dim3 nblock(gx, gy);
    dim3 nthread(N_BLK_DIM, N_BLK_DIM);
    extract_from_fft2_ker<<<nblock, nthread>>>(nx1, ny1, coeff1, ex2, coeff2);
    cudaThreadSynchronize();
}

/**
 * Expand a list bivariate polynomials into fft layout.
 *
 * @q, the number of bivariate polynomials 
 * @ex2, the expanded exponent in x (rows)
 * @ey2, the expanded exponent in y (columns)
 * @coeff2, the expanded coefficients (output)
 * @nx1, the input size in x (rows)
 * @ny1, the input size in y (columns)
 * @coeff1, the input coefficients
 *
 * For example,  ex2 = ey2 = 2, nx1 = ny1 = 3
 *
 * I = [ 0  1  2 ]   [  9 10 11 ]
 *     [ 3  4  5 ] & [ 12 13 14 ]
 *     [ 6  7  8 ]   [ 15 16 17 ]
 *
 * O = [ 0 1 2 0 ]   [  9 10 11  0 ]
 *     [ 3 4 5 0 ] & [ 12 13 14  0 ]
 *     [ 6 7 8 0 ]   [ 15 16 17  0 ]
 *     [ 0 0 0 0 ]   [  0  0  0  0 ]
 *
 * The number of threads is  q * nx2 * ny2  = (q << (ex2 + ey2)).
 *
 */
__global__ void expand_to_list_fft2_ker(sfixn ex2, sfixn ey2, sfixn *coeffs2, 
    sfixn nx1, sfixn ny1, const sfixn *coeffs1)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    
    // qtid = tid / (nx2 * ny2), the matrix works on
    sfixn qtid = (tid >> (ex2 + ey2));
    sfixn rtid = (tid & (((sfixn)1 << (ex2 + ey2)) - 1));

    const sfixn *din = coeffs1 + qtid * nx1 * ny1; 
    sfixn *dout = coeffs2 + (qtid << (ex2 + ey2));

    sfixn y = (rtid >> ex2);
    sfixn x = (rtid & (((sfixn)1 << ex2) - 1));
    
    if (x < nx1 && y < ny1) {
        dout[(y << ex2) + x] = din[y * nx1 + x];   
    } else {
        dout[(y << ex2) + x] = 0;
    }
}

void expand_to_list_fft2_dev(sfixn q, sfixn ex2, sfixn ey2, sfixn *coeffs2, 
    sfixn nx1, sfixn ny1, const sfixn *coeffs1) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn nb = (q << (ex2 + ey2)) / nthds;
    if (DEBUG) assert(nb >= 1);
    dim3 nBlks(nb, 1, 1);

    if (nb > (1L << 15)) { nBlks.x = (1L << 15); nBlks.y = (nb >> 15); }
    expand_to_list_fft2_ker<<<nBlks, nthds>>>(ex2, ey2, 
        coeffs2, nx1, ny1, coeffs1);
    cudaThreadSynchronize();
}

#undef E_BLK_DIM
#undef N_BLK_DIM
////////////////////////////////////////////////////////////////////////////////

template <int nthds>
__global__ void has_zero_ker(sfixn *ret, const sfixn *Ad, sfixn n) {
    sfixn bid = blockIdx.y * gridDim.x + blockIdx.x;
    sfixn i = bid * nthds + threadIdx.x;
    if ((i < n) && (Ad[i] == 0)) { *ret = i; }
}

/**
 * Check if a vector contains a zero
 * 
 * @Ad, device array of length n
 * @n, the size
 * @returns true if all entries are zero, false otherwise.
 *
 **/
bool has_zero_in_vector(const sfixn *Ad, sfixn n) {
    const int nthds = 256;
    sfixn nb = (n / nthds) + ((n % nthds) ? 1 : 0);
    // Note that nb might not be a power of 2!
    // We factor nb approximately.
    sfixn nx, ny, found;
    found = approx_factor(nb, &nx, &ny, (1L << 16));
    if (DEBUG) assert(found);
    dim3 nBlk(nx, ny);
    sfixn ret = -1, *retd;
    cudaMalloc((void**)&retd, sizeof(sfixn));
    cudaMemcpy(retd, &ret, sizeof(sfixn), cudaMemcpyHostToDevice);
    has_zero_ker<nthds><<<nBlk, nthds>>>(retd, Ad, n);
    cudaMemcpy(&ret, retd, sizeof(sfixn), cudaMemcpyDeviceToHost);
    cudaFree(retd);
    return (ret >= 0);

    //sfixn *A = new sfixn[n];
    //cudaMemcpy(A, Ad, n*sizeof(sfixn), cudaMemcpyDeviceToHost);
    //for (sfixn i = 0; i < n; ++i) { 
    //    if (A[i] == 0) {
    //        delete [] A;
    //        return true;
    //    }
    //}
    //delete [] A;
    //return false;
}

////////////////////////////////////////////////////////////////////////////////

/*
 * Classic polynomial multiplication for univariate polynomials
 * 
 * Assume dH = dF + dG
 * 
 **/
void plain_polymul_uni_host(sfixn dH, sfixn *H, sfixn dF, 
    const sfixn *F, sfixn dG, const sfixn *G, sfixn p) 
{
    if (DEBUG) assert(dH == dF + dG);
    printf("dH = %d, dF = %d, dG = %d\n", dH, dF, dG);
    for(sfixn i = 0; i <= dF; ++i) {
        if (!F[i]) continue;
        if (F[i] == 1) {
            for(sfixn j = 0; j <= dG; ++j) 
                H[i + j] = add_mod(G[j], H[i + j], p);
        } else {
            for(sfixn j = 0; j <= dG; ++j) 
                H[i + j] = add_mod(mul_mod(G[j], F[i], p), H[i + j], p);
        }
    }
}

/**
 * Check if two univariate polynomials are the same.
 */
bool is_the_same_poly_host(sfixn d, const sfixn *H, const sfixn *G) {
    if (DEBUG) assert(d >= 0);
    while (d >= 0) {
        if (H[d] != G[d]) return false;
        --d;
    }
    return true;
}

/**
 * Check polynomial multiplications
 */
void check_polymul(sfixn d1, sfixn d2, sfixn p) 
{
    const sfixn SMALL = 1000;

    sfixn *F = new sfixn[d1 + 1];
    sfixn *G = new sfixn[d2 + 1];
    for (sfixn i = 0; i <= d1; ++i) { F[i] = 1; }
    for (sfixn i = 0; i <= d2; ++i) { G[i] = i; }

    if (d1 + d2 < SMALL) {
        sfixn *H1 = new sfixn[d1 + d2 + 1]();
        sfixn *H2 = new sfixn[d1 + d2 + 1]();

        plain_polymul_uni_host(d1 + d2, H1, d1, F, d2, G, p); 
        stockham_poly_mul_host(d1 + d2, H2, d1, F, d2, G, p);
        
        if (is_the_same_poly_host(d1 + d2, H1, H2)) {
            printf("The multiplication is correct!\n");
        } else {
            printf("The multiplication is NOT correct!\n");
        }

        delete [] H1;
        delete [] H2;
    } else {
        sfixn *H0 = new sfixn[d1 + d2 + 1]();
        stockham_poly_mul_host(d1 + d2, H0, d1, F, d2, G, p);

        sfixn e = ceiling_log2(d1 + d2 + 1);
        sfixn n = (1L << e);
        sfixn wn = primitive_root(e, p);
        sfixn invwn = inv_mod(wn, p);
        sfixn invn = inv_mod(n, p);

        sfixn *F0 = new sfixn[n]();
        sfixn *G0 = new sfixn[n]();
        for (sfixn i = 0; i <= d1; ++i) { F0[i] = F[i]; }
        for (sfixn i = 0; i <= d2; ++i) { G0[i] = G[i]; }
        gsnn_fft_ip(n, e, F0, p, wn);
        gsnn_fft_ip(n, e, G0, p, wn);
        for (sfixn i = 0; i < n; ++i) { F0[i] = mul_mod(F0[i], G0[i], p); }
        gsnn_fft_ip(n, e, F0, p, invwn);
        for (sfixn i = 0; i < n; ++i) { F0[i] = mul_mod(F0[i], invn, p); }
        
        if (is_the_same_poly_host(d1 + d2, F0, H0)) {
            printf("The multiplication is correct!\n");
        } else {
            printf("The multiplication is NOT correct!\n");
        }

        delete [] H0;
        delete [] F0;
        delete [] G0;
    }

    delete [] F;
    delete [] G;
}

////////////////////////////////////////////////////////////////////////////////
void test_expand_to_list_fft2_dev() {
    sfixn nx = 3;
    sfixn ny = 3;
    sfixn ex2 = 2;
    sfixn ey2 = 2;
    sfixn nx2 = (1 << ex2);
    sfixn ny2 = (1 << ey2);
     
    sfixn *X = new sfixn[nx * ny]();
    sfixn *Y = new sfixn[nx2 * ny2]();
    for (int i = 0; i < nx*ny; ++i) X[i] = i;
    print_matrix(nx, ny, X);
    
    sfixn *X_d, *Y_d;
    cudaMalloc((void**)&X_d, sizeof(sfixn)*nx*ny);    
    cudaMalloc((void**)&Y_d, sizeof(sfixn)*nx2*ny2);    
    cudaMemcpy(X_d, X, sizeof(sfixn)*nx*ny, cudaMemcpyHostToDevice);
    cudaMemcpy(Y_d, Y, sizeof(sfixn)*nx2*ny2, cudaMemcpyHostToDevice);
    expand_to_list_fft2_dev(1, ex2, ey2, Y_d, nx, ny, X_d);
    cudaMemcpy(Y, Y_d, sizeof(sfixn)*nx2*ny2, cudaMemcpyDeviceToHost);
    print_matrix(nx2, ny2, Y);

    cudaFree(X_d);
    cudaFree(Y_d);
    delete [] X;
    delete [] Y;
}

void test_expand_to_fft2_dev() {
    sfixn nx = 3;
    sfixn ny = 3;
    sfixn ex2 = 2;
    sfixn ey2 = 2;
    sfixn nx2 = (1 << ex2);
    sfixn ny2 = (1 << ey2);
     
    sfixn *X = new sfixn[nx * ny]();
    sfixn *Y = new sfixn[nx2 * ny2]();
    for (int i = 0; i < nx*ny; ++i) X[i] = i;
    print_matrix(nx, ny, X);
    
    sfixn *X_d, *Y_d;
    cudaMalloc((void**)&X_d, sizeof(sfixn)*nx*ny);    
    cudaMalloc((void**)&Y_d, sizeof(sfixn)*nx2*ny2);    
    cudaMemcpy(X_d, X, sizeof(sfixn)*nx*ny, cudaMemcpyHostToDevice);
    expand_to_fft2_dev(ex2, ey2, Y_d, nx, ny, X_d);
    cudaMemcpy(Y, Y_d, sizeof(sfixn)*nx2*ny2, cudaMemcpyDeviceToHost);

    print_matrix(nx2, ny2, Y);

    cudaFree(X_d);
    cudaFree(Y_d);
    delete [] X;
    delete [] Y;
}

void test_transpose() {
    sfixn nx = 4;
    sfixn ny = 24;
     
    sfixn *X = new sfixn[nx * ny]();
    sfixn *Y = new sfixn[nx * ny]();
    for (int i = 0; i < nx*ny; ++i) X[i] = i;
    print_matrix(nx, ny, X);
    
    sfixn *X_d, *Y_d;
    cudaMalloc((void**)&X_d, sizeof(sfixn)*nx*ny);    
    cudaMalloc((void**)&Y_d, sizeof(sfixn)*nx*ny);    
    cudaMemcpy(X_d, X, sizeof(sfixn)*nx*ny, cudaMemcpyHostToDevice);
    transpose_dev(Y_d, X_d, nx, ny);
    cudaMemcpy(Y, Y_d, sizeof(sfixn)*nx*ny, cudaMemcpyDeviceToHost);
    print_matrix(ny, nx, Y);

    cudaFree(X_d);
    cudaFree(Y_d);
    delete [] X;
    delete [] Y;
}
