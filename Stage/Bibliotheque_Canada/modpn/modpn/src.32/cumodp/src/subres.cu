////////////////////////////////////////////////////////////////////////////////
#include "../include/subres.h"

/**
 * This implements the subresultant chain computation of two univariate
 * polynomials over a finite field. The subresultant chain is represented 
 * by a plain array of integers. 
 *
 * [[CASE 1]]
 *
 * The array size is w^2, where w is the smaller degree of the input 
 * polynomials. Hence, the subresultant chain is a square, that is,
 *
 * S = [ S_{w-1}, S_{w-2}, ... , S_{0} ].
 *
 * and each S_{i} is of size w. The advantages of this encoding are:
 * - Index computations are straightforward. 
 * - No extra workspace needed; all the computations can be done in-place. 
 *   This implies less data movements. 
 *
 * The main disadvantage is: the size of the subresultant chain is w^2,
 * comparing to the size acutally needed w*(w+1)/2. One possible way is to
 * compact the data after computed the subresultant chain.
 *
 * [[CASE 2]]
 *
 * The array size is w(w+1)/2, and data in S are organized as 
 * 
 * S = [S_{w-1}, ...., S_0], 
 *
 * where S_i is the ith-subresultant of P and Q. Each S_i is a vector of 
 * size i+1, encoding a univariate polynomial of degree at most i. 
 *
 * For both cases, the Brown's algorthm is used to compute the subresultant 
 * chain of two univariate polynomials. No optimizations of Docus have been 
 * applied, which are not suitable for our setting.
 *
 * Pseudo code in Maple syntax : 
 *
 * Input  : P, Q in F[X] such that deg(P) >= deg(Q) > 0.
 * Output : List of subresultants of P and Q.
 *
 * S := [];
 * s := lcoeff(Q)^(degree(P)-degree(Q));
 * A := Q;
 * B := prem(P, -Q, X) 
 *
 * ASSERT(degree(P)>=degree(Q));
 *
 * do
 *    d := degree(A);
 *    e := degree(B); 
 *    
 *    if B = 0 then return S; end if;
 *    S := [B, op(S)];
 *    delta := d - e;
 *    if delta > 1 then
 *       C := (lcoeff(B)/s)^(delta-1)*B;
 *       S := [C, op(S)];
 *    else
 *       C := B;
 *    end if;
 *    if e = 0 then return S; end if;
 *    B := prem(A, -B, X)/(s^delta*lcoeff(A));
 *
 *    A := C;
 *    s := lcoeff(A);
 * end do;
 *
 */

////////////////////////////////////////////////////////////////////////////////

/**
 * Compute the subresultant chain of two univariate polynomials
 *
 * @Ssz : the size of S 
 * @S   : (Output) vector encoding the subresultant chain   
 * @dgP : degree of P
 * @P   : coefficient vector of P
 * @dgQ : degree of Q
 * @Q   : coefficient vector of Q
 * @p   : Fourier prime 
 *
 * Assumptions :
 *
 * (0) deg(P) >= deg(Q) = w > 0 
 * (1) Q[dgQ] != 0 and P[dgP] != 0
 * (2) Ssz is at least w^2
 * (3) Initial values in S are 0
 * (4) Ssz >= dgP + 1, this is due to the fact that we need to perform 
 *     a pseudo division computation using S as a workspace.
 *
 * Output :
 *  
 * Data stored in S are organized as S = [S_{w-1}, ...., S_0], where S_i is
 * the ith-subresultant of P and Q. Each S_i is a vector of size w, encoding
 * a univariate polynomial of degree at most i. 
 *
 * S[0]        ..  S[w-1]        ===> i = w-1, size w, S_{w-1}
 * S[w]        ..  S[2w-1]       ===> i = w-2, size w, S_{w-2}
 * S[2w]       ..  S[3w-1]       ===> i = w-3, size w, S_{w-3}
 * 
 * S[(w-i-1)w] ..  S[(w-i)w-1]   ===> i = i  , size w, S_{i}
 *
 * S[w^2-w]    ..  S[w^2-1]      ===> i = 0  , size w, S_{0}
 *
 * For example, when dgP = 5, deQ = 4, we have w = 4 and Ssz = 16.
 *
 * Inside S, the data are
 *
 * S00  S01  S02  S03
 * S10  S11  S12  0
 * S20  S21  0    0 
 * S30  0    0    0
 *
 * which represents subresultants
 *
 * S3 = S00 + S01*Y + S02*Y^2 + S03*Y^3
 * S2 = S10 + S11*Y + S12*Y^2
 * S1 = S20 + S21*Y 
 * S0 = S30
 *
 * Should NOT be used due to the data corruption.
 **/
__device__ void 
subres_chain_uni(sfixn *S, sfixn dgP, const sfixn *P, 
    sfixn dgQ, const sfixn *Q, sfixn p, double pinv)
{ 
    sfixn s, w, d, e, delta, c;
    sfixn *B, *C, *T;
    const sfixn *A; 

    // s = lc(Q)^(deg(P)-deg(Q))
    s = pow_mod(Q[dgQ], dgP - dgQ, p, pinv);
    w = dgQ;

    // Starting pair (A, B)
    A = Q;
    d = dgQ;
    // Compute B := prem(P, -Q) = S_{w-1}
    // The assumption dgP + 1 <= Ssz is used in the following function call.
    // This is a fatal problem for this implementation, and
    // data got corrupt during the computation!
    B = S; 
    negate_prem_poly_uni_ip(dgP, P, dgQ, Q, B, p);
    e = shrink_deg_uni(w - 1, B);
    
    // Starting with S_{w-1}
    while (true) {
        // A ~ S_d if d = deg(Q) 
        // A = S_d if d < deg(Q)
        // B = S_{d-1}, s = lc(S_d) for d <= deg(Q)
         
        // if B = 0, then done.
        if (e == 0 && B[0] == 0) { return; }
        delta = d - e;

        if (delta > 1) {
            // C := (lcoeff(B)/s)^(delta-1)*B;
            c = pow_mod(quo_mod(B[e], s, p), delta - 1, p);
            // Compute the starting address of C = S_e
            C = S + (w - e - 1) * w;
            // C has the same degree as B, C = c * B
            scalar_mul_poly_uni(c, e, C, B, p);
        } else {
            C = B;
        }
        // if B is a nonzero constant, then done.
        if (e == 0) return;
        
        // From now on, e >= 1 and compute the next subresultant S_{e-1}
        // T = prem(A, -B)/(s^delta*lc(A));
        // T is assigned to the slot S_{e - 1}.
        T = S + (w - e) * w;
        // T has size at least d + 1
        negate_prem_poly_uni_ip(d, A, e, B, T, p);
        c = inv_mod(mul_mod(pow_mod(s, delta, p), A[d], p), p);
        scalar_mul_poly_uni_ip(c, e - 1, T, p);

        // move A to the next regular subresultant C = S_e
        A = C, d = e, s = C[e];
        // move B to S_{e-1}
        B = T, e = shrink_deg_uni(e - 1, T);
    }
}

/**
 * Compute the sub-resultant chain of two univariate polynomials
 *
 * @Ssz : the size of S 
 * @S   : (Output) vector encoding the subresultant chain   
 * @dgP : degree of P
 * @P   : coefficient vector of P
 * @dgQ : degree of Q
 * @Q   : coefficient vector of Q
 * @p   : Fourier prime
 * @W   : work space of length at least dgP + 1
 *
 * Assumptions :
 *
 * (0) deg(P) >= deg(Q) = w > 0 
 * (1) Q[dgQ] != 0 and P[dgP] != 0
 * (2) Ssz is at least (w^2+w)/2
 * (3) Initial values in S are 0
 *
 * Output :
 *  
 * Data stored in S are organized as S = [S_{w-1}, ...., S_0], where S_i is
 * the ith-subresultant of P and Q. Each S_i is a vector of size i+1, encoding
 * a univariate polynomial of degree at most i. 
 *
 * S[0]     ..   S[w-1]     ===> j = 0, i = w - 1, size w,   S_{w-1}
 * S[w]     ..   S[2w-2]    ===> j = 1, i = w - 2, size w-1, S_{w-2}
 * S[2w-1]  ..   S[3w-4]    ===> j = 2, i = w - 3, size w-2, S_{w-3}
 * 
 * S[ (w + i + 2)(w - i - 1)/2 ] .. S[ (w + i + 2)(w - i - 1)/2 + i ]  
 * 
 * or 
 *
 * S[ (2 * w - j + 1) * j / 2 ] .. S[ (2 * w - j + 1) * j / 2 + i ]  
 *
 * ===> j = w - i - 1, i, size i + 1 = w - j, S_{i}
 *
 * S[(w^2 + w)/2 - 1]       ===> j = w - 1, i = 0, size 1, S_{0}
 *
 * For example, when dgP = 5, deQ = 4, we have w = 4 and Ssz = 10.
 *
 * Inside S, the data are
 *
 * S00  S01  S02  S03
 * S10  S11  S12 
 * S20  S21
 * S30
 *
 * which represents subresultants
 *
 * S3 = S00 + S01*Y + S02*Y^2 + S03*Y^3
 * S2 = S10 + S11*Y + S12*Y^2
 * S1 = S20 + S21*Y 
 * S0 = S30
 *
 **/
__device__ __host__ void 
subres_chain_uni_tri(sfixn *S, sfixn dgP, const sfixn *P, sfixn dgQ, 
    const sfixn *Q, sfixn p, double pinv, sfixn *W)
{ 
    sfixn s, w, d, e, delta, c;
    sfixn *B, *C, *T;
    const sfixn *A; 

    // s = lc(Q)^(deg(P)-deg(Q))
    s = pow_mod(Q[dgQ], dgP - dgQ, p, pinv);
    w = dgQ;

    // Starting pair (A, B)
    A = Q;
    d = dgQ;
    // Compute B := prem(P, -Q)
    B = S; 
    negate_prem_poly_uni(dgP, P, dgQ, Q, B, p, W);
    e = shrink_deg_uni(w - 1, B);

    // Starting with S_{w-1}
    while (true) {
        // A ~ S_d if d = deg(Q) 
        // A = S_d if d < deg(Q)
        // B = S_{d-1}, s = lc(S_d) for d <= deg(Q)
         
        // if B = 0, then done.
        if (e == 0 && B[0] == 0) { return; }

        delta = d - e;
        if (delta > 1) {
            // C := (lcoeff(B)/s)^(delta-1)*B;
            c = pow_mod(quo_mod(B[e], s, p, pinv), delta - 1, p, pinv);
            // Compute the starting address of C = S_e
            C = S + (w + e + 2) * (w - e - 1) / 2;
            // C has the same degree as B.
            scalar_mul_poly_uni(c, e, C, B, p);
        } else {
            C = B;
        }
        // if B is a nonzero constant, then done.
        if (e == 0) return;
        
        // From now on, e >= 1.
        // compute the next subresultant S_{e-1}
        // T = prem(A, -B)/(s^delta*lc(A));
        // T is assigned to the slot S_{e-1}.
        T = S + (w + e + 1) * (w - e) / 2;
        negate_prem_poly_uni(d, A, e, B, T, p, W);
        c = inv_mod(mul_mod(pow_mod(s, delta, p, pinv), A[d], p, pinv), p);
        scalar_mul_poly_uni_ip(c, e - 1, T, p);

        // move A to the next regular subresultant C = S_e
        A = C, d = e, s = C[e];
        // move B to S_{e-1}
        B = T, e = shrink_deg_uni(e-1, T);
    }
}

/**
 * Compute the subresultant chain of two bivariate polynomials P(x, y) and
 * Q(x, y) by FFT evaluation and interpolation:
 *
 * (0) Compute the degree bound B = 2^k of the resultant in x;
 *
 * (1) Find a B-th primitive root w of unity which does not cancel any 
 *     leading coefficients. (Assume that P and Q have been processed);
 *
 * (2) Evaluate coefficients of P by a call to list_fft;
 *
 * (3) Evaluate coefficients of Q by a call to list_fft;
 *
 * (4) Compute the subresultant chain for each evaluation point;
 *
 * (5) Construct the data structure SCUBE. 
 *
 * For example, let 
 *
 * P = A(x) + B(x)*y + C(x)*y^2 + D(x)*y^3  and Q = E(x) + F(x)*y + G(x)*y^2
 * 
 * Assume the degree bound of the resultant is 8 (could be larger).
 *
 * After evaluating P at [1, w, w^2, w^3, w^4, w^5, w^6, w^7], the layout of 
 * the evaluation data is 
 *
 * A(1) A(w) A(w^2) A(w^3) A(w^4) A(w^5) A(w^6) A(w^7)
 * B(1) B(w) B(w^2) B(w^3) B(w^4) B(w^5) B(w^6) B(w^7)
 * C(1) C(w) C(w^2) C(w^3) C(w^4) C(w^5) C(w^6) C(w^7)
 * D(1) D(w) D(w^2) D(w^3) D(w^4) D(w^5) D(w^6) D(w^7)
 *                                                         (I)
 * E(1) E(w) E(w^2) E(w^3) E(w^4) E(w^5) E(w^6) E(w^7)
 * F(1) F(w) F(w^2) F(w^3) F(w^4) F(w^5) F(w^6) F(w^7)
 * G(1) G(w) G(w^2) G(w^3) G(w^4) G(w^5) G(w^6) G(w^7)
 *
 * To run the subresultant chain algorithm for each evaluation point, 
 * we might need to transpose data as follows.
 *
 * A(1)   B(1)   C(1)   D(1)        E(1)   F(1)   G(1)  
 * A(w)   B(w)   C(w)   D(w)        E(w)   F(w)   G(w)  
 * A(w^2) B(w^2) C(w^2) D(w^2)      E(w^2) F(w^2) G(w^2)
 * A(w^3) B(w^3) C(w^3) D(w^3)      E(w^3) F(w^3) G(w^3)
 * A(w^4) B(w^4) C(w^4) D(w^4)      E(w^4) F(w^4) G(w^4)    (II)
 * A(w^5) B(w^5) C(w^5) D(w^5)      E(w^5) F(w^5) G(w^5)
 * A(w^6) B(w^6) C(w^6) D(w^6)      E(w^6) F(w^6) G(w^6)
 * A(w^7) B(w^7) C(w^7) D(w^7)      E(w^7) F(w^7) G(w^7)
 *
 * The subresultant chain in compact form has the following layout
 *
 * x =   1, S10 S11 S00
 * x =   w, S10 S11 S00
 * x = w^2, S10 S11 S00
 * x = w^3, S10 S11 S00
 * x = w^4, S10 S11 S00
 * x = w^5, S10 S11 S00
 * x = w^6, S10 S11 S00
 * x = w^7, S10 S11 S00
 */

/**
 * @S, subresultant chain data
 * @B, the total number of evaluation points
 * @m, m = deg(P, y)
 * @n, n = deg(Q, y), m >= n > 0
 * @p, prime number
 * @sz, the size of each subresultant chain
 *
 * The length of S is B * Ssz. The total number of threads is B. 
 * Each thread run a subres chain algorithm.
 */
__global__ void 
coarse_subres_uni_ker(sfixn *S, sfixn sz, sfixn dgP, const sfixn *P, 
    sfixn dgQ, const sfixn *Q, sfixn p, double pinv, sfixn *W)
{
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn tid = bid * blockDim.x + threadIdx.x;

    subres_chain_uni_tri(S + tid * sz, dgP, P + tid * (dgP + 1), dgQ, 
        Q + tid * (dgQ + 1), p, pinv, W + tid * (dgP + 1));
}

bool coarse_subres_uni_dev(sfixn *S, sfixn B, sfixn dgP, 
    const sfixn *P, sfixn dgQ, const sfixn *Q, sfixn p) 
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    if (DEBUG) assert(dgP >= dgQ && dgQ > 0);
    double pinv = 1 / (double)p;
    sfixn nb = B / nthds;
    if (DEBUG) assert(nb >= 1);
    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 15) == 0) {
        nBlks.x = (sfixn(1) << 15);
        nBlks.y = (nb >> 15);
    }   

    // work space of size (dgP + 1) * B
    sfixn *W;
    cudaError_t err = cudaMalloc((void**)&W, sizeof(sfixn)*(dgP + 1)*B);
    if (err != cudaSuccess) {
        fprintf(stderr, "coarse_subres_uni_dev :: out of global memory\n");
        return false;
    }

    sfixn sz = dgQ * (dgQ + 1) / 2;
    coarse_subres_uni_ker<<<nBlks, nthds>>>(S, sz, dgP, P, dgQ, Q, p, pinv, W);
    cudaThreadSynchronize();
    cudaFree(W);
    return true;
}

/**
 * Compute subresultant chains subres(P, Q, y).
 *
 * @S   : (Output) the subresultant chain, host array
 * @B   : the degree bound of the resultant
 * @w   : B-th primitive root of unity modulo p
 * @P   : bivariate rdr-poly over Z_p[x, y], host array
 * @Q   : bivaraite rdr-poly over Z_p[x, y], host array
 * @npx : the partial size of P in x
 * @npy : the partial size of P in y
 * @nqx : the partial size of Q in x
 * @nqy : the partial size of Q in y
 *
 * Assume that dpy >= dqy holds and S is of size B * (nqy - 1) * nqy / 2.
 * Assume that none of initials of P and Q are zero on the powers of w.
 */
sfixn subres_chain2_coarse_host(sfixn *S, sfixn B, sfixn w, sfixn npx, sfixn npy,
    const sfixn *P, sfixn nqx, sfixn nqy, const sfixn *Q, sfixn p)
{
    sfixn bpow = ceiling_log2(B);

    sfixn *W;
    cudaMalloc((void **)&W, sizeof(sfixn)<<(bpow - 1));
    get_powers_binary(bpow - 1, W, w, p); 

    // Step (2) Evaluate P at powers of w.
    sfixn *P_d, *P2_d;
    cudaMalloc((void**)&P_d, sizeof(sfixn) * B * npy);
    cudaMemcpy(P_d, P, sizeof(sfixn) * nqx * nqy, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&P2_d, sizeof(sfixn) * B * npy);
    expand_to_list_fft_dev(npy, B, bpow, P2_d, npx, P_d);
    // now P2_d holds the evaluated data, layout (I)
    list_stockham_dev(P2_d, npy, bpow, W, p);
    // now P_d holds the transposed data, layout (II)
    transpose_dev(P_d, P2_d, B, npy);
    // free P2_d
    cudaFree(P2_d);
             
    // Step (3) Evaluate Q at powers of w.
    sfixn *Q_d, *Q2_d;
    cudaMalloc((void**)&Q_d, sizeof(sfixn) * nqy * B);
    cudaMemcpy(Q_d, Q, sizeof(sfixn) * nqx * nqy, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Q2_d, sizeof(sfixn) * nqy * B);
    expand_to_list_fft_dev(nqy, B, bpow, Q2_d, nqx, Q_d);
    // Q2_d holds the evaluated data, layout (I)
    list_stockham_dev(Q2_d, nqy, bpow, W, p);
    // Q_d holds the transposed data, layout (II)
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
    bool ret = coarse_subres_uni_dev(Sd, B, npy - 1, P_d, nqy - 1, Q_d, p);
    cudaFree(P_d);
    cudaFree(Q_d);

    // Step (5) Update data fields
    if (!ret) { cudaFree(Sd); return 1; }
    cudaMemcpy(S, Sd, sizeof(sfixn) * B * sz , cudaMemcpyDeviceToHost);
    cudaFree(Sd);
    if (DEBUG) checkCudaError("subres_chain2_coarse_host");
    return 0;
}


/**
 * Compute subresultant chains subres(P, Q, z).
 *
 * @S   : (Output) the subresultant chain, host array
 * @Bx  : the degree bound of the resultant in x
 * @By  : the degree bound of the resultant in y
 * @wx  : Bx-th primitive root of unity modulo p
 * @wy  : By-th primitive root of unity modulo p
 * @P   : trivariate rdr-poly over Z_p[x, y, z], host array
 * @Q   : trivariate rdr-poly over Z_p[x, y, z], host array
 * @npx : the partial size of P in x
 * @npy : the partial size of P in y
 * @npz : the partial size of P in z
 * @nqx : the partial size of Q in x
 * @nqy : the partial size of Q in y
 * @nqz : the partial size of Q in z
 *
 * Assume that dpz >= dqz holds and S is of size Bx * By * (nqz - 1) * nqz / 2.
 * Assume that none of initials of P and Q are zero on the powers of wx, wy.
 *
 **/
sfixn subres_chain3_coarse_host(sfixn *S, sfixn Bx, sfixn By, sfixn wx, sfixn wy, 
    sfixn npx, sfixn npy, sfixn npz, const sfixn *P,
    sfixn nqx, sfixn nqy, sfixn nqz, const sfixn *Q, sfixn p)
{
    sfixn bxpow = ceiling_log2(Bx);
    sfixn bypow = ceiling_log2(By);

    // Step (1) Precompute B-th primitive root of unity
    sfixn *Wx;
    cudaMalloc((void **)&Wx, sizeof(sfixn)<<(bxpow - 1));
    get_powers_binary(bxpow - 1, Wx, wx, p); 

    sfixn *Wy;
    cudaMalloc((void **)&Wy, sizeof(sfixn)<<(bypow - 1));
    get_powers_binary(bypow - 1, Wy, wy, p); 

    // Step (2) Evaluate P at powers of wx and wy
    sfixn *P_d, *P2_d;
    cudaMalloc((void**)&P_d, sizeof(sfixn) * (npz << (bxpow + bypow)));
    cudaMemcpy(P_d, P, sizeof(sfixn) * npx * npy * npz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&P2_d, sizeof(sfixn) * (npz << (bxpow + bypow)));
    expand_to_list_fft2_dev(npz, bxpow, bypow, P2_d, npx, npy, P_d);

    //printDeviceMatrix(P2_d, Bx, By);

    // now P2_d holds the evaluated data, layout (I)
    list_bivariate_stockham_dev(P2_d, npz, bxpow, Wx, bypow, Wy, p);
    // now P_d holds the transposed data, layout (II)
    transpose_dev(P_d, P2_d, Bx * By, npz);
    // free P2_d
    cudaFree(P2_d);
    
    // Step (3) Evaluate Q at powers of w.
    sfixn *Q_d, *Q2_d;
    cudaMalloc((void**)&Q_d, sizeof(sfixn) * (nqz << (bxpow + bypow)));
    cudaMemcpy(Q_d, Q, sizeof(sfixn) * nqx * nqy * nqz, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Q2_d, sizeof(sfixn) * (nqz << (bxpow + bypow)));
    expand_to_list_fft2_dev(nqz, bxpow, bypow, Q2_d, nqx, nqy, Q_d);
   
    // now Q2_d holds the evaluated data, layout (I)
    list_bivariate_stockham_dev(Q2_d, nqz, bxpow, Wx, bypow, Wy, p);
    // now Q_d holds the transposed data, layout (II)
    transpose_dev(Q_d, Q2_d, Bx * By, nqz);
    // free Q2_d, Wx, Wy
    cudaFree(Q2_d);
    cudaFree(Wx);
    cudaFree(Wy);
    
    // Step (4) Compute subres chain for each evaluation point
    sfixn *Sd;
    sfixn sz = (nqz * nqz - nqz) / 2;
    cudaError_t err = cudaMalloc((void **)&Sd, sizeof(sfixn) * Bx * By * sz);
    if (err != cudaSuccess) {
        cudaFree(P_d);
        cudaFree(Q_d);
        return 1;
    }

    // Step (5) Update data fields S
    reset_vector_dev(sz * Bx * By, Sd);
    bool ret = coarse_subres_uni_dev(Sd, Bx * By, npz - 1, P_d, nqz - 1, Q_d, p);
    cudaFree(P_d);
    cudaFree(Q_d);
    if (!ret) { cudaFree(Sd); return 1; }

    cudaMemcpy(S, Sd, sizeof(sfixn) * Bx * By * sz, cudaMemcpyDeviceToHost);
    cudaFree(Sd);
    if (DEBUG) checkCudaError("subres_chain3_coarse_host");
    return 0;
}

/***************************** END_OF_FILE ************************************/
