#include "../include/scube.h"

// constructor for bivariate or trivariate
scube_t::scube_t(const rdr_poly &P, const rdr_poly &Q, sfixn p) 
    : fp(p), lyt(UNKNOWN) 
{
    assert(P.num_of_vars() == Q.num_of_vars());
    assert(P.num_of_vars() == 2 || P.num_of_vars() == 3);
    if (P.num_of_vars() == 2) {
        init2(P.partial_size(0), P.partial_size(1), 
              Q.partial_size(0), Q.partial_size(1));
    } else {
        init3(P.partial_size(0), P.partial_size(1), P.partial_size(2),
              Q.partial_size(0), Q.partial_size(1), Q.partial_size(2));
    }
}

void scube_t::info() const {
    printf("##-------------------------------------------------\n");
    printf("##          fourier prime %d\n", fp);
    printf("##    number of variables %d\n", nvars);
    printf("##            high degree %d\n", hdeg);
    printf("##             low degree %d\n", ldeg);
    switch (lyt) {
        case FINE   : 
            printf("##            data layout fine-grained\n"); 
            break;
        case COARSE : 
            printf("##            data layout coarse-grained\n"); 
            break;
        case UNKNOWN: 
            printf("##            data layout uninitialized\n"); 
            break;
        default     : 
            printf("##            data layout error\n");
    }
    printf("##     cube size in words %d\n", cube_size());
    printf("## cube size in megabytes %.2f\n", float(cube_size())/(1L<<18));
    printf("##       coefficient size %d\n", coefficient_size());
    printf("##        fft size vector ");
    for (int i = 0; i < nvars - 1; ++i) printf("%d ", bound_es[i]);
    printf("\n");
    printf("##     root size in words %d\n", root_size());
    printf("##number of cached coeffs %d\n", coeffs.size());
    printf("##   translations applied %d\n", is_translated());
    printf("##-------------------------------------------------\n");
}

void scube_t::print_image(sfixn i) const {
    sfixn *Td;
    cudaMalloc((void**)&Td, sizeof(sfixn)*ldeg);
    for (sfixn j = ldeg - 1; j >= 0; --j) {
        const sfixn *Sj = ith_image_jth_subres(i, j); 
        for (int v = 0; v <= j; ++v) { printf("%8d  ", Sj[v]); }
        printf("\n");
    }
    cudaFree(Td);
}

void scube_t::init2(int npx, int npy, int nqx, int nqy) 
{
    if (nqy > npy) {
        init2(nqx, nqy, npx, npy);
    } else {
        nvars = 2;
        bound_es = new sfixn[nvars - 1];
        translator = new sfixn[nvars - 1]();
        hdeg = npy - 1;
        ldeg = nqy - 1;

        // FFT size in the first variable x
        sfixn d1 = npx - 1;
        sfixn d2 = nqx - 1;
        sfixn B = hdeg * d2 + ldeg * d1 + 1;
        bound_es[0] = ceiling_log2(B);
        coeff_size = sfixn(1) << bound_es[0];
        check_fourier_degree();
    }
}

void scube_t::init3(int npx, int npy, int npz, int nqx, int nqy, int nqz) 
{
    if (nqz > npz) {
        init3(nqx, nqy, nqz, npx, npy, npz);
    } else {
        nvars = 3;
        bound_es = new sfixn[nvars - 1];
        translator = new sfixn[nvars - 1]();

        hdeg = npz - 1;
        ldeg = nqz - 1;
        
        // FFT size in the first variable x
        sfixn d1 = npx - 1;
        sfixn d2 = nqx - 1;
        sfixn B = hdeg * d2 + ldeg * d1 + 1;
        bound_es[0] = ceiling_log2(B);

        // FFT size in the second variable y
        d1 = npy - 1;
        d2 = nqy - 1;
        B = hdeg * d2 + ldeg * d1 + 1;
        bound_es[1] = ceiling_log2(B);
        coeff_size = sfixn(1) << (bound_es[0] + bound_es[1]);
        check_fourier_degree();
    }
}

// return the address of jth subresultant of the ith image
const sfixn* scube_t::ith_image_jth_subres(sfixn i, sfixn j) const {
    if (i < 0 || i >= coefficient_size() || j < 0 || j >= ldeg) return NULL;
    if (lyt == FINE) {
        return Sd + coefficient_size() * (2 * ldeg + j + 2) * (ldeg - j - 1) / 2 
                + i * (j + 1);
    } else if (lyt == COARSE) {
        return Sd + i * (ldeg + 1) * ldeg / 2 
                + (2 * ldeg + j + 2) * (ldeg - j - 1) / 2;
    }
    return NULL;
}

void scube_t::free_subres_coeffs() {
    subres_coeff_iterator it = coeffs.begin(); 
    while (it != coeffs.end()) {
        delete [] it->second;
        it->second = NULL;
        ++it;
    }
}

///////////////////////////////////////////////////////////////////////////////
// 
// The computation of subresultant chains is split into several stages.
//
// (1) Generate valid FFT grid by means of FFTs
// (2) Construct scube
// (3) Interoplate coefficients 
//
// WP  Tue Nov 23 13:07:16 EST 2010
//
///////////////////////////////////////////////////////////////////////////////

/**
 * Check if a FFT grid is valid in the sense that both initials of P and Q 
 * are nonzero at any power of primitive roots of unities. This function
 * only checks a grid is valid or not. Generating a grid is the task for a 
 * seperate function.
 *
 * @initP, the leading coefficient of P
 * @initQ, the leading coefficient of Q
 *
 * @return, true if the FFT grid is valid, false otherwise.
 */
bool scube_t::is_valid_fft_grid2(sfixn npx, const sfixn *initP, 
    sfixn nqx, const sfixn *initQ)
{
    sfixn k = get_bounds_exp(0);
    sfixn n = sfixn(1) << k;
    
    // primitive roots computed inside the device
    const sfixn *W = get_roots(0);

    // check if initP is valid or not
    sfixn *A_d;
    cudaMalloc((void **)&A_d, sizeof(sfixn) * n);
    cudaMemcpy(A_d, initP, sizeof(sfixn) * npx, cudaMemcpyHostToDevice);
    expand_to_fft_dev(n, npx, A_d);
    stockham_dev(A_d, n, k, W, fp);

    if (has_zero_in_vector(A_d, n)) { 
        cudaFree(A_d); 
        return false; 
    }

    // check if initQ is valid or not
    cudaMemcpy(A_d, initQ, sizeof(sfixn) * nqx, cudaMemcpyHostToDevice);
    expand_to_fft_dev(n, nqx, A_d);
    stockham_dev(A_d, n, k, W, fp);

    bool ret = has_zero_in_vector(A_d, n);
    // printDeviceVariable(W + 1);
    cudaFree(A_d);

    if (DEBUG) checkCudaError("error found in is_valid_fft_grid2");
    return (ret == false);
}

/**
 * initP represents a bivariate polynomial of partial sizes npx and npy
 * initQ represents a bivariate polynomial of partial sizes nqx and nqy
 * 
 * Example: Let npx = 5, npy = 3 then initP has the following data
 *
 *  0  1  2  3  4 
 *  5  6  7  8  9 
 * 10 11 12 13 14 
 * 
 * After expanding initP into FFT size 8 in x and 4 in y, the data is
 * 
 *  0  1  2  3  4  0  0  0 
 *  5  6  7  8  9  0  0  0
 * 10 11 12 13 14  0  0  0
 *  0  0  0  0  0  0  0  0 
 *
 */
bool scube_t::is_valid_fft_grid3(sfixn npx, sfixn npy, const sfixn *initP,
    sfixn nqx, sfixn nqy, const sfixn *initQ) 
{
    if (DEBUG) assert(num_of_vars() == 3);

    sfixn ex = get_bounds_exp(0);
    sfixn ey = get_bounds_exp(1);
    const sfixn *Wx = get_roots(0);
    const sfixn *Wy = get_roots(1);

    // check if initP is valid or not
    sfixn *A_d, *X_d;
    cudaMalloc((void **)&A_d, sizeof(sfixn) * npx * npy);
    cudaMalloc((void **)&X_d, sizeof(sfixn) << (ex + ey));
    cudaMemcpy(A_d, initP, sizeof(sfixn) * npx * npy, cudaMemcpyHostToDevice);
    
    // expanded data for the in-place bivariate fft
    expand_to_fft2_dev(ex, ey, X_d, npx, npy, A_d);

    bivariate_stockham_dev(X_d, ey, Wy, ex, Wx, fp);
    cudaFree(A_d);
    if (has_zero_in_vector(X_d, sfixn(1) << (ex + ey))) {
        cudaFree(X_d); 
        return false;
    }

    // check if initQ is valid or not
    cudaMalloc((void **)&A_d, sizeof(sfixn) * nqx * nqy);
    cudaMemcpy(A_d, initQ, sizeof(sfixn) * nqx * nqy, cudaMemcpyHostToDevice);
    // expanded data for the in-place bivariate fft
    expand_to_fft2_dev(ex, ey, X_d, nqx, nqy, A_d);
    bivariate_stockham_dev(X_d, ey, Wy, ex, Wx, fp);
    cudaFree(A_d);

    bool ret = has_zero_in_vector(X_d, sfixn(1) << (ex + ey));
    cudaFree(X_d);
    return (ret == false);
}

///////////////////////////////////////////////////////////////////////////////
//  Linear translations related 
///////////////////////////////////////////////////////////////////////////////

/**
 * Taylor shift by one, a special case of translation.
 *
 * Computes P(x) = P(x + 1) in-place.
 *
 * For example, let P(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3.
 * We compute P(x+1) as follows
 *
 * (S0) [a0 + a1 + a2 + a3, a1 + a2 + a3, a2 + a3, a3]
 * (S1) [a0 + a1 + a2 + a3, a1 + 2 * a2 + 3 * a3, a2 + 2 * a3, a3]
 * (S2) [a0 + a1 + a2 + a3, a1 + 2 * a2 + 3 * a3, a2 + 3 * a3, a3]
 *
 * That is the i-th step, the ith slot is filled by the final value,
 * for i = 0, 1, ..., m - 2. The last element a_{m-1} does not change 
 * throughout.
 */
void taylor_shift(sfixn m, sfixn *P, sfixn fp) {
    for(sfixn i = 0; i <= m - 2; ++i){
        for(sfixn j = m - 2; j >= i; --j) {
            P[j] = add_mod(P[j], P[j+1], fp);
        }
    }
}

/* P(x) = P(x - 1) */
void inverse_taylor_shift(sfixn m, sfixn *P, sfixn fp) {
    for(sfixn i = 0; i <= m - 2; ++i){
        for(sfixn j = m - 2; j >= i; --j) {
            P[j] = sub_mod(P[j], P[j+1], fp);
        }
    }
}

/////////////////////////////////////////////////////////////////////
// P(x) = P(x + a), generalized version 
// one modular multiplication is required 
// and assume that a > 1.
/////////////////////////////////////////////////////////////////////
static inline void taylor_shift(sfixn a, sfixn m, sfixn *P, sfixn fp) {
    for(sfixn i = 0; i <= m - 2; ++i){
        for(sfixn j = m - 2; j >= i; --j) {
            sfixn t = mul_mod(P[j+1], a, fp);
            P[j] = add_mod(P[j], t, fp);
        }
    }
}

/* P(x) = P(x - a) */
static inline void inverse_taylor_shift(sfixn a, sfixn m, sfixn *P, sfixn fp) {
    for(sfixn i = 0; i <= m - 2; ++i){
        for(sfixn j = m - 2; j >= i; --j) {
            sfixn t = mul_mod(P[j+1], a, fp);
            P[j] = sub_mod(P[j], t, fp);
        }
    }
}

/**
 * Linear translation of univariate or bivariate polynomials over finite fields.
 * 
 * @a, the offset 
 * @P, the input coefficient vector of length m
 * @Q, the output coefficient vector of length m
 * @m, the size of polynomials
 * @fp, fourier prime
 *
 * translation(m, Q, translator, P, fp) computes Q = P(x + a)
 *
 * That is, given P = [p_0, p_1, ..., p_{m-1}] and 
 *
 * P(x) = p_0 + p_1 * x + ... + p_{m-1} * x^m, 
 *
 * we compute
 * 
 * Q(x) = p_0 + p_1 * (x + a) + ... + p_{m-1} * (x + a)^m
 *      = q_0 + q_1 * x + ... + q_{m-1} * x^m 
 * 
 * Entries of Q are [q_0, q_1, ..., q_{m-1}]
 */
void translation1(sfixn a, sfixn m, sfixn *Q, const sfixn *P, sfixn fp) 
{
    // initialize Q as a copy of P
    // assume that P and Q do not overlap.
    memcpy(Q, P, sizeof(sfixn)*m);
    if (a == 1) {
        taylor_shift(m, Q, fp); 
    } else if (a > 0) {
        taylor_shift(a, m, Q, fp);
    }
}

/* In-place version */
void translation1(sfixn a, sfixn m, sfixn *P, sfixn fp) {
    if (a == 1) {
        taylor_shift(m, P, fp); 
    } else if (a > 0) {
        taylor_shift(a, m, P, fp);
    }
}


/* P(x) = P(x - a) */
void inverse_translation1(sfixn a, sfixn m, sfixn *P, sfixn fp) {
    if (a == 1) {
        inverse_taylor_shift(m, P, fp); 
    } else if (a > 0) {
        inverse_taylor_shift(a, m, P, fp);
    }
}

/** 
 * Given F(x, y) = f0(x) + f1(x) * y + f2(x) * y^2, we compute
 * F(x + a, y) = f0(x + a) + f1(x + a) * y + f2(x + a) * y^2
 */
void list_translation1(sfixn a, sfixn nx, sfixn ny, sfixn *F, sfixn fp)
{
    for (sfixn i = 0; i < ny; ++i) { 
        translation1(a, nx, F + i * nx, fp); 
    }
}

void list_inverse_translation1(sfixn a, sfixn nx, sfixn ny, sfixn *F, sfixn fp)
{
    for (sfixn i = 0; i < ny; ++i) { 
        inverse_translation1(a, nx, F + i * nx, fp); 
    }
}

/* Out-of-place version */
void list_translation1(sfixn a, sfixn nx, sfixn ny, 
    sfixn *F, const sfixn *G, sfixn fp)
{
    memcpy(F, G, sizeof(sfixn)*nx*ny);
    for (sfixn i = 0; i < ny; ++i) {
        translation1(a, nx, F + i * nx, fp);
    }
}

/**
 * Navie matrix transposition.
 *
 * A[i, j] = B[j, i] for all 0 <= i < ny, 0 <= j < nx. 
 */
static inline void
matrix_transpose(sfixn nx, sfixn ny, sfixn *A, const sfixn *B) {
    for (sfixn i = 0; i < ny; ++i) {
        for (sfixn j = 0; j < nx; ++j) {
            A[j * ny + i] = B[i * nx + j];
        }
    }
}

/**
 * Bivariate translations can be implemented as a composition of
 * two univariate translations as follows.
 *
 * F(x, y) --> F(x + a, y)      list translation along x
 *         --> G(y, x + a)      matrix transposition 
 *         --> G(y + b, x + a)  list translation along y
 *         --> F(x + a, y + b)  matrix transposition 
 *
 * If b = 0, then no need to translate y variable.
 */
void translation2(sfixn a, sfixn b, sfixn nx, sfixn ny, 
    sfixn *F, sfixn fp) 
{
    list_translation1(a, nx, ny, F, fp);
    if (b == 0) return;
    sfixn *G = new sfixn[nx * ny];
    matrix_transpose(nx, ny, G, F);
    list_translation1(b, ny, nx, G, fp);
    matrix_transpose(ny, nx, F, G);
    delete [] G;
}

/* F(x, y) = F(x - a, y - b) */
void inverse_translation2(sfixn a, sfixn b, sfixn nx, sfixn ny, 
    sfixn *F, sfixn fp)
{
    list_inverse_translation1(a, nx, ny, F, fp);
    if (b == 0) return;
    sfixn *G = new sfixn[nx * ny];
    matrix_transpose(nx, ny, G, F);
    list_inverse_translation1(b, ny, nx, G, fp);
    matrix_transpose(ny, nx, F, G);
    delete [] G;
}

/* Out-of-place version */
void translation2(sfixn a, sfixn b, sfixn nx, sfixn ny, sfixn *F, 
    const sfixn *H, sfixn fp)  
{
    memcpy(F, H, sizeof(sfixn)*nx*ny);
    list_translation1(a, nx, ny, F, fp);
    sfixn *G = new sfixn[nx * ny];
    matrix_transpose(nx, ny, G, F);
    list_translation1(b, ny, nx, G, fp);
    matrix_transpose(ny, nx, F, G);
    delete [] G;
}

/* List version, in-place */
void list_translation2(sfixn a, sfixn b, 
    sfixn nx, sfixn ny, sfixn nz, sfixn *F, sfixn fp) 
{
    for (sfixn i = 0; i < nz; ++i) {
        translation2(a, b, nx, ny, F + i * nx * ny, fp);
    }
}

/* List version, out-of-place */
void list_translation2(sfixn a, sfixn b, 
    sfixn nx, sfixn ny, sfixn nz, sfixn *F, const sfixn *H, sfixn fp) 
{
    memcpy(F, H, sizeof(sfixn)*nx*ny*nz);
    for (sfixn i = 0; i < nz; ++i) {
        translation2(a, b, nx, ny, F + i * nx * ny, fp);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Building the scube
///////////////////////////////////////////////////////////////////////////////

/**
 * Return the initial of a polynomial.
 *
 * @coeff_sz, the size of a coefficient 
 * @mdeg, the main degree
 * @P,  coefficient vector of size coeff_sz * (mdeg + 1)
 *
 * The return value is a constant pointer the the leading coefficient. 
 *
 * For example, coeff_sz = 4, mdeg = 2 and P = 
 *
 *   {a0, a1, a2, a3,  a4, a5, a6, a7,  a8, a9, a10, a11}
 *
 * then it returns P + coeff_sz * mdeg. 
 *
 * The assumption is that one of a8, a9, a10 and a11 is not zero.
 *
 */
const sfixn *init_poly(sfixn coeff_sz, sfixn mdeg, const sfixn *P) {
    const sfixn *init = P + coeff_sz * mdeg;
    if (DEBUG) {
        assert(mdeg >= 1);
        bool is_zero = true;
        for (int i = 0; i < coeff_sz; ++i) {
            if (init[i] != 0) { is_zero = false; break; }
        }
        assert(!is_zero);
    }
    return init;
}

/* Out-of-place version*/
void init_poly(sfixn *initP, sfixn coeff_sz, sfixn mdeg, const sfixn *P) 
{
    const sfixn *init = P + coeff_sz * mdeg;
    if (DEBUG) {
        assert(mdeg >= 1);
        bool is_zero = true;
        for (int i = 0; i < coeff_sz; ++i) {
            if (init[i] != 0) { is_zero = false; break; }
        }
        assert(!is_zero);
    }
    memcpy(initP, init, sizeof(sfixn) * coeff_sz);
}

///////////////////////////////////////////////////////////////////////////////
// Function fill_roots generates primitive roots of unity deterministically.
// Do not try to use different primitive roots of unity, 
// since the condition of whether a fft grid is valid or not is independent
// of roots chosen. 
///////////////////////////////////////////////////////////////////////////////
void scube_t::fill_roots() {
    // initialize the primitive root of unity
    sfixn *W = get_roots();
    const sfixn *es = get_bounds_exp();
    get_powers_of_roots(num_of_vars() - 1, W, es, fp);
    if (DEBUG) checkCudaError("error found in fill_roots");
}

///////////////////////////////////////////////////////////////////////////////
// Function fill_inverse_roots inverses all the precomputed primitive roots
// of unity. Be ready for the interpolations. 
///////////////////////////////////////////////////////////////////////////////
void scube_t::fill_inverse_roots() {
    sfixn *W = get_roots();
    const sfixn *es = get_bounds_exp();
    inv_powers_of_roots(num_of_vars() - 1, W, es, fp);
    if (DEBUG) checkCudaError("error found in fill_inverse_roots");
}

/**
 * The version accepts raw input data. Assume that P and Q are bivariate
 * polynomials in recursive dense representation, and assume that 
 * the main degree of P is as large as that of Q. 
 */
bool scube_t::build_scube_data2(sfixn npx,
    const sfixn *P, sfixn nqx, const sfixn *Q)
{
    cudaError_t err;
    assert(nvars == 2);
    sfixn npy = hdeg + 1, nqy = ldeg + 1;
    
    // fill primitive root of unity
    err = cudaMalloc((void**)&Wd, sizeof(sfixn) * root_size());
    if (err != cudaSuccess) { 
        if (DEBUG) fprintf(stderr, "##error : not enough GPU memory\n");
        Wd = NULL;
        return false; 
    }
    fill_roots();

    // to generate a valid FFT grid via translations
    const sfixn *initP = init_poly(npx, hdeg, P);
    const sfixn *initQ = init_poly(nqx, ldeg, Q);
    bool translated = false;

    // try random translations 5 times if initially the inputs are not valid.
    if (!is_valid_fft_grid2(npx, initP, nqx, initQ)) {
        sfixn *initF = new sfixn[npx];
        sfixn *initG = new sfixn[nqx];
        const int MAX_TRIALS = 5;
        int num_of_trials = 0;
        for ( ; num_of_trials < MAX_TRIALS; ++num_of_trials) {
            // first try if a = 1 works, since this is cheaper 
            // than other liner translations.
            translator[0] = (num_of_trials == 0) ? 1 : (rand() % fp);
            init_poly(initF, npx, hdeg, P);
            init_poly(initG, nqx, ldeg, Q);
            translation1(translator[0], npx, initF, fp);
            translation1(translator[0], nqx, initG, fp);
            if (INFOLEVEL >= 2) 
                printf("##info : translation with a = %d\n", translator[0]);
            if (is_valid_fft_grid2(npx, initF, nqx, initG)) { 
                translated = true; 
                break; 
            }
        }
        delete [] initF;
        delete [] initG;
        if (num_of_trials >= MAX_TRIALS) {
            if (INFOLEVEL >= 2) printf("##info : translation not found\n"); 
            return false;  // failed
        }
    } else {
        if (INFOLEVEL >= 2) 
            printf("##info : no translation needed for the input\n");
    }

    // ready to do the real construction
    // F and G hold the translated polynomials of P and Q
    sfixn *F, *G;
    if (translated) {
        F = new sfixn[npx * npy];
        G = new sfixn[nqx * nqy];
        list_translation1(translator[0], npx, npy, F, P, fp);
        list_translation1(translator[0], nqx, nqy, G, Q, fp);
    }

    sfixn bpow = get_bounds_exp(0); 
    sfixn B = (sfixn(1) << bpow);
    const sfixn *W = get_roots(0);

    // Evaluate P at W
    sfixn *P_d, *P2_d;
    cudaMalloc((void**)&P_d, sizeof(sfixn) * npy * B);
    if (translated) {
        cudaMemcpy(P_d, F, sizeof(sfixn) * npx * npy, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(P_d, P, sizeof(sfixn) * npx * npy, cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&P2_d, sizeof(sfixn) * npy * B);
    expand_to_list_fft_dev(npy, B, bpow, P2_d, npx, P_d);

    // now P2_d holds the evaluated data, layout (I) at subres.cu
    list_stockham_dev(P2_d, npy, bpow, W, fp);
    // now P_d holds the transposed data, layout (II) at subres.cu
    transpose_dev(P_d, P2_d, B, npy);
    cudaFree(P2_d);
             
    // Evaluate Q at W
    sfixn *Q_d, *Q2_d;
    cudaMalloc((void**)&Q_d, sizeof(sfixn) * nqy * B);

    if (translated) {
        cudaMemcpy(Q_d, G, sizeof(sfixn) * nqx * nqy, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(Q_d, Q, sizeof(sfixn) * nqx * nqy, cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&Q2_d, sizeof(sfixn) * nqy * B);
    expand_to_list_fft_dev(nqy, B, bpow, Q2_d, nqx, Q_d);

    // Q2_d holds the evaluated data, layout (I) at subres.cu
    list_stockham_dev(Q2_d, nqy, bpow, W, fp);
    // Q_d holds the transposed data, layout (II) at subres.cu
    transpose_dev(Q_d, Q2_d, B, nqy);
    cudaFree(Q2_d);

    if (translated) { delete [] F; delete [] G; }

    // Compute subres chain for each evaluation point
    err = cudaMalloc((void**)&Sd, sizeof(sfixn) * cube_size());
    if (err != cudaSuccess) { 
        if (DEBUG) fprintf(stderr, "##error : not enough GPU memory\n");
        Sd = NULL; 
        cudaFree(P_d); 
        cudaFree(Q_d);
        return false; 
    }

    sfixn sz = cube_size();
    reset_vector_dev(sz, Sd);

    // perform fine-grained Brown's algorithm
    cumodp_err ret = fine_subres_uni_dev(Sd, B, npy - 1, P_d, nqy - 1, Q_d, fp);
    if (ret == CUMODP_SUCCESS) { 
        if (INFOLEVEL >= 2) printf("##info : fine-grained scube constructed\n");
        set_layout(FINE);
    } else if (ret == CUMODP_OUT_OF_MEMORY) {
        cudaFree(P_d); 
        cudaFree(Q_d);
        if (DEBUG) fprintf(stderr, "out of meomory\n");
        return false;
    } else if (ret == CUMODP_SUBRES_NON_REGULAR) {
        // use coarse grained implementation
        bool val = coarse_subres_uni_dev(Sd, B, npy - 1, P_d, nqy - 1, Q_d, fp);
        if (!val) { 
            set_layout(ERROR); 
            cudaFree(P_d); 
            cudaFree(Q_d);
            return false; 
        } 
        set_layout(COARSE);
        if (INFOLEVEL >= 2) printf("##info : coarse-grained scube constructed\n");
    }

    cudaFree(P_d); 
    cudaFree(Q_d);

    // take the inverse of all precomputed 
    // roots of unity for interpolations
    if (INFOLEVEL >= 2) printf("##info : roots are ready for interpolations\n");
    fill_inverse_roots();
    if (DEBUG) checkCudaError("error found in build_scube_data2");
    return true;
}

/**
 * Trivariate case, parallel to build_scube_data2.
 */
bool scube_t::build_scube_data3(sfixn npx, sfixn npy, 
    const sfixn *P, sfixn nqx, sfixn nqy, const sfixn *Q)
{
    cudaError_t err;

    if (DEBUG) assert(nvars == 3);
    sfixn npz = hdeg + 1, nqz = ldeg + 1;

    // fill primitive root of unity
    err = cudaMalloc((void**)&Wd, sizeof(sfixn) * root_size());
    if (err != cudaSuccess) { 
        if (DEBUG) fprintf(stderr, "##error : not enough GPU memory\n");
        Wd = NULL; 
        return false; 
    }
    fill_roots();
    
    // to generate a valid FFT grid via translations, todo
    const sfixn *initP = init_poly(npx * npy, hdeg, P);
    const sfixn *initQ = init_poly(nqx * nqy, ldeg, Q);
    bool translated = false;
    
    if (!is_valid_fft_grid3(npx, npy, initP, nqx, nqy, initQ)) {
        
        sfixn *initF = new sfixn[npx * npy];
        sfixn *initG = new sfixn[nqx * nqy];
        sfixn a, b; // translators
        int MAX_TRIALS = 5;
        int num_of_trials = 0;
        for ( ; num_of_trials < MAX_TRIALS; ++num_of_trials) {
            if (num_of_trials == 0) {
                a = 1, b = 0;
            } else if (num_of_trials == 1) {
                a = 0, b = 1;
            } else if (num_of_trials == 2) {
                a = 1, b = 1;
            } else {
                a = rand() % fp, b = rand() % fp;
            }
            translator[0] = a, translator[1] = b;
            init_poly(initF, npx * npy, hdeg, P);
            init_poly(initG, nqx * nqy, ldeg, Q);
            translation2(a, b, npx, npy, initF, fp);
            translation2(a, b, nqx, nqy, initG, fp);
            if (INFOLEVEL >= 2) 
                printf("##info : translation with a = %d, b = %d\n", 
                    translator[0], translator[1]);
            if (is_valid_fft_grid3(npx, npy, initF, nqx, nqy, initG)) { 
                translated = true; 
                break; 
            }
        }
        delete [] initF;
        delete [] initG;
        if (num_of_trials >= MAX_TRIALS) {
            if (INFOLEVEL >= 2) printf("##info : translation not found\n");
            return false;  // failed
        }
    }

    sfixn *F, *G;
    if (translated) {
        F = new sfixn[npx * npy * npz];
        G = new sfixn[nqx * nqy * nqz];
        list_translation2(translator[0], translator[1], npx, npy, npz, F, P, fp);
        list_translation2(translator[0], translator[1], nqx, nqy, nqz, G, Q, fp);
    }

    sfixn ex = get_bounds_exp(0);
    sfixn ey = get_bounds_exp(1); 
    sfixn  B = (sfixn(1) << (ex + ey));
    const sfixn *Wx = get_roots(0);
    const sfixn *Wy = get_roots(1);

    // Evaluate P at [Wx, Wy]
    // the size of P_d is larger than needed for fft
    // this is needed for transposition following
    sfixn *P_d, *P2_d;
    cudaMalloc((void**)&P_d, sizeof(sfixn) * npz * B);
    if (translated) {
        cudaMemcpy(P_d, F, sizeof(sfixn) * npx * npy * npz, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(P_d, P, sizeof(sfixn) * npx * npy * npz, cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&P2_d, sizeof(sfixn) * npz * B);
    expand_to_list_fft2_dev(npz, ex, ey, P2_d, npx, npy, P_d);

    ///////////////////////////////////////////////////////////////////////
    //  Suspect that the calling sequence should be
    //  list_bivariate_stockham_dev(P2_d, npz, ey, Wy, ex, Wx, fp);
    ///////////////////////////////////////////////////////////////////////

    // P2_d holds the evaluated data
    // P_d holds the transposed data
    // list_bivariate_stockham_dev(P2_d, npz, ex, Wx, ey, Wy, fp);
    list_bivariate_stockham_dev(P2_d, npz, ey, Wy, ex, Wx, fp);
    transpose_dev(P_d, P2_d, B, npz);
    cudaFree(P2_d);

    // Evaluate Q at [Wx, Wy]
    // the size of Q_d is larger than needed for fft
    // this is needed for transposition following
    sfixn *Q_d, *Q2_d;
    cudaMalloc((void**)&Q_d, sizeof(sfixn) * nqz * B);
    if (translated) {
        cudaMemcpy(Q_d, G, sizeof(sfixn)*nqx*nqy*nqz, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(Q_d, Q, sizeof(sfixn)*nqx*nqy*nqz, cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&Q2_d, sizeof(sfixn) * nqz * B);
    expand_to_list_fft2_dev(nqz, ex, ey, Q2_d, nqx, nqy, Q_d);

    // Q2_d holds the evaluated data
    // Q_d holds the transposed data
    // list_bivariate_stockham_dev(Q2_d, nqz, ex, Wx, ey, Wy, fp);
    list_bivariate_stockham_dev(Q2_d, nqz, ey, Wy, ex, Wx, fp);
    transpose_dev(Q_d, Q2_d, B, nqz);
    cudaFree(Q2_d);

    if (translated) { delete [] F; delete [] G; }

    // Compute subres chain for each evaluation point
    err = cudaMalloc((void**)&Sd, sizeof(sfixn) * cube_size());
    if (err != cudaSuccess) { 
        if (DEBUG) fprintf(stderr, "##error : not enough GPU memory\n");
        Sd = NULL; 
        cudaFree(P_d); 
        cudaFree(Q_d);
        return false; 
    }
    sfixn sz = cube_size();
    reset_vector_dev(sz, Sd);

    // perform fine-grained Brown's algorithm
    cumodp_err ret = fine_subres_uni_dev(Sd, B, npz - 1, P_d, nqz - 1, Q_d, fp);
    if (ret == CUMODP_SUCCESS) { 
        if (INFOLEVEL >= 2) printf("##info : fine-grained scube constructed\n");
        set_layout(FINE);
    } else if (ret == CUMODP_OUT_OF_MEMORY) {
        cudaFree(P_d); 
        cudaFree(Q_d);
        if (DEBUG) fprintf(stderr, "out of memory\n");
        return false;
    } else if (ret == CUMODP_SUBRES_NON_REGULAR) {
        // use coarse grained implementation
        bool val = coarse_subres_uni_dev(Sd, B, npz - 1, P_d, nqz - 1, Q_d, fp);
        if (!val) { 
            if (DEBUG) fprintf(stderr, "## coarse-grained failed\n");
            cudaFree(P_d); 
            cudaFree(Q_d);
            set_layout(ERROR); 
            return false; 
        }
        set_layout(COARSE);
        if (INFOLEVEL >= 2) printf("##info : coarse-grained scube constructed\n");
    }

    cudaFree(P_d); 
    cudaFree(Q_d);

    // take the inverse of all precomputed 
    // roots of unity for interpolations
    if (INFOLEVEL >= 2) printf("##info : roots are ready for interpolations\n");
    fill_inverse_roots();
    if (DEBUG) checkCudaError("##error : error found in build_scube_data2");
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//  Interpolation of coefficients 
////////////////////////////////////////////////////////////////////////////////

/**
 * Extract elements from S to array X. 
 * 
 * @stride, the stride size
 * @offset, offset of each element 
 * @B, the number of elements to be extracted, power of 2
 * @X, the output device array of length B
 * @S, the input devlce array
 *
 * Total number of threads is B, the ith-thread reads S[i * stride + offset] 
 * and stores to X[i]. 
 *
 */
__global__ void 
extract_item_ker(sfixn stride, sfixn offset, sfixn *X, const sfixn *S) {
    sfixn bid = blockIdx.x + (blockIdx.y << 15);
    sfixn tid = bid * blockDim.x + threadIdx.x;
    X[tid] = S[tid * stride + offset];
}

/**
 * Extract coefficients for a subresultant
 *
 * @n, the stride size
 * @m, offset of each element 
 *
 * Coarse-grained data version
 *
 * Example
 *
 * S = [ 1  2  3 
 *       4  5       
 *       6          
 *
 *       7  8  9
 *       10 11      
 *       12         
 *
 *       13 14 15
 *       16 17      
 *       18      
 *
 *       19 20 21
 *       22 23 
 *       24 ]
 *
 * We extract the resultant images by set n = 6, m = 5. 
 * This fills X by [6, 12, 18, 24].
 *
 * We extract S(1, 1) by set n = 6, m = 4. 
 * This fills X by [5, 11, 17, 23].
 *
 * Hence stride n = ldeg * (ldeg + 1) / 2 and 
 *
 * offset m = ldeg + (ldeg - 1) + ... + (i + 2) + j
 *          = (ldeg + i + 2) * (ldeg - i - 1) / 2 + j 
 *
 * ============================================================
 *
 * Fine-grained data version
 *
 * Example
 *
 * S = [ 1  2  3,   7  8  9,   13 14 15,  19 20 21 ] 
 *       4  5,      10 11,     16 17,     22 23,
 *       6,         12,        18,        24 
 *
 * We extract the resultant images by set m = 5, n = 1. 
 * This fills X by [6, 12, 18, 24].
 *
 * We extract the S(1, 1) by set m = 4, n = 2.
 * This fills X by [5, 11, 17, 23].
 *
 * We extract the S(1, 0) by set m = 3, n = 2.
 * This fills X by [4, 10, 16, 22].
 *
 * Hence stride size n = i + 1 and offset 
 *
 * m = ldeg * B + (ldeg - 1) * B + ... + (i + 2) * B + j
 *   = (ldeg + i + 2) * (ldeg - i - 1) * B / 2 + j 
 *
 */
void extract_item_dev(const scube_t* scb, sfixn i, sfixn j, sfixn *X)
{
#if DEBUG > 0
    const sfixn nthds = 4;
#else
    const sfixn nthds = 128;
#endif
    sfixn B = scb->coefficient_size();
    sfixn ldeg = scb->get_ldeg();

    sfixn nb = B / nthds;
    if (DEBUG) assert((nb >= 1) && (B % nthds == 0));

    dim3 nBlks(nb, 1, 1);
    if (rem2e(nb, 15) == 0) { 
        nBlks.x = (sfixn(1) << 15); nBlks.y = (nb >> 15); 
    }
    
    sfixn stride, offset;
    if (scb->get_layout() == scube_t::FINE) {
        stride = i + 1;
        offset = (ldeg + i + 2) * (ldeg - i - 1) * B / 2 + j;
    } else {
        stride = ldeg * (ldeg + 1) / 2;
        offset = (ldeg + i + 2) * (ldeg - i - 1) / 2 + j;
    }

    extract_item_ker<<<nb, nthds>>>(stride, offset, X, scb->get_cube());
    cudaThreadSynchronize();
    if (DEBUG) checkCudaError("##error : error found in extract_item_dev2");
}

/**
 * See subres_p.cu for data layout 
 * 
 * Example let nx = 2, ny = 2.
 *
 * Coare-grained layout
 *
 * S = [ 1  2  3 
 *       4  5     (1, 1)  
 *       6          
 *
 *       7  8  9
 *       10 11    (wx, 1)  
 *       12         
 *
 *       13 14 15
 *       16 17    (1, wy)   
 *       18      
 *
 *       19 20 21
 *       22 23    (wx, wy) 
 *       24 ]
 *
 *  [ 6  12]  
 *  [18  24] is ready for interpolation as the resultant. 
 *  Extracting data from scube is the same as in the univariate case.
 *
 */

/** 
 * Interpolate a cofficient of from the scube.
 *
 * @i, the subresultant index
 * @j, the coefficient index
 *
 *   i  j     i  j     i  j     i  j 
 * S(3, 0)  S(3, 1)  S(3, 2)  S(3, 3) 
 * S(2, 0)  S(2, 1)  S(2, 2)
 * S(1, 0)  S(1, 1)
 * S(0, 0)
 *
 */
scube_t::subres_coeff_type 
scube_t::subres_coeff(int i, int j) {
    assert(i >= j && j >= 0 && ldeg > i);
    assert(lyt == FINE || lyt == COARSE);

    subres_coeff_index idx = make_coeff_index(i, j);
    subres_coeff_iterator it = coeffs.find(idx);
    if (it != coeffs.end()) {
        if (INFOLEVEL >= 2) 
            printf("##info : old coefficient S(%d, %d) used\n", i, j);
        return it->second;
    }
    if (INFOLEVEL >= 2) 
        printf("##info : new coefficient S(%d, %d) interpolated\n", i, j);
    return interp_subres_coeff(i, j)->second;
}

scube_t::subres_coeff_iterator 
scube_t::interp_subres_coeff(int i, int j) {
    
    // coefficient array is freed by the deconstructor
    sfixn B = coefficient_size();
    sfixn *cf = new sfixn[B];
    subres_coeff_index idx = make_coeff_index(i, j);
 
    sfixn *Xd;
    cudaError_t err = cudaMalloc((void **)&Xd, sizeof(sfixn) * B);
    assert(err == cudaSuccess);

    if (num_of_vars() == 2) {
        const sfixn *invW = get_roots();
        sfixn ex = get_bounds_exp(0);

        extract_item_dev(this, i, j, Xd);
        inverse_stockham_dev(Xd, B, ex, invW, fp);
        cudaMemcpy(cf, Xd, sizeof(sfixn) * B, cudaMemcpyDeviceToHost);
        cudaFree(Xd);

        // check if inverse translations needed
        if (is_translated()) { 
            inverse_translation1(translator[0], B, cf, fp); 
            if (INFOLEVEL >= 2) {
                printf("##info : inverse translation applied : a = %d\n",
                    translator[0]);
            }
        }

    } else {
        assert(num_of_vars() == 3);
        sfixn ex = get_bounds_exp(0);
        sfixn ey = get_bounds_exp(1);
        sfixn nx = (sfixn(1) << ex);
        sfixn ny = (sfixn(1) << ey);
        const sfixn *invWx = get_roots(0);
        const sfixn *invWy = get_roots(1);

        extract_item_dev(this, i, j, Xd);
        inverse_bivariate_stockham_dev(Xd, ey, invWy, ex, invWx, fp);
        cudaMemcpy(cf, Xd, sizeof(sfixn) * B, cudaMemcpyDeviceToHost);
        cudaFree(Xd);

        // check if inverse translations needed
        if (is_translated()) { 
            inverse_translation2(translator[0], translator[1], nx, ny, cf, fp);
            if (INFOLEVEL >= 2) {
                printf("##info : inverse translation applied : a = %d, b = %d\n",
                    translator[0], translator[1]);
            }
        }
    }

    std::pair<subres_coeff_iterator, bool> ret 
        = coeffs.insert(std::make_pair(idx, cf));

    // S(i, j) should not exist!
    if (DEBUG) assert(ret.second);
    return ret.first;
}

/***************************** END_OF_FILE ************************************/
