#include <cstring>
#include <iostream>
#include <cassert>

#include "../include/defines.h"
#include "../include/rdr_poly.h"


/** 
 * Constructor
 *
 * @_n  : the number of variables
 * @_ns : the partial size vector, _ns[i] >= deg(f, x_i) + 1
 * 
 * Return : a zero rdr_poly object of the specified size
 *
 * */ 
rdr_poly::rdr_poly(sfixn _n, sfixn *_ns) : n(_n), ns(NULL), coeffs(NULL)
{
    #if DEBUG >= 1 
    assert(n > 0);
    for (sfixn i = 0; i < n; ++i) { assert(_ns[i] > 0); }
    #endif

    ns = new sfixn[n];
    memmove(ns, _ns, sizeof(sfixn)*n);

    sz = 1;
    for (sfixn i = 0; i < n; ++i) sz *= ns[i];
    coeffs = new sfixn[sz]();
}

/**
 * Constructor
 *
 * @_n      : the number of variables
 * @_ns     : the partial size vector, _ns[i] >= deg(f, x_i) + 1
 * @_coeffs : the coefficient vector 
 *
 * Return : a rdr_poly object 
 **/
rdr_poly::rdr_poly(sfixn _n, sfixn *_ns, sfixn *_coeffs, bool host) : 
    n(_n), ns(NULL), coeffs(NULL) 
{
    #if DEBUG >= 1 
    assert(n > 0);
    for (sfixn i = 0; i < n; ++i) { assert(_ns[i] > 0); }
    #endif

    sz = 1;
    for (sfixn i = 0; i < n; ++i) { sz *= _ns[i]; }
    ns = new sfixn[n]();
    memmove(ns, _ns, sizeof(sfixn)*n);
    coeffs = new sfixn[sz];
    if (host) {
        memmove(coeffs, _coeffs, sizeof(sfixn)*sz);
    } else {
        cudaMemcpy(coeffs, _coeffs, sizeof(sfixn)*sz, cudaMemcpyDeviceToHost);
    }
}

/** 
 * Constructor, specialized for bivariate polynomials
 *
 * Return : a bivariate zero rdr_poly object 
 *
 * */ 
rdr_poly::rdr_poly(sfixn nx, sfixn ny) 
    : n(2), sz(nx*ny), ns(NULL), coeffs(NULL) {
    
    #if DEBUG >= 1 
    assert(nx > 0 && ny > 0);
    #endif
    
    ns = new sfixn[2]();
    ns[0] = nx; ns[1] = ny;       

    coeffs = new sfixn[sz]();
}

/*
 * Constructor, specialized for bivariate polynomials
 *
 * @nx  : the partial size in x, the first variable
 * @ny  : the partial size in y, the second variable
 * 
 * Return : a rdr_poly object (bivariate)
 *
 * Example: let
 *
 * f = (1 + 2x) + (3 + 4x + 5x^2)y + (6 + 7x^2) y^2
 *
 * If nx = 3 and ny = 3, then its (3, 3)-rdr is 
 *
 * coeffs = {{1, 2, 0}, {3, 4, 5}, {6, 0, 7}}   
 *
 * If nx = 4 and ny = 4, then its (4, 4)-rdr is
 *
 * coeffs = {{1, 2, 0, 0}, {3, 4, 5, 0}, {6, 0, 7, 0}, {0, 0, 0, 0}}   
 *
 **/ 
rdr_poly::rdr_poly(sfixn nx, sfixn ny, const sfixn *_coeffs, bool host) 
        : n(2), sz(nx*ny), ns(NULL), coeffs(NULL) {
    
    #if DEBUG >= 1 
    assert(nx > 0 && ny > 0);
    #endif
    
    ns = new sfixn[2]();
    ns[0] = nx; ns[1] = ny;       

    coeffs = new sfixn[sz]();
    if (host) {
        memmove(coeffs, _coeffs, sizeof(sfixn)*sz);
    } else {
        cudaMemcpy(coeffs, _coeffs, sizeof(sfixn)*sz, cudaMemcpyDeviceToHost);
    }
}

/**
 * Copy constructor, deep copy.
 *
 */
rdr_poly::rdr_poly(const rdr_poly &P) 
    : n(P.n), ns(NULL), sz(P.sz), coeffs(NULL)
{
    ns = new sfixn[n]; 
    memmove(ns, P.ns, sizeof(sfixn)*n);
    coeffs = new sfixn[sz]; 
    memmove(coeffs, P.coeffs, sizeof(sfixn)*sz);
}


// destructor
rdr_poly::~rdr_poly() 
{
    if (ns != NULL) { delete [] ns; ns = NULL; }
    if (coeffs != NULL) { delete [] coeffs; coeffs = NULL; }
}

bool rdr_poly::is_zero() const {
    for (sfixn i = 0; i < sz; ++i) {
        if (coeffs[i] != 0) return false;   
    }
    return true;
}

// display the internal data of a rdr_poly object
void rdr_poly::info() const 
{
    printf("number of variables: %d\n", n);
    printf("partial sizes: ");
    for (sfixn i = 0; i < n; ++i) printf("%3d ", ns[i]);
    printf("\ncoefficient vector of size %d:", sz);
    for (sfixn i = 0; i < sz; ++i) printf("%3d ", coeffs[i]);
    printf("\n");
}

/**
 * Compute the actual partial degree of a multivariate rdr polynomial. 
 *
 * @deg: the given degree
 * @F: the coefficient vector
 * @coefSz: the size of a coefficient
 * 
 * View F as an univariate polynomial has deg + 1 coefficients. 
 * Each coefficient has size of coefSz. By ignoring the leading 
 * zeroes, we compute the actual degree.
 *
 * This function is the same as the modpn function shrinkDeg. 
 * It returns 0 whenever F is a const polynomial including 0.
 *
 **/
sfixn 
rdr_poly::shrink_deg(sfixn deg, const sfixn *F, sfixn coefSz) const {
    const sfixn *tPtr = F + deg * coefSz;
    while (deg > 0) {
        for (sfixn i = 0; i < coefSz; ++i) { 
            if (tPtr[i] != 0) return deg; 
        }
        tPtr -= coefSz; --deg;
    }
    return deg;
}

/**
 * Compute the actual total degree of rdr polynomials.
 *
 * @accum : the coefficient size vector
 *
 **/
sfixn rdr_poly::tdeg_inner(const sfixn *accum) const {
    sfixn d = 0, i;
    
    // special case if the last element is nonzero.
    if (coeffs[sz - 1] != 0) {
        for (i = 0; i < n; ++i) d += ns[i];
        return d - n;
    }
    
    sfixn di, r, j;
    for (i = sz - 2; i > 0; --i) {
        if (coeffs[i] != 0) {
            // compute the total degree d_i of the i-th monomial
            // deg(M_i, x_{n - 1}) = i / accum[n - 1], i = i % accum[n - 1]
            // deg(M_i, x_{n - 2}) = i / accum[n - 2], i = i % accum[n - 2]
            r = i;
            di = 0;
            for (j = n - 1; j >= 0; --j) {
                di += (r / accum[j]);
                r %= accum[j];
            }
            d = (d > di) ? d : di;
        }
    }

    return d;    
}

sfixn rdr_poly::pdeg_inner(sfixn k, const sfixn *accum) const {

    if (k == n - 1) { return shrink_deg(ns[k] - 1, coeffs, accum[k]); }

    // the case 0 <= k <= n - 2, and m is the number of polys to be checked 
    sfixn m = sz / accum[k + 1]; 
    sfixn d = -1, di;
    for (sfixn i = 0; i < m; ++i) {
        di = shrink_deg(ns[k] - 1, coeffs + i * accum[k + 1], accum[k]);
        if (di == ns[k] - 1) { 
            return di; 
        } else {
            d = (d > di) ? d : di; 
        }
    }
    return d;
}

/**
 * Compute the actual degree of a rdr polynomial.
 *
 * @k     : the index, 0 <= k <= n. 
 *
 * If k is n, then returns the total degree, otherwise returns 
 * the partial degree in the variable with index k.
 *
 **/
sfixn rdr_poly::tdeg() const {

    // Compute the coefficient size in each variable on the fly.
    // accum[i] is the size of the coefficient in variable var[i].
    sfixn *accum = new sfixn[n];
    accum[0] = 1;
    for (sfixn i = 1; i < n; ++i) { accum[i] = accum[i-1] * ns[i-1]; }

    sfixn d = tdeg_inner(accum);
    delete [] accum;
    return d;
}

sfixn rdr_poly::pdeg(sfixn k) const {
    if (DEBUG) assert((k >= 0) && (k < n));

    // Compute the coefficient size in each variable on the fly.
    // accum[i] is the size of the coefficient in variable var[i].
    sfixn *accum = new sfixn[n];
    accum[0] = 1;
    for (sfixn i = 1; i < n; ++i) { accum[i] = accum[i-1] * ns[i-1]; }

    sfixn d = pdeg_inner(k, accum);
    delete [] accum;
    return d;
}

/**
 * @k, the index of the greatest variable
 * @data, the coefficient vector
 * @accum, the partial coefficient sizes
 * @vars, chars used
 *
 * Example, let f = (1 + 2*x) + (3 + 4*x)*y + (5 + 6*x)*y^2.
 *
 * Then k = 2, data = {1, 2, 3, 4, 5, 6}, accum = {1, 2}, vars = {x, y}.
 *
 */
void rdr_poly::print_inner(sfixn k, const sfixn *data, const sfixn *accum,
                           const char *vars) const
{
    // base case, only one variable 
    if ( k == 0) {
        printf("(%d", data[0]);
        for (sfixn i = 1; i < ns[k]; ++i)
            printf("+%d*%c^%d", data[i], vars[k], i);
        printf(")");
    } else {
        printf("(");
        print_inner(k-1, data, accum, vars);
        for (sfixn i = 1; i < ns[k]; ++i) {
            printf("+");
            print_inner(k-1, data + i*accum[k], accum, vars);
            printf("*%c^%d", vars[k], i);
        }
        printf(")");
    }
}

void rdr_poly::print_to_maple(const char *msg) const {
    char vars[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                   'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                   'u', 'v', 'w', 'x', 'y', 'z'};

    assert(n <= 26);

    printf("%s", msg);
    sfixn *accum = new sfixn[n];
    accum[0] = 1;
    for (sfixn i = 1; i < n; ++i) { accum[i] = accum[i - 1] * ns[i - 1]; }
    print_inner(n-1, coeffs, accum, vars);
    printf(":\n");
    delete [] accum;
}
