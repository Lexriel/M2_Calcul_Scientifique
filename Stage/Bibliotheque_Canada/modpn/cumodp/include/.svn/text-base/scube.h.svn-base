#ifndef _SCUBE_H_
#define _SCUBE_H_

#include <map>
#include <cassert>
#include "defines.h"
#include "rdr_poly.h"
#include "fft_aux.h"
#include "inlines.h"
#include "cudautils.h"
#include "stockham.h"
#include "list_stockham.h"
#include "subres.h"

/**
 * Data structure for storing the subresultant chain of two multivariate 
 * polynomials using FFT evaluation and interpolation. The first n - 1 variables
 * of the input polynomials are evaluated at sufficient many points. 
 * 
 * Data layout:
 *
 * (1) List of subresultant chains, coarsed-grained representation
 *
 *     S30 S31 S32 S33
 *     S20 S21 S22
 *     S10 S11
 *     S00
 *
 *     T30 T31 T32 T33
 *     T20 T21 T22
 *     T10 T11
 *     T00
 *
 * (2) subresultant lists, fine-grained representation 
 *
 *     S30 S31 S32 S33   T30 T31 T32 T33
 *     S20 S21 S22       T20 T21 T22
 *     S10 S11           T10 T11
 *     S00               T00
 *
 * NOTE:
 *  
 * Each Sij is a subresultant coefficient for one evaluation point. 
 * 
 * Let n = 3, es[0] = 16, es[1] = 32. Then the size of the FFT grid is 16 * 32.
 * The size of Wd is  16 / 2 + 32 / 2 = 24, since in each variable, 
 * only half of roots are needed.
 *
 **/

class scube_t {
public:
    typedef enum {COARSE, FINE, UNKNOWN, ERROR} layout;

    typedef std::pair<int, int> subres_coeff_index;
    typedef const sfixn *subres_coeff_type;
    typedef std::map<subres_coeff_index, subres_coeff_type> subres_coeff_data;
    typedef subres_coeff_data::iterator subres_coeff_iterator;

    // the characteristic of the finite field 
    // we assume that the fft degree of fp is large enough
    const sfixn fp; 

    // bivariate input
    scube_t(int npx, int npy, int nqx, int nqy, sfixn p) 
        : fp(p), nvars(2), Wd(NULL), Sd(NULL), lyt(UNKNOWN)
    {
        init2(npx, npy, nqx, nqy);
    }

    // trivariate input
    scube_t(int npx, int npy, int npz, int nqx, int nqy, int nqz, sfixn p)
        : fp(p), nvars(3), Wd(NULL), Sd(NULL), lyt(UNKNOWN)
    {
        init3(npx, npy, npz, nqx, nqy, nqz);
    }

    // general constructor N = 2 or N = 3
    scube_t(sfixn N, const sfixn *sz_p, const sfixn *sz_q, sfixn p)
        : fp(p), nvars(N), Wd(NULL), Sd(NULL), lyt(UNKNOWN) 
    {
        if (N == 2) {
            init2(sz_p[0], sz_p[1], sz_q[0], sz_q[1]);
        } else {
            init3(sz_p[0], sz_p[1], sz_p[2], sz_q[0], sz_q[1], sz_q[2]);
        }
    }

    // general constructor
    scube_t(const rdr_poly &P, const rdr_poly &Q, sfixn p);

    // deconstructor
    ~scube_t() {
        delete [] bound_es;
        delete [] translator;
        if (Wd) { cudaFree(Wd); }
        if (Sd) { cudaFree(Sd); }
        free_subres_coeffs();
    }

    // set the data layout of the scube
    void set_layout(layout lt) { lyt = lt; }
    layout get_layout() const { return lyt; }
    bool is_valid_scube() { return (lyt != scube_t::ERROR); }

    // check if initials are valid for the grid or not
    // bivariate case
    bool is_valid_fft_grid2(sfixn npx, const sfixn *initP,
        sfixn nqx, const sfixn *initQ);

    // check if initials are valid for the grid or not
    // trivariate case
    bool is_valid_fft_grid3(sfixn npx, sfixn npy, const sfixn *initP,
        sfixn nqx, sfixn nqy, const sfixn *initQ);

    // build scube data and return false if failed, 
    // bivariate case
    bool build_scube_data2(sfixn npx, const sfixn *P, 
        sfixn nqx, const sfixn *Q);

    // build scube data and return false if failed, 
    // trivariate case
    bool build_scube_data3(sfixn npx, sfixn npy, const sfixn *P, 
        sfixn nqx, sfixn nqy, const sfixn *Q);

    // interpolation of coefficients from the images
    subres_coeff_type subres_coeff(int i, int j);
    
    // the number of words required 
    sfixn cube_size() const { return coeff_size * (ldeg * ldeg + ldeg) / 2; }
    sfixn num_of_vars() const { return nvars; }

    // the size of a coefficient S(i,j)
    sfixn coefficient_size() const { return coeff_size; }
    sfixn get_hdeg() const { return hdeg; }
    sfixn get_ldeg() const { return ldeg; }

    sfixn get_bounds_exp(int i) const { return bound_es[i]; } 
    const sfixn *get_bounds_exp() const { return bound_es; }  

    sfixn *get_cube() { return Sd; }
    const sfixn *get_cube() const { return Sd; }
    
    // precompute powers of primitive roots of unity
    void fill_roots();
    void fill_inverse_roots();

    // get the roots for the i-th variables
    sfixn *get_roots(int i) { return Wd + offset_helper(i); }
    sfixn *get_roots() { return Wd; }
    const sfixn *get_roots(int i) const { return Wd + offset_helper(i); }
    const sfixn *get_roots() const { return Wd; }

    // print general information
    void info() const;
    void print_image(sfixn i = 0) const;

private:
    sfixn nvars;              // number of variables (nvars >= 2)  
    sfixn hdeg;               // hdeg = deg(P, y)
    sfixn ldeg;               // ldeg = deg(Q, y), hdeg >= ldeg
    sfixn *bound_es;          // the exponents of the FFT size
                              // in each variable for i = 0 .. nvars - 2
    sfixn *Wd;                // the precomputed DFT grid on the device
    sfixn *Sd;                // actual data on the device
    layout lyt;               // the data layout 
    subres_coeff_data coeffs; // cached coefficients in a map
    sfixn coeff_size;         // size of a coefficient 
    sfixn *translator;        // for linear translations, range 0 .. nvar-2

    sfixn root_size() const {
        sfixn sz = 0;
        for (int i = 0; i < nvars - 1; ++i) 
            sz += (sfixn(1) << (bound_es[i] - 1));
        return sz;
    }

    void init2(int npx, int npy, int nqx, int nqy);
    void init3(int npx, int npy, int npz, int nqx, int nqy, int nqz);

    // return the leading address of jth subresultant of the ith image
    // for debugging purpose
    const sfixn *ith_image_jth_subres(int i, int j) const;

    sfixn offset_helper(int i) const {
        sfixn offset = 0;
        for (; i > 0; --i) {
            offset += (sfixn(1) << (bound_es[i-1] - 1));
        }
        return offset;
    }
    
    bool is_translated() const {
        if (num_of_vars() == 2) {
            return (translator[0] != 0);
        } else {
            return (translator[0] != 0) || (translator[1] != 0);
        }
    }

    subres_coeff_iterator interp_subres_coeff(int i, int j);

    subres_coeff_index make_coeff_index(int i, int j) const {
        if (DEBUG) assert(i >= 0 && j >= 0 && i < ldeg && j <= i);  
        return std::make_pair(i, j);
    }

    void free_subres_coeffs();

    bool check_fourier_degree() {
        int fdeg = fourier_degree(fp);
        for (int i = 0; i < nvars - 1; ++i) {
            if (fdeg < get_bounds_exp(i)) {
                if (1) {
                    printf("## info : insufficient fourier degree : %d < %d\n",
                        fdeg, get_bounds_exp(i));
                }
                set_layout(scube_t::ERROR);
                return false;
            }
        }
        return true;
    }

    // NO COPY OR ASSIGNMENT
    scube_t(const scube_t &);
    scube_t& operator=(const scube_t&);
};

#endif // END_OF_FILE
