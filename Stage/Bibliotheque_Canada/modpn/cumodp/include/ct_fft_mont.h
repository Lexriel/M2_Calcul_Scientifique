#ifndef _CT_FFT_MONT_H_
#define _CT_FFT_MONT_H_

#include "defines.h"

/******************************* Exported Types ******************************/
struct uni_mont_fft_plan {
    // the length of the data array
    int n; 
    // the number of base FFT to conduct inside a block
    int num_of_threads;
    // the exponent of the base FFT 
    int base_fft_exponent;

    // constructors
    uni_mont_fft_plan(int nn, int nt, int be) :
        n(nn), num_of_threads(nt), base_fft_exponent(be) {}
    
    uni_mont_fft_plan(int nn) : n(nn) {
        num_of_threads = 16;
        base_fft_exponent = 4;   
    }

    // n is a power of 2
    // n >= 256, do not do small FFTs
    // n >= m * nThread, at least one block of base FFTs
    // nThread <= 2^9, the number of threads in a block constriant
    // m >= 2^4, for data alignment
    // m * nThread <= 2^12, the shared memory size constraint
    // nThread * 18 <= 2^13, the number of registers constriant 
    bool is_valid_plan() const {
        if ((n & (n - 1)) != 0) return false;   
        if (n < 256) return false; 
        if (num_of_threads > 512) return false;     
        if (base_fft_exponent < 4) return false;
        if (n < (num_of_threads << base_fft_exponent)) return false; 
        if ((num_of_threads << base_fft_exponent) > 4096) return false;
        return true;
    }

    int n_thd() const { return num_of_threads; }
    int b_exp() const { return base_fft_exponent; }
};

// [m, nThread] = [2^4,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8}]
// [m, nThread] = [2^5,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7}]
// [m, nThread] = [2^6,  {1, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6}]
// [m, nThread] = [2^7,  {1, 2^1, 2^2, 2^3, 2^4, 2^5}]
// [m, nThread] = [2^8,  {1, 2^1, 2^2, 2^3, 2^4}]
// [m, nThread] = [2^9,  {1, 2^1, 2^2, 2^3}]
// [m, nThread] = [2^10, {1, 2^1, 2^2}]
// [m, nThread] = [2^11, {1, 2^1}]
// [m, nThread] = [2^12, {1}]

/******************************* Exported Routines ***************************/
/*
 * tensor_mont_fft : compute DFT using the tensor product formulation
 *
 * @n : DFT size, length of A 
 * @k : exponent of n = 2^k
 * @A : coefficient vector on device
 * @w : n-th primitive root of unity modulo p
 * @p : prime number
 * 
 * The size of n is at least 256.
 *
 **/
void tensor_mont_fft(sfixn n, sfixn k, sfixn *A, sfixn wn, sfixn p);
void tensor_mont_fft(sfixn n, sfixn k, sfixn *A, sfixn wn, sfixn p, 
               const uni_mont_fft_plan &plan); 

/*****************************************************************************/
#endif // END OF FILE
