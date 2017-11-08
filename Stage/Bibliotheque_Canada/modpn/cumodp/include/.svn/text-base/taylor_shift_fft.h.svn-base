#ifndef _TAYLOR_SHIFT_FFT_H_
#define _TAYLOR_SHIFT_FFT_H_


__global__ void mult_adjust_GPU(sfixn *Polynomial_shift, sfixn *fft, sfixn n, sfixn local_n, sfixn winv, sfixn p, double pinv);


//__global__ void transfert_array_fft_GPU(sfixn *fft, sfixn *Polynomial_shift, sfixn *Monomial_shift, sfixn n, sfixn local_n);


__global__ void transfert_array_fft_GPU(sfixn *fft, sfixn *Mgpu, sfixn n, sfixn local_n);


__global__ void full_monomial(sfixn *Mgpu, sfixn *Monomial_shift, sfixn n, sfixn local_n);


#endif // _TAYLOR_SHIFT_FFT_H_
