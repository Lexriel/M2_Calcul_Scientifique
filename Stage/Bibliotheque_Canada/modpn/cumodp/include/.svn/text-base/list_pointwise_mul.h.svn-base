#ifndef _LIST_POINTWISE_MUL_H
#define _LIST_POINTWISE_MUL_H
#include <stdio.h>
#include <stdlib.h>


#include <stdio.h>
#include <stdlib.h>
#include "../include/inlines.h"
#include "../include/defines.h"
#include "../include/types.h"
#include "../include/printing.h"
#include "../include/cudautils.h"
#include "../include/rdr_poly.h"


#define N_TREAD 512 
#define PROCESS_PER_TRD 16
__device__ __global__ void list_pointwise_mul(sfixn *L, sfixn ln, sfixn p, double pinv, sfixn length_layer);
__device__ __global__ void list_expand_to_fft(sfixn *M, sfixn *L, sfixn length_poly, sfixn ln, sfixn start_offset, sfixn length_layer);
__device__ __global__ void list_truncate_for_invfft(sfixn *L, sfixn *K, sfixn ln);
__device__ __global__ void list_shrink_after_invfft(sfixn *L, sfixn *S, sfixn ln, sfixn length_result, sfixn length_layer);


__host__ void CPU_list_pointwise_mul(sfixn *L, sfixn ln, sfixn num_poly, sfixn p, double pinv);
__host__ void CPU_list_expand_to_fft(sfixn *M, sfixn *L, sfixn length_M, sfixn length_poly, sfixn ln, sfixn start_offset);
__host__ void CPU_list_truncate_for_invfft(sfixn *L, sfixn *K, sfixn ln, sfixn num_poly);
__host__ void CPU_list_shrink_after_invfft(sfixn *L, sfixn *S, sfixn ln, sfixn length_result, sfixn num_poly);
#endif

