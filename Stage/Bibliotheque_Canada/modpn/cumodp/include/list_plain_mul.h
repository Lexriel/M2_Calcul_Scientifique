
#ifndef LIST_PLAIN_MUL
#define LIST_PLAIN_MUL
#include "../include/subproduct_tree.h"
const int Tmul = 512;
//__global__ void listPlainMulGpu2( sfixn *Mgpu1, sfixn *Mgpu2 , int length_poly, int poly_on_layer, int threadsForAmul, int mulInThreadBlock, int p);
__global__ void listPlainMulGpu(sfixn *M_dev, sfixn start_offset, sfixn length_poly, sfixn num_poly, sfixn threadsForAmul, sfixn mulInThreadBlock, sfixn p);
#endif
