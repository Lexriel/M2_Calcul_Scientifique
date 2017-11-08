#include <stdio.h>

int main() {
  int dev;
  cudaError_t err;
  struct cudaDeviceProp prop;

  /* dev is the current device for the calling host thread */
  err = cudaGetDevice(&dev);

  if (err != cudaSuccess) {
    printf("Err cudaGetDevice : %d\n", err);
    exit(1);
  }
  printf("dev : %d\n", dev);
 
  /* prop is a structure containing the properties of the device dev */  
  if (cudaGetDeviceProperties(&prop, dev)) {
    printf("Err cudaGetDeviceProperties\n");
    exit(1);
  }

  /* Display the property total constant memory of the device dev */   
  printf("totalConstMem : %ld\n", prop.totalConstMem);

  /* This code is executing well if CUDA is correctly working */

  return 0;
}
