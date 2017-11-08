#include "taylor_shift.h"
#include "taylor_shift_cpu.h"
#include "taylor_shift_conf.h"
#include "taylor_shift_kernel.h"

int main(int argc, char* argv[])
{
  // temporal data
  float total_time;
  cudaEvent_t start, stop;     /* Initial and final time */

  // TIME
  cudaEventCreate(&start); 
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // declaration of variables
  int n, e, p;
  double pinv;
  char name_file[100];

  error_message(argc);
  p = atoi(argv[2]);
  pinv = (double) 1/p;
  n = size_file(argv[1]);
  e = (int) log2((double) n);

  sprintf(name_file, "Pol%d.shiftGPU_%d.dat\0", e, p);

  taylor_shift_GPU(n, e, argv[1], p, pinv);

  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);
  cudaEventDestroy(stop);
  total_time /= 1000.0;
  printf("%d %.6f ", e, total_time);

  return 0;
}
