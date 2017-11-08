# include "taylor_shift.cu"
# include "taylor_shift_cpu.cu"
# include "taylor_shift_conf.h"
# include "taylor_shift_kernel.cu"

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

  error_message(argc);
  p = atoi(argv[2]);
  pinv = (double) (1<<30)/p;  // 1/p;
  n = size_file(argv[1]);
  e = (int) log2((double) n);

  taylor_shift_GPU(n, e, argv[1], p, pinv);

  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);
  cudaEventDestroy(stop);
  total_time /= 1000.0;
  printf("  * total_time = %.6f s\n\n", total_time);

  return 0;
}
