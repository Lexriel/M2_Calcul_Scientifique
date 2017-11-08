#include "taylor_shift_conf.h"
#include "taylor_shift_cpu.h"
#include "taylor_shift_kernel.h"
#include "taylor_shift.h"

/* Important to notice :

  n  : number of coefficients of the polynomial considered
 n-1 : degree of the polynomial considered
  p  : prime number, it must be greater than n
*/


// Taylor_shift procedure
void taylor_shift_GPU(int n, int e, char *file, int p, double pinv)
{

  // declaration of variables
  int i, nb_blocks, local_n;

  int *Factorial_device;
  int *Polynomial, *Polynomial_device;
  int *Monomial_shift_device;
  int *temp;
  int *Mgpu;

  // temporal data
  float cpu_time, gpu_time, outerTime;
  cudaEvent_t start, stop, start2, stop2;     /* Initial and final time */

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // beginning parameters
//  error_message(argc);
//  p = atoi(argv[2]);
//  pinv = (double) 1/p;
//  n = size_file(argv[1]);
  local_n = 2;
//  e = (int) log2((double) n);
  stock_file_in_array(file, n, Polynomial);


  // We will use this for each step of our computation
  int *Polynomial_shift_device[e];


  // display parameters
  printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\
                    TAYLOR_SHIFT ON GPU\n\n\
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

  printf("\nPARAMETERS :\n------------\n\n");
  printf("  * n = %d\n", n);
  printf("  * e = %d\n", e);
  printf("  * p = %d\n", p);
  printf("  * local_n = %d\n", local_n);
  printf("  * pinv = %0.20lf\n", pinv);

  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time, start, stop);
  cudaEventDestroy(stop);
  cpu_time /= 1000.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);


  // Create the array Factorial
  cudaMalloc( (void **) &Factorial_device, (n+1) * sizeof(int) );
  nb_blocks = number_of_blocks(n+1);

  float identity_GPU_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  identity_GPU<<<nb_blocks, NB_THREADS>>>(Factorial_device, n+1);
//  cudaThreadSynchronize();
  printf("\n      --> identity_GPU done\n");

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&identity_GPU_time, start2, stop2);
  cudaEventDestroy(stop2);
  identity_GPU_time /= 1000.0;
  printf("identity_GPU_time = %.6f\n", identity_GPU_time);

  float fact_GPU_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  nb_blocks = number_of_blocks(n/2);
  create_factorial_GPU<<<nb_blocks, NB_THREADS>>>(Factorial_device + 1, n, e, p, pinv);
//  cudaThreadSynchronize();
  printf("\n      --> create_factorial_GPU done\n");

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&fact_GPU_time, start2, stop2);
  cudaEventDestroy(stop2);
  fact_GPU_time /= 1000.0;
  printf("fact_GPU_time = %.6f\n", fact_GPU_time);


  // display Factorial_device
/*  temp = (int*) calloc(n+1, sizeof(int));
  cudaMemcpy( temp, Factorial_device, (n+1)*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("\nFactorial_device :\n");
  display_array(temp, (n+1));
  free(temp); */


  float develop_xshift_GPU_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  // Create the array of the (x+1)^i
  cudaMalloc( (void **) &Monomial_shift_device , n * sizeof(int) ); // n+1
//  cudaThreadSynchronize();
  nb_blocks = number_of_blocks(n);
  develop_xshift_GPU<<<nb_blocks, NB_THREADS>>>(Monomial_shift_device, n, Factorial_device, p, pinv);
  cudaThreadSynchronize();
  printf("\n      --> develop_xshift_GPU done\n");
  cudaFree(Factorial_device);

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&develop_xshift_GPU_time, start2, stop2);
  cudaEventDestroy(stop2);
  develop_xshift_GPU_time /= 1000.0;
  printf("develop_xshift_time = %.6f\n", develop_xshift_GPU_time);


  // display Monomial_shift_device
/*  temp = (int*) calloc(n+1, sizeof(int));
  cudaMemcpy( temp, Monomial_shift_device, (n+1)*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("\nMonomial_shift_device :\n");
  display_array(temp, n+1);
  free(temp);*/


  /* ************************************************************

                     1st step : initialization 

     ************************************************************ */
  float init_GPU_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  printf("\n\nStep 1 :\n-------- \n");
  cudaMalloc( (void **) &Polynomial_device, n * sizeof(int) );
//  cudaThreadSynchronize();
  cudaMemcpy( Polynomial_device, Polynomial, n*sizeof(int), cudaMemcpyHostToDevice );
//  cudaThreadSynchronize();
  free(Polynomial);
  cudaMalloc( (void **) &Polynomial_shift_device[0], n * sizeof(int) );
//  cudaThreadSynchronize();


  // initialize polynomial_shift
  nb_blocks = number_of_blocks(n/2);
  init_polynomial_shift_GPU<<<nb_blocks, NB_THREADS>>>(Polynomial_device, Polynomial_shift_device[0], n, p);
//  cudaThreadSynchronize();
  printf("\n      --> init_polynomial_shift_GPU done\n");

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&init_GPU_time, start2, stop2);
  cudaEventDestroy(stop2);
  init_GPU_time /= 1000.0;
  printf("init_GPU_time = %.6f\n", init_GPU_time);


  // display Polynomial_shift_device[0]
/*  temp = (int*) calloc(n, sizeof(int));
  cudaMemcpy( temp, Polynomial_shift_device[0], n*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  cudaFree(Polynomial_device);
  printf("\nPolynomial_shift_device[0] :\n");
  display_array(temp, n);
  free(temp);*/



  /* ************************************************************

                              next steps 

     ************************************************************ */

int polyOnLayerCurrent = n/2;
int mulInThreadBlock;

cudaMalloc((void **)&Mgpu, n * sizeof(int));
//cudaThreadSynchronize();


for (i=1; i<e; i++)
{
  if (i < 9)
    printf("\n\nStep %d :\n-------- \n\n", i+1);
  else
    printf("\n\nStep %d :\n--------- \n\n", i+1);

  printf("  * local_n = %d\n", local_n);
  printf("  * B = %d\n", 2 * local_n);
  printf("  * polyOnLayerCurrent = %d\n", polyOnLayerCurrent);

  float transfert_GPU_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  cudaMalloc( (void **) &Polynomial_shift_device[i], n * sizeof(int) );
//  cudaThreadSynchronize();


  // transfer the polynomials which will be computed
  nb_blocks = number_of_blocks(n);
  transfert_array_GPU<<<nb_blocks, NB_THREADS>>>(Mgpu, Polynomial_shift_device[i-1], Monomial_shift_device, n, local_n);
//  cudaThreadSynchronize();
  printf("\n      --> transfert_array_GPU2 on Mgpu done\n");

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&transfert_GPU_time, start2, stop2);
  cudaEventDestroy(stop2);
  transfert_GPU_time /= 1000.0;
  printf("transfert_GPU_time = %.6f\n", transfert_GPU_time);


  // display Mgpu
/*  temp = (int*) calloc(n, sizeof(int));
  cudaMemcpy( temp, Mgpu, n*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("\nMgpu :\n", i);
  display_array(temp, n);
  free(temp);*/

  float listPlainMulGpu_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);
  /* compute the product of the polynomials in Mgpu ('P2 * Bin' with Bin the array of binomials) and store them in Polynomial_shift_device[i] shifted at the right for the multiplication by x so do [( (x+1)^i - 1 ) / x ] * P2(x+1), then multiply it by x so we have [(x+1)^i - 1] * P2(x+1) */
  mulInThreadBlock = (int) floor((double) NB_THREADS / (double) (2*local_n));//1;
  nb_blocks = (int) ceil(((double) polyOnLayerCurrent / (double) mulInThreadBlock) * 0.5);
//  printf("\n  * mulInThreadBlock = %d\n", mulInThreadBlock);
//  printf("  * nb_blocks = %d\n", nb_blocks);
  listPlainMulGpu_and_right_shift_GPU<<<nb_blocks, NB_THREADS>>>(Mgpu, Polynomial_shift_device[i], local_n, polyOnLayerCurrent, 2*local_n, mulInThreadBlock, p, pinv);
//  cudaThreadSynchronize();
  printf("\n      --> listPlainMulGpu_and_right_shift_GPU done\n");

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&listPlainMulGpu_time, start2, stop2);
  cudaEventDestroy(stop2);
  listPlainMulGpu_time /= 1000.0;
  printf("listPlainMulGpu_time = %.6f\n", listPlainMulGpu_time);


  // listPlainMulGpu(array of polynomials to multiplicate, array product of these multiplications, polynomial length, number of polynomials, number of threads for a mul, number of multiplication in a block, p)


  // display Polynomial_shift_device[i] = [(x+1)^i - 1] * P2(x+1)
/*  temp = (int*) malloc(n * sizeof(int));
  cudaMemcpy( temp, Polynomial_shift_device[i], n*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("\nPolynomial_shift_device[%d] after listPlainMulGpu_and_right_shift_GPU:\n", i);
//  display_array(temp, n);
  free(temp);*/


  float semi_add_GPU_time;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  cudaEventRecord(start2, 0);

  // add [(x+1)^i - 1] * P2(x+1) with P2(x+1) then we get (x+1)^i * P2(x+1)
  // then do P1(x+1) + (x+1)^i*P2(x+1)
  nb_blocks = number_of_blocks(n/2);
  semi_add_GPU<<<nb_blocks, NB_THREADS>>>(Polynomial_shift_device[i], Mgpu, Polynomial_shift_device[i-1], n, local_n, p);
//  cudaThreadSynchronize();
  printf("\n      --> semi_add_GPU done\n");

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&semi_add_GPU_time, start2, stop2);
  cudaEventDestroy(stop2);
  semi_add_GPU_time /= 1000.0;
  printf("semi_add_GPU_time = %.6f\n", semi_add_GPU_time);


  // display Polynomial_shift_device[i]
//  temp = (int*) malloc(n * sizeof(int));
//  cudaMemcpy( temp, Polynomial_shift_device[i], n*sizeof(int), cudaMemcpyDeviceToHost );
//  cudaThreadSynchronize();
//  printf("\nPolynomial_shift_device[%d] after semi_add_GPU:\n", i);
//  display_array(temp, n);
//  free(temp);

  polyOnLayerCurrent /= 2;
  local_n *= 2;
}

temp = (int*) malloc(n * sizeof(int));
cudaMemcpy( temp, Polynomial_shift_device[e-1], n*sizeof(int), cudaMemcpyDeviceToHost );
//cudaThreadSynchronize();
printf("\n      --> cudaMemcpy on temp done\n");
//cudaFree(temp_device);


  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  cudaEventDestroy(stop);
  gpu_time /= 1000.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);


  // stockes the array of Newton's coefficients in a file
  char name_file[30];
  sprintf(name_file, "Pol%d.shiftGPU.dat\0", e);
  stock_array_in_file(name_file, temp, n);
  printf("\n      --> Polynomial_shift_device stored in %s done\n", name_file);


  // deallocation of the last arrays
  free(temp);
  cudaFree(Monomial_shift_device);
  cudaFree(Mgpu);
  for (i=0; i<e; i++)
    cudaFree(Polynomial_shift_device[i]);


  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&outerTime, start, stop);
  cudaEventDestroy(stop);
  outerTime /= 1000.0;
  cpu_time += outerTime;


  // execution time
  printf("\n  * cpu_time = %.6f s\n", cpu_time);
  printf("  * gpu_time = %.6f s\n", gpu_time);
}
