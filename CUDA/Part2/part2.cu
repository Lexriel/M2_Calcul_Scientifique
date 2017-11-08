/* part2.cu */

# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <sys/time.h>
# include <unistd.h>
# include "part2_cpu.h"
# include "part2_kernel.cu"
# include "part2_conf.h"


void compute_gpu(int n, int nb_iters) {

  int i, block_number;
  int *T, *T_aux, *T_device;

  
  
  T  = (int*) malloc(n*sizeof(int));
  T_aux  = (int*) malloc(n*sizeof(int));

  cudaMalloc( (void**) &T_device, n*sizeof(int) );

  
  for(i=0; i<n; i++)
    T[i]  = rand() % (nb_iters-1) + 1;


  cudaMemcpy( T_device, T, n*sizeof(int), cudaMemcpyHostToDevice );


/*  printf("T = [ ");
  for (i=0; i<n; i++)
    printf("%d ", T[i]);
  printf("]\n"); */


  // appel au kernel kernel_compute_gpu(int n, int nb_iters, int *T)
  block_number = n/BLOCK_SIZE;
  if ( (n % BLOCK_SIZE) != 0 )
    block_number++;
 kernel_compute_gpu<<<block_number, BLOCK_SIZE>>>(n, nb_iters, T_device);


  cudaMemcpy( T_aux, T_device, n*sizeof(int), cudaMemcpyDeviceToHost );
/*  printf("Avec compute_gpu, la copie du calcul sur T est donnée par :\n T_aux = [ ");
  for (i=0; i<n; i++)
    printf("%d ", T_aux[i]);
  printf("]\n"); */


 free(T);
 free(T_aux);
 cudaFree(T_device);

}



int main(int argc, char* argv[])
{
  if (argc < 3)
    {
      printf("Donner au moins 3 arguments.\n");
      return 0;
    }
        
  int n = atoi(argv[1]); // array size
  int nb_iters = atoi(argv[2]); // number of iterations
  timeval t1, t2, t3;
  double elapsedTime1, elapsedTime2;
  
int temp = time(NULL);

gettimeofday(&t1, NULL);
srand(temp);
compute_cpu(n,nb_iters);

gettimeofday(&t2, NULL);

srand(temp);
compute_gpu(n,nb_iters);
gettimeofday(&t3, NULL);

// compute and print the elapsed time in sec :
elapsedTime1  =  t2.tv_sec - t1.tv_sec; //sec
elapsedTime1 += (t2.tv_usec - t1.tv_usec) / 1000000.0; //us to sec
elapsedTime2  =  t3.tv_sec - t2.tv_sec; //sec
elapsedTime2 += (t3.tv_usec - t2.tv_usec) / 1000000.0; //us to sec

printf("Processing time for compute_cpu : %1.3lf (s) \n", elapsedTime1);
printf("Processing time for compute_gpu : %1.3lf (s) \n", elapsedTime2);


}
