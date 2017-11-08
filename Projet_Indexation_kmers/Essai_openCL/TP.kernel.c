

/*
__kernel void square(
   __global float* input,
   __global float* output,
   const unsigned int count)
{    
  int i = get_global_id(0);
  if (i < count) 
    output[i] = input[i] * input[i];
}
*/


__kernel void find_kmers_GPU(
   __global unsigned long* input1,
   __global unsigned long* input2,
   __global unsigned long* output,
   unsigned long n)
{
  unsigned long i = get_global_id(0);
  unsigned long result = 0;

  if (i < n)
    output[i] = output[i] + min(input1[i], input2[i]);
}
