#include "../include/list_plain_mul.h"
/**********************************************************
* This code is intended for for computing multiplication  *
* for sub-product tree.                                   *
*             2*poly_length  <= T           *
* According to this constraint subproduct tree level      *
* where poly_length is not more than 129 is possible by   *
*  this code.
*ceil((poly_on_layer*0.5)/(floor(T/2*poly_length)) < 2^16 *
***********************************************************/


/***************************************************************
* one thread is responsible for computing one coefficient of c *
****************************************************************/
/*
__global__ void listPlainMulGpu( sfixn *Mgpu1, sfixn *Mgpu2 , int length_poly, int poly_on_layer, int threadsForAmul, int mulInThreadBlock, int p)
{
	__shared__ sfixn sM[2*Tmul];	
	int mulID= ((threadIdx.x/threadsForAmul) + blockIdx.x*mulInThreadBlock);
	
	if( mulID < (poly_on_layer/2) && threadIdx.x < threadsForAmul*mulInThreadBlock)
	{

		int j = ( mulID* length_poly*2);		
		int q = ( mulID*(2*length_poly-1));

		int t = (threadIdx.x/threadsForAmul);
		int u = threadIdx.x % threadsForAmul;

		int s = t*(4*length_poly-1);
		int k = s + length_poly;
		int l = k + length_poly;
		int c = l+u;
		int a, b, i;

		sM[s+u] = Mgpu1[j + u];
		__syncthreads();
		
		if(u != (2*length_poly-1) )
		{
			if(u < length_poly)
			{
				a = s;
				b = k + u;			
				sM[c] =  mul_mod(sM[a],sM[b],p);
				++a; --b;
				for(i = 0; i < u; ++i, ++a, --b)
					sM[c] =  add_mod(mul_mod(sM[a],sM[b],p),sM[c] ,p);
				

				Mgpu2[q+u] = sM[c];				
			}
			else
			{
				b = l - 1;
				a = (u - length_poly) + 1 + s;
				sM[c] =  mul_mod(sM[a],sM[b],p);
				++a; --b;
				
				int tempU = u;
				u = (2*length_poly-2) - u;
				for(i = 0; i < u; ++i, ++a, --b)
					sM[c] =  add_mod(mul_mod(sM[a],sM[b],p),sM[c] ,p);
				

				Mgpu2[q+tempU] = sM[c];			
			}	
		}
	}		
}
*/

__global__ void listPlainMulGpu( sfixn *Mgpu, int start_offset, int length_poly, int poly_on_layer, int threadsForAmul, int mulInThreadBlock, int p)
{
	__shared__ sfixn sM[2*Tmul];	
	int mulID= ((threadIdx.x/threadsForAmul) + blockIdx.x*mulInThreadBlock);
	
	if( mulID < (poly_on_layer/2) && threadIdx.x < threadsForAmul*mulInThreadBlock)
	{

		int j = start_offset + ( mulID* length_poly*2);		
		int q = start_offset + ( poly_on_layer*length_poly) + ( mulID*(2*length_poly-1));

		int t = (threadIdx.x/threadsForAmul);
		int u = threadIdx.x % threadsForAmul;

		int s = t*(4*length_poly-1);
		int k = s + length_poly;
		int l = k + length_poly;
		int c = l+u;
		int a, b, i;

		sM[s+u] = Mgpu[j + u];
		__syncthreads();
		
		if(u != (2*length_poly-1) )
		{
			if(u < length_poly)
			{
				a = s;
				b = k + u;			
				sM[c] =  mul_mod(sM[a],sM[b],p);
				++a; --b;
				for(i = 0; i < u; ++i, ++a, --b)
					sM[c] =  add_mod(mul_mod(sM[a],sM[b],p),sM[c] ,p);
				/*
				Mgpu[j + u] = i;
				*/

				Mgpu[q+u] = sM[c];				
			}
			else
			{
				b = l - 1;
				a = (u - length_poly) + 1 + s;
				sM[c] =  mul_mod(sM[a],sM[b],p);
				++a; --b;
				
				int tempU = u;
				u = (2*length_poly-2) - u;
				for(i = 0; i < u; ++i, ++a, --b)
					sM[c] =  add_mod(mul_mod(sM[a],sM[b],p),sM[c] ,p);
				/*
				Mgpu[j + u] = i;
				*/

				Mgpu[q+tempU] = sM[c];			
			}	
			
	
		}/*
		else			
		{		
			Mgpu[j + u] = -1;				
		}*/
	}		
}



