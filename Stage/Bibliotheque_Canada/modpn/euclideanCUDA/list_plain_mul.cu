/**********************************************************
* This code is intended for for computing multiplication  *
* for sub-product tree.                                   *
*             2*poly_length  <= T           *
* According to this constraint subproduct tree level      *
* where poly_length is not more than 129 is possible by   *
*  this code.
*ceil((poly_on_layer*0.5)/(floor(T/2*poly_length)) < 2^16 *
***********************************************************/

#include<iostream>
#include <ctime>
#include<cmath>

#define BASE_1 31

using namespace std;

const int Tmul = 512;
/***************************************************************
* one thread is responsible for computing one coefficient of c *
****************************************************************/

typedef int            sfixn;
typedef unsigned int   usfixn;


__device__ __host__ __inline__ 
sfixn add_mod(sfixn a, sfixn b, sfixn P) {
    sfixn r = a + b;
    r -= P;
    r += (r >> BASE_1) & P;
    return r;
}

__device__ __host__ __inline__ 
sfixn mul_mod(sfixn a, sfixn b, int P) {
    double ninv = 1 / (double)P;
    sfixn q  = (sfixn) ((((double) a) * ((double) b)) * ninv);
    sfixn res = a * b - q * P;
    res += (res >> BASE_1) & P;
    res -= 	P;
    res += (res >> BASE_1) & P;
    return res;
}




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



sfixn add_modCPU(sfixn a, sfixn b, sfixn P) {
    sfixn r = a + b;
    r -= P;
    r += (r >> BASE_1) & P;
    return r;
}


sfixn mul_modCPU(sfixn a, sfixn b, int P) {
    double ninv = 1 / (double)P;
    sfixn q  = (sfixn) ((((double) a) * ((double) b)) * ninv);
    sfixn res = a * b - q * P;
    res += (res >> BASE_1) & P;
    res -= 	P;
    res += (res >> BASE_1) & P;
    return res;
}

void list_mul(sfixn *M , int n, int start_offset, int length_poly, int poly_on_layer, int result_offset, int p)
{
	/********************************************************
	* Number of threads responsible for one multiplication  *
	*********************************************************/	
	int threadsForAmul = 2*length_poly;

	/************************************************
	* Number of multiplications in one thread block *
	*************************************************/
	int mulInThreadBlock = (int)floor((double)Tmul/(double)threadsForAmul);

	/****************************
	* Number of blocks required *
	****************************/
	int blockNo = (int)ceil( ((double)poly_on_layer/(double) mulInThreadBlock)*0.5  );	

	sfixn *Mgpu;
	cudaMalloc((void **)&Mgpu, sizeof(sfixn)*n);
	cudaThreadSynchronize();
	cudaMemcpy(Mgpu, M, sizeof(sfixn)*n, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();



	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	
	listPlainMulGpu<<<blockNo, Tmul>>>(Mgpu, start_offset, length_poly, poly_on_layer, threadsForAmul, mulInThreadBlock, p);
	
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float outerTime;
        cudaEventElapsedTime(&outerTime, start, stop);
	cout<<outerTime<<" miliseconds"<<endl;


	cudaFree(Mgpu);  
	/*
	cout<<"threadsForAmul: "<<threadsForAmul<<" mulInThreadBlock: "<<mulInThreadBlock<<" blockNo: " <<blockNo<<endl;
	int i, j, k, l, q, a, b, c;
	for(i = 0; i < poly_on_layer/2; ++i )
	{
		j = start_offset + 2*i*length_poly;
		k = j + length_poly;
		l = k + length_poly;		
		q = result_offset + i*(2*length_poly -1);
		for(c = q; c < q+(2*length_poly -1); ++c)		
			M[c] = 0;
		for(a = j; a < k; ++a)
		{
			c = q;
			for(b = k; b < l; ++b, ++c)
			{
				M[c] = add_modCPU(mul_modCPU(M[a], M[b], p), M[c], p);				 
			}
			++q;		
		}
	}

	//
	sfixn *temp = new sfixn[n];
	cudaMemcpy(temp, Mgpu, sizeof(sfixn)*n, cudaMemcpyDeviceToHost);
	//
	cudaFree(Mgpu);        
	//	


	for(i = 0; i < n; ++i)
	{	
		if(M[i] != temp[i])
		{	
			cout<<i<<" "<<M[i]<<" "<<temp[i]<<endl;
			break;
		}
	}
	cout<<endl;
	
	delete [] temp;
	//	
	*/
}



int main(int argc, char *argv[])
{	
	int p= 7, start_offset = 0, length_poly = 4, poly_on_layer = 4, result_offset ;	

	if (argc > 1) start_offset = atoi(argv[1]);
	if (argc > 2) length_poly = atoi(argv[2]);
        if (argc > 3) poly_on_layer = atoi(argv[3]);
	if (argc > 4) p = atoi(argv[4]);
	
	result_offset = start_offset + length_poly*poly_on_layer;
	

	/**************************************
	* poly_on_layer MUST BE DIVISIBLE BY 2 *
	***************************************/
	int n, i;
	n = result_offset + (2*length_poly-1)*(poly_on_layer/2);	
	sfixn *M = new int[n];
	
	
	cout<<endl;
	for(i = start_offset; i < result_offset; ++i)
		M[i] = rand()% p;
		
			
	
	
	
	cout<<endl;
	
	cout<<p<<" "<<start_offset<<" "<<length_poly<<" "<<poly_on_layer<<" "<<result_offset<<" "<<n<<endl;
	/*
	for(i = 0; i < n; ++i)	
		cout<<M[i]<<" ";
	cout<<endl;	
	*/
		
	list_mul(M , n, start_offset, length_poly, poly_on_layer, result_offset, p );
	
		

	delete [] M;
	
	return 0;
}
	
	
