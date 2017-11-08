#include "../include/fastPolyEvaluation.h"

struct PolyEvalSteps check; // this will hold all the intermediate results.

/*
The polynomial in source start from 0.
NO OFFSET.
The number of blocks is exactly the same as the number of polynomials.
The number of threads is exactly the same as the poly_length-1.
So it works with only poly of length 512.
total number of polynomials is 2^16 -1;
This copy list of polynomials in source to destination.
Each of the polynomial in source is of length_poly.
In dest all coefficients are copied except the most significant one.
*/

__global__ void copyMgpu(sfixn *dest, sfixn *source, int length_poly)
{
	dest[blockIdx.x *(length_poly-1) + threadIdx.x] = source[blockIdx.x *length_poly + threadIdx.x];	
}


/*
It simply make X a zero vector
*/
__global__ void allZero( sfixn *X, int n)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n) X[tid] = 0;
}

/*
It simply negate X
*/
__global__ void allNeg( sfixn *X, int n, sfixn p)
{
        int tid= blockIdx.x*blockDim.x + threadIdx.x;
        if(tid < n) X[tid] = neg_mod(X[tid],p);
}




/*
The polynomials in source and destination start from 0.
NO OFFSET.

The number of blocks is:
(int)ceil((double)(n/2)/(double)Tmax)
where n = size of source and destination or source

The number of threads is Tmax.

length of each polynomial in source is l.
length of source and destination is 2*n each.
Each of the corresponding coefficient of source 
will be added and stored at the significant l/2 coefficient 
of source.
n must divisible by 2.
*/


__global__ void pointAdd2(sfixn *dest, sfixn *source, int l,  int n, int p)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n)
	{
		int a = tid/l;
		int b = tid%l;
		dest[l*(2*a+1) +b ] = add_mod(source[l*(2*a+1) +b], source[l*2*a + b], p);	
	}		
}

/*
The polynomials in X and Y start from 0.
NO OFFSET.

The number of blocks is:
(int)ceil((double)L/(double)(Tmax*2))
where L = size of source and destination or source.
The number of threads is Tmax.

length of each polynomial in X and Y is l.
But L = 2*l**no_of_poly
The significant l coefficients are put to 0.
L must be divisible by 2.
*/

__global__ void  zeroInbetween(sfixn *X, sfixn *Y, int n, int l)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < n)
	{
		int a = tid/l;
		int b = tid%l;
		X[l*(2*a+1) + b ] = 0;
		Y[l*(2*a+1) + b ] = 0;
	}
}

/*
The polynomial in source start from 0.
NO OFFSET.

The number of blocks is:
(int)ceil((double)n/(double)Tmax)
where n = size of dest or source

The number of threads is Tmax.


total number of polynomials is 2^16 -1;
dest will store the point wise multiplications.
*/
__global__ void pointMul(sfixn *dest, sfixn *source, int n, int p)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n)
		dest[tid] = mul_mod(source[tid],dest[tid], p);	
}

__global__ void scalarMul(sfixn *A, sfixn ninv, int L, int p)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < L)
		A[tid] = mul_mod(A[tid], ninv, p);	
}


/*
The polynomial in source start from 0.
NO OFFSET.

The number of blocks is:
(int)ceil((double)n/(double)Tmax)
where n = size of dest or source

The number of threads is Tmax.


total number of polynomials is 2^16 -1;
dest will store the point wise multiplications.
*/


__global__ void pointAdd(sfixn *dest, sfixn *source, int n, int p)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n)
		dest[tid] = add_mod(source[tid],dest[tid], p);	
}

/*			

'sfixn *Mgpu1' is a list of 'poly_on_layer' polynomials. 
'poly_on_layer' is an even number.
Each polynomials has length of 'length_poly'.

'sfixn *Mgpu2' stores the list multiplications of consecutive pairs
i.e multiplication of poly i and poly i+1 in Mgpu1 is stored in poly i/2_{th}
slot of Mgpu2.

The length of a poly in Mgpu2 is '2*length_poly - 1'.
We dedicate 2*length_poly threads for a poly multiplication.
so the number of threads responsible for one poly multiplication
'threadsForAmul = 2*length_poly'.

The multiplication is done using shared memory.
The same shared memory can not be accesses by
two different thread blocks. So a thread block
is responsible for doing a number of poly multiplication.
But the reverse is not true. That means two or more 
thread block can not do a single poly multiplication together.
Thats why the number of multiplications done by one thread block
'mulInThreadBlock = (int)floor((double)Tmul/(double)threadsForAmul)'
Where 'Tmul' is the number of threads in a thread block. We keep it as 512.

These constraint us to the limitation of our poly multiplications.
First 2*length_poly <= Tmul.
poly_on_layer/2 <= maximum number of thread block.

*/

__global__ void listPlainMulGpu( sfixn *Mgpu1, sfixn *Mgpu2 , int length_poly, int poly_on_layer, int threadsForAmul, int mulInThreadBlock, int p)
{

	__shared__ sfixn sM[2*Tmul];
	/*
	sM is the shared memory where the all the coefficients and intermediate multiplications results
	are stored. For each multiplication it reserve 4*length_poly -1 spaces.
	mulID is the multiplication ID. It refers to the poly in Mgpu2 on which it will work.
	mulID must be less than (poly_on_layer/2).
	*/	
	int mulID= ((threadIdx.x/threadsForAmul) + blockIdx.x*mulInThreadBlock);
	
	if( mulID < (poly_on_layer/2) && threadIdx.x < threadsForAmul*mulInThreadBlock)
	{
		/*
		The next 10 lines of code copy the polynomials in Mgpu1 from global memory to shared memory.
		Each thread is responsible of copying one coefficient.
		A thread will copy a coefficient from Mgpu1[( mulID* length_poly*2)...( mulID* length_poly*2) + length_poly*2 -1] 
		j+u gives the right index of the coefficient in Mgpu1.

		In sM, the coefficients are stored at the lower part.
		t will find the right (4*length_poly-1) spaced slot for it.
		s gives the start index of its right slot.
		s+u gives right position for the index.

		
		*/

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
			/*
			In the multiplication space, the half of the leading coefficients 
			are computed differently than the last half. Here the computation of 
			first half are shown. the last half is shown in else statement.
			In both cases sM[c] is the cofficient on which this thread will work on.
			sM[a] is the coefficient of one poly.
			sM[b] is the coefficient of the other poly.
			*/
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

/*
This kernel can compute the
inverse of a poly of length 
257 mod x^{2^8}.
The number of threads Tinv
in a thread block is small.
We fix Tinv = 16.
Each thread is working 
with a polynomial.

*/

__global__ void listPolyinv( sfixn *Mgpu, sfixn *invMgpu,  int poly_on_layer, int prime)
{
	int PolyId = blockIdx.x*blockDim.x + threadIdx.x;	
	
	if(PolyId < poly_on_layer)
	{
		const int  SIZE = 256;
		const int r = 8;
		int i, j;
		sfixn f[SIZE+1], g[SIZE], gtemp[SIZE], gtemp2[SIZE];


		j = (PolyId+1)*(SIZE+1) -1;		
		for(i = 0; i <= SIZE; ++i, --j)
			f[i] = Mgpu[j];

		g[0] = 1;
		gtemp[0] = 1;
		gtemp2[0] = 1;
		for(i = 1; i < SIZE; ++i)		
		{
			g[i] = 0;
			gtemp[i] = 0;
			gtemp2[i] = 0;
		}
		int  p, q, start = 1, end = 2;	
		for(i = 1; i <= r; ++i)
		{
			for(j = start; j < end; ++j)
			{
				for(p = 0, q = j; p < SIZE && q >= 0; ++p, --q)
					gtemp[j] = add_mod(mul_mod(g[p], f[q], prime), gtemp[j] ,prime); 
				gtemp[j] = neg_mod(gtemp[j], prime);	
			} 

			for(j = start; j < end; ++j)
				for(p = 0, q = j; p < SIZE && q >= start; ++p, --q)
					g[j] = add_mod(mul_mod(gtemp[q],gtemp2[p], prime),g[j], prime); 			


			for(j = start; j < end; ++j)
				gtemp2[j] = g[j];
			
			start = start*2;
			end = end*2;
		}		
		for(j = 0, i = PolyId*(SIZE); j < SIZE; ++i, ++j)
			invMgpu[i] = g[j];
	}
}



/*
	const int Tmax = 512;
	int blockNo = (int)ceil((double)(poly_on_layer*length_poly)/(double)(Tmax));
	listReversePloy<<<blockNo, Tmax>>>(revMgpu, Mgpu, length_poly, poly_on_layer);
Mgpu is the array polynomoals. 
revMgpu reverse each poly in Mgpu.
of legth length_poly in reverse order. 

(poly_on_layer*length_poly) < 2^24

*/
__global__ void listReversePoly(sfixn *revMgpu, sfixn *Mgpu, int length_poly, int poly_on_layer)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < poly_on_layer*length_poly)
	{
		int polyID = tid/length_poly;
		int offset= tid%length_poly;
		revMgpu[(polyID+1)*length_poly - offset -1  ] = Mgpu[polyID*length_poly + offset];
	}
}

__global__ void listCpLdZeroPoly(sfixn *B, sfixn *A, int length_poly, int poly_on_layer)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < poly_on_layer*length_poly)
        {
                int polyID = tid/length_poly;
                int offset= tid%length_poly;
		if(offset != length_poly -1)
                	B[polyID*length_poly + offset]=  A[polyID*length_poly + offset];
		else
			 B[polyID*length_poly + offset]= 0;
        }

}


/*
	const int Tmax = 512;
	int newLength = newDegree+1;
	int blockNo = (int)ceil((double)(poly_on_layer*newLength)/(double)(Tmax));

	listPolyDegInc<<<blockNo, Tmax>>>(Mgpu, extDegMgpu, length_poly, poly_on_layer, newLength);
Mgpu is the array of M polynomoals. 
"extDegMgpu" is the list of "poly_on_layer" polynomials.
extDegMgpu[0..poly_on_layer*newLength]
Each polynomial is same as corresponding poly in Mgpu
Except each of its degree is equal to newDegree.
The extra coefficients are padded by zero.

(poly_on_layer*newLength]) < 2^24

*/
__global__ void listPolyDegInc( sfixn *Mgpu, sfixn *extDegMgpu, int length_poly, int poly_on_layer, int newLength)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < poly_on_layer*newLength)
	{
		int polyID = tid/newLength;
		int offset= tid%newLength;		
		if(offset < length_poly) 
			extDegMgpu[polyID*newLength + offset] = Mgpu[ polyID*length_poly + offset];
		else 
			extDegMgpu[polyID*newLength + offset] = 0;		
	}
}


/*
The polynomial in source start from 0.
NO OFFSET.

The number of blocks is:
(int)ceil((double)n/(double)Tmax)
where n = half of the size of source and destination
dest array is of poly of length l padded by l extra zero
in upper part. 
source is of poly of length 2*l.

The lower part (lower l coefficients) of each poly in source 
are copied to the zero part of the corresponding poly in destination.

*/


__global__ void listCpUpperCuda(sfixn *dest, sfixn *source, int n, int l)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n)
	{
		int x = tid/l;
		x = (x*l) + tid;
		dest[x + l] = source[x];	
	}
}


/*
The polynomial in source start from 0.
NO OFFSET.

The number of blocks is:
(int)ceil((double)n/(double)Tmax)
where n = half of the size of source.
dest array is of poly of length l. 
source is of poly of length 2*l.

The lower part (lower l coefficients) of each poly in source 
are copied to the corresponding poly in destination.

*/


__global__ void listCpLowerCuda(sfixn *dest, sfixn *source, int n, int l)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n)
	{
		int x = tid/l;
		int y = tid%l;
		dest[tid] = source[x*2*l + y];	
		//dest[tid] = x;
	}
}





/*
NO START OFFSET.
 totalLength = length(dest)/2;

The number of blocks is:
(int)ceil((double)(num_poly*l)/(double)Tmax)
where l = poly_length/2.

The lower l coeffs of each poly in source and
the upper l coeffs of each poly in dest will be added
and stored in the upper l coeffs of dest.
*/


__global__ void list2wayCp(sfixn *dest, sfixn *source, int l, int totalLength, sfixn p)
{
        int tid= blockIdx.x*blockDim.x + threadIdx.x;
        if(tid < totalLength)
        {
                int L = 2*l;

                int x = tid/l;
                int y = tid%l;
                dest[x*L+l+y] = add_mod(dest[x*L+l+y] , source[x*L +y], p);
        }
}


/*
Mgpu holds the numPoints*2 poits x_0, x_1...
M1gpu will create numPoints polynomials (X - x_i)
if rightSubtree == 0, it works with first half of Mgpu.
else it works with the last half of Mgpu.
The code is tested for numpoints = 2^24

*/

__global__ void	leavesSubproductTree(sfixn *M1gpu, sfixn *Mgpu, int numPoints, int rightSubtree)
{
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < numPoints)
	{
		M1gpu[tid*2] = -Mgpu[rightSubtree*numPoints + tid ];
		M1gpu[tid*2+1] = 1;		
	}
}



void subProductTree(int k, sfixn p)
{
	int polyLengthCurrent = 2; // the length of poly at leaves.
	int polyOnLayerCurrent = 1L << (k-1); //the number of poly in the leaves of each subtree. 
	int polyLengthNext, polyOnLayerNext;
	int threadsForAmul, mulInThreadBlock, blockNo;
	int L = 1L << (k-1); // spaces required for a level in any subtree from plainMulLimit
	int l;
	sfixn w, winv, ninv;
	sfixn *Al, *Bl, *Cl; 
	cudaMalloc((void **)&Al,    sizeof(sfixn)*L); // temporary storage required for doing FFT based multiplication.
	cudaMalloc((void **)&Bl,    sizeof(sfixn)*L); // temporary storage required for doing FFT based multiplication.			
	sfixn *Ar, *Br, *Cr; 
	cudaMalloc((void **)&Ar,    sizeof(sfixn)*L); // temporary storage required for doing FFT based multiplication.
	cudaMalloc((void **)&Br,    sizeof(sfixn)*L); // temporary storage required for doing FFT based multiplication.			


	for(int i = 1; i < k; ++i)
	{
		if(i <= plainMulLimit)
		{
			polyOnLayerNext = polyOnLayerCurrent/2; // at the immediate upper level poly number will be reduced by half.
			polyLengthNext = 2*polyLengthCurrent -1; // at the immediate upper level poly length will be double minus 1.

			threadsForAmul = 2*polyLengthCurrent; // how many threads are necessary for each plain multiplication.
                        mulInThreadBlock = (int)floor((double)Tmul/(double)threadsForAmul); // how many multiplications that a thread block can do.
                        blockNo = (int)ceil( ((double) polyOnLayerCurrent/(double) mulInThreadBlock)*0.5  );			

			cudaMalloc((void **)&check.Ml[i], sizeof(sfixn)*polyOnLayerNext*polyLengthNext); // allocating space for the immediate upper level of left  subtree
			cudaThreadSynchronize();
			cudaMalloc((void **)&check.Mr[i], sizeof(sfixn)*polyOnLayerNext*polyLengthNext); // allocating space for the immediate upper level of right subtree
			cudaThreadSynchronize();
			listPlainMulGpu<<<blockNo, Tmul>>>(check.Ml[i-1], check.Ml[i], polyLengthCurrent, polyOnLayerCurrent, threadsForAmul, mulInThreadBlock, p);
			cudaThreadSynchronize(); // creating the immediate upper level of left  subtree
			listPlainMulGpu<<<blockNo, Tmul>>>(check.Mr[i-1], check.Mr[i], polyLengthCurrent, polyOnLayerCurrent, threadsForAmul, mulInThreadBlock, p);
			cudaThreadSynchronize(); // creating the immediate upper level of right subtree

			if(i == plainMulLimit)
			{				
				// We need the subinverse tree at level plainMulLimit directly.
				// From this level we will not store the leading coefficient of subproduct tree.
				// So poly in supproduct tree created at this level should also need to be modified. 
				//We will use the pointer to the next level as temporary storage.
				cudaMalloc((void **)&check.Ml[i+1], sizeof(sfixn)*(polyOnLayerNext)*(polyLengthNext-1));
				cudaThreadSynchronize(); 
				cudaMalloc((void **)&check.Mr[i+1], sizeof(sfixn)*(polyOnLayerNext)*(polyLengthNext-1));
				cudaThreadSynchronize(); 

				cudaMalloc((void **)&check.InvMl[i], sizeof(sfixn)*(polyOnLayerNext)*(polyLengthNext-1));
                                cudaThreadSynchronize();// allocating space for subinverse tree of left  subtree for this level
				cudaMalloc((void **)&check.InvMr[i], sizeof(sfixn)*(polyOnLayerNext)*(polyLengthNext-1));
                                cudaThreadSynchronize();// allocating space for subinverse tree of right subtree for this level

				blockNo = (int)ceil((double)(polyOnLayerNext)/(double)(Tinv));
				listPolyinv<<<blockNo, Tinv>>>(check.Ml[i], check.InvMl[i], polyOnLayerNext, p);
				cudaThreadSynchronize();// creating subinverse tree of left  subtree for this level
				listPolyinv<<<blockNo, Tinv>>>(check.Mr[i], check.InvMr[i], polyOnLayerNext, p);
				cudaThreadSynchronize();// creating subinverse tree of right subtree for this level
				
				// Now we will store the poly in subproduct tree at this level according to our new data structure. i.e no leading coefficient

				copyMgpu<<<polyOnLayerNext ,(polyLengthNext -1)>>>(check.Ml[i+1], check.Ml[i], polyLengthNext);
				copyMgpu<<<polyOnLayerNext ,(polyLengthNext -1)>>>(check.Mr[i+1], check.Mr[i], polyLengthNext);
				cudaThreadSynchronize();

				cudaFree(check.Ml[i]); cudaFree(check.Mr[i]); // free the temporary spaces				
				check.Ml[i] = check.Ml[i+1]; check.Mr[i] = check.Mr[i+1]; // adjust the pointer	
							
			}			
			polyLengthCurrent = polyLengthNext;
			polyOnLayerCurrent = polyOnLayerNext;

		}
		else
		{

			l = 1L << (i-1); // length of each poly at (i-1)th level.

			Cl = check.Ml[i-1]; // Cl points the poly of level (i-1)th of left  subtree
			Cr = check.Mr[i-1]; // Cr points the poly of level (i-1)th of right subtree

			cudaMemcpy(Al,   Cl,     sizeof(sfixn)*L,     cudaMemcpyDeviceToDevice); // Al contains the poly of level (i-1)th of left  subtree
			cudaMemcpy(Ar,   Cr,     sizeof(sfixn)*L,     cudaMemcpyDeviceToDevice); // Ar contains the poly of level (i-1)th of right subtree
			cudaMemcpy(Bl, &(Cl[l]), sizeof(sfixn)*(L-l), cudaMemcpyDeviceToDevice); // Bl contains the poly of level (i-1)th of left  subtree excluding the first poly	
			cudaMemcpy(Br, &(Cr[l]), sizeof(sfixn)*(L-l), cudaMemcpyDeviceToDevice); // Bl contains the poly of level (i-1)th of right subtree excluding the first poly		
			// Every alternate poly is making 0 poly. By this the poly need to be multiplied are placed in the same indices. This is done for FFT. Now length of the poly 
			// becomes double. 
			zeroInbetween<<<(int)ceil((double)L/(double)(Tmax*2)), Tmax>>>(Al, Bl, L/2, l ); 
			zeroInbetween<<<(int)ceil((double)L/(double)(Tmax*2)), Tmax>>>(Ar, Br, L/2, l );
			// Allocating space for the poly in the immediate upper level
			cudaMalloc((void **)&check.Ml[i], sizeof(sfixn)*L);
			cudaMalloc((void **)&check.Mr[i], sizeof(sfixn)*L);
			// initializing check.Ml[i] and check.Mr[i] as zero array
			allZero<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>(check.Ml[i], L);	
			allZero<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>(check.Mr[i], L);
			// As we are not storing the leading coefficients, we need to add the coefficients with it.
			pointAdd2<<<(int)ceil((double)(L/2)/(double)Tmax) , Tmax>>>(check.Ml[i] , check.Ml[i-1], l, L/2, p);	
			pointAdd2<<<(int)ceil((double)(L/2)/(double)Tmax) , Tmax>>>(check.Mr[i] , check.Mr[i-1], l, L/2, p);	

			w = primitive_root(i, p); // primitive root of unity of 2^i
			l = 1L << (k-i -1); // no of poly in Al, Bl, Ar, Br
			// computing FFT
			list_stockham_dev(Al, l, i, w, p);
			list_stockham_dev(Bl, l, i, w, p);
			list_stockham_dev(Ar, l, i, w, p);
			list_stockham_dev(Br, l, i, w, p);
			//pointwise multiplication
			pointMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( Al, Bl, L, p);
			pointMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( Ar, Br, L, p);
			// inverse FFT
			winv = inv_mod(w, p);
			list_stockham_dev(Al, l, i, winv, p);
			list_stockham_dev(Ar, l, i, winv, p);

			w = (1L << i);
			ninv = inv_mod(w, p);			
			scalarMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( Al, ninv, L, p);
			scalarMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( Ar, ninv, L, p);
		
			pointAdd<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( check.Ml[i], Al, L, p);	
			pointAdd<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( check.Mr[i], Ar, L, p);						
		
		}
	}
	cudaFree(Al);
	cudaFree(Bl);	
	cudaFree(Ar);
	cudaFree(Br);	
}


void subInvTree(int k, sfixn p)
{
	sfixn *Al, *Bl, *Cl, *Dl, *El, *Fl, *Gl;
	sfixn *Ar, *Br, *Cr, *Dr, *Er, *Fr, *Gr; 
	int L = 1L << (k-1);     // the number of coefficients representing the polys of any subtree of subinverse tree of one level. 
	int l = sizeof(sfixn)*L; // the length of the array storing the polys of any subtree of subinverse tree of one level. 
	int j;
	cudaMalloc((void **)&Al, l);	cudaMalloc((void **)&Bl, l);	cudaMalloc((void **)&Cl, l);
	cudaMalloc((void **)&Ar, l);	cudaMalloc((void **)&Br, l);	cudaMalloc((void **)&Cr, l);
	l = l*2;
	cudaMalloc((void **)&Dl, l);	cudaMalloc((void **)&El, l);	cudaMalloc((void **)&Fl, l);	cudaMalloc((void **)&Gl, l);
	cudaMalloc((void **)&Dr, l);	cudaMalloc((void **)&Er, l);	cudaMalloc((void **)&Fr, l);	cudaMalloc((void **)&Gr, l);

	
	int blockNo1 = (int)ceil((double)(L)/(double)(Tmax));
	int blockNo2 = (int)ceil((double)(L*2)/(double)(Tmax));
	sfixn w, winv, ninv;
	for(int i =  plainMulLimit; i < (k-1); ++i)
	{
		l =  1L << (i); // the length of each poly of subinverse tree at level i
		j = 1L << (k - i - 1); // the number of polys of any subtree of subinverse tree at level i
		/*
		rev(M)*inv_i(M) = 1 mod x^{2^i}
		We need inv_i(M) such that rev(M)*inv_i(M) = 1 mod x^{2^{i+1}}
		We need to compute rev(M)*inv_i(M) mod x^{2^{i+1}}
		We have, M' = M without leading term.
		I = rev(inv_i(M)) without leading term.
		M' and inv_i(M) has same degree.
		M*rev(inv_i(M)) = rev(conv(M', rev(inv_i(M))) + I) trailing by (2^i - 1) zeros then 1
		*/		
		//rev(inv_i(M))
		listReversePoly<<<blockNo1,  Tmax>>>(Al, check.InvMl[i], l, j); // reversing polys in level i of left  subtree of subinverse tree
		listReversePoly<<<blockNo1,  Tmax>>>(Ar, check.InvMr[i], l, j); // reversing polys in level i of right subtree of subinverse tree
		//I
		listCpLdZeroPoly<<<blockNo1, Tmax>>>(Bl, Al ,l , j); // copying Al to Bl except the leading coefficient which is set 0 in Bl
		listCpLdZeroPoly<<<blockNo1, Tmax>>>(Br, Ar ,l , j); // copying Ar to Br except the leading coefficient which is set 0 in Br
		//M'
		cudaMemcpy(Cl, check.Ml[i], sizeof(sfixn)*L, cudaMemcpyDeviceToDevice); // copying the left  subtree of subproduct tree into Cl
		cudaMemcpy(Cr, check.Mr[i], sizeof(sfixn)*L, cudaMemcpyDeviceToDevice); // copying the right subtree of subproduct tree into Cl
		// conv(M', rev(inv_i(M)))
		w =  primitive_root(i, p);
		list_stockham_dev(Al, j, i, w, p);
		list_stockham_dev(Ar, j, i, w, p);
		list_stockham_dev(Cl, j, i, w, p);
		list_stockham_dev(Cr, j, i, w, p);

		pointMul<<<blockNo1 , Tmax>>>( Al, Cl, L, p);
		pointMul<<<blockNo1 , Tmax>>>( Ar, Cr, L, p);
		winv = inv_mod(w, p);	
		list_stockham_dev(Al, j, i, winv, p);
		list_stockham_dev(Ar, j, i, winv, p);

		ninv = inv_mod(l, p);
		scalarMul<<<blockNo1, Tmax>>>(Al, ninv, L, p);
		scalarMul<<<blockNo1, Tmax>>>(Ar, ninv, L, p);
		// adding I
		pointAdd<<<blockNo1,  Tmax>>> (Al, Bl, L, p);
		pointAdd<<<blockNo1,  Tmax>>> (Ar, Br, L, p);
		//reversing
		listReversePoly<<<blockNo1, Tmax>>>(Bl, Al, l , j);
		listReversePoly<<<blockNo1, Tmax>>>(Br, Ar, l , j);
		//setting negative to all coefficients 		
		allNeg<<<blockNo1, Tmax >>>(Bl, L, p);
		allNeg<<<blockNo1, Tmax >>>(Br, L, p);
		// making inv_i(M) bigger poly by padding zeros
		listPolyDegInc<<<blockNo2, Tmax>>>(check.InvMl[i], Dl, l,   j, 2*l);
		listPolyDegInc<<<blockNo2, Tmax>>>(check.InvMr[i], Dr, l,   j, 2*l);
		// making -(M*rev(inv_i(M))) bigger by padding zeros
		listPolyDegInc<<<blockNo2, Tmax>>>(Bl, El, l,   j, 2*l);
		listPolyDegInc<<<blockNo2, Tmax>>>(Br, Er, l,   j, 2*l);
		cudaMemcpy(Fl,  Dl,  sizeof(sfixn)*L*2,  cudaMemcpyDeviceToDevice);
		cudaMemcpy(Fr,  Dr,  sizeof(sfixn)*L*2,  cudaMemcpyDeviceToDevice);
		// -(M*rev(inv_i(M))) * inv_i(M)
		w =  primitive_root(i+1, p);
		list_stockham_dev(El, j, i+1, w, p);
		list_stockham_dev(Er, j, i+1, w, p);
		list_stockham_dev(Dl, j, i+1, w, p);		
		list_stockham_dev(Dr, j, i+1, w, p);
		pointMul<<<blockNo2, Tmax>>>( El, Dl, 2*L, p);
		pointMul<<<blockNo2, Tmax>>>( Er, Dr, 2*L, p);
		winv = inv_mod(w, p);
		list_stockham_dev(El, j, i+1, winv, p);
		list_stockham_dev(Er, j, i+1, winv, p);
		ninv = inv_mod(l*2, p);
		scalarMul<<<blockNo2, Tmax>>>( El, ninv, L*2, p);
		scalarMul<<<blockNo2, Tmax>>>( Er, ninv, L*2, p);
		// copy the upper part as the lower part is known from inv_i(M) or F
		listCpUpperCuda<<<blockNo1 , Tmax>>>( Fl, El, L, l);
		listCpUpperCuda<<<blockNo1 , Tmax>>>( Fr, Er, L, l);
		//Now F contains inv_i(M) with more precision
		//Multiply pair of inv_i(M) to get inv_{i+1}(M)
		cudaMemcpy(Gl, &(Fl[2*l]), sizeof(sfixn)*(2*L-2*l), cudaMemcpyDeviceToDevice);
		cudaMemcpy(Gr, &(Fr[2*l]), sizeof(sfixn)*(2*L-2*l), cudaMemcpyDeviceToDevice);
		zeroInbetween<<<blockNo1, Tmax>>>(Fl, Gl, L, 2*l );
		zeroInbetween<<<blockNo1, Tmax>>>(Fr, Gr, L, 2*l );		
		j = j/2;
		w =  primitive_root(i+2, p);
		list_stockham_dev(Fl, j, i+2, w,   p);
		list_stockham_dev(Fr, j, i+2, w,   p);
		list_stockham_dev(Gl, j, i+2, w,   p);
		list_stockham_dev(Gr, j, i+2, w,   p);
		pointMul<<<blockNo2 , Tmax>>>( Fl, Gl, L*2, p);
		pointMul<<<blockNo2 , Tmax>>>( Fr, Gr, L*2, p);
		winv = inv_mod(w, p);
		list_stockham_dev(Fl, j, i+2, winv, p);
		list_stockham_dev(Fr, j, i+2, winv, p);
		ninv = inv_mod(l*4, p);
		scalarMul<<<blockNo2 , Tmax>>>( Fl, ninv, 2*L, p);
		scalarMul<<<blockNo2 , Tmax>>>( Fr, ninv, 2*L, p);
		cudaMalloc((void **)&check.InvMl[i+1], sizeof(sfixn)*L);
		cudaMalloc((void **)&check.InvMr[i+1], sizeof(sfixn)*L);
		listCpLowerCuda<<<blockNo1 , Tmax>>>( check.InvMl[i+1], Fl, L, 2*l);
		listCpLowerCuda<<<blockNo1 , Tmax>>>( check.InvMr[i+1], Fr, L, 2*l);
		cudaThreadSynchronize();		
	}
	cudaFree(Al);	cudaFree(Bl);	cudaFree(Cl);	cudaFree(Dl);	cudaFree(El);	cudaFree(Fl);	cudaFree(Gl);
	cudaFree(Ar);	cudaFree(Br);	cudaFree(Cr);	cudaFree(Dr);	cudaFree(Er);	cudaFree(Fr);	cudaFree(Gr);
}


struct PolyEvalSteps fastEvaluation(sfixn *F, sfixn *points, int k, int p, int flag)
{
	//declaring the struct PolyEvalSteps check which holds the intermediate results.
	

	int numPoints = (1 << k);
	sfixn *pointsGPU;
	cudaMalloc((void **)&pointsGPU,   sizeof(sfixn)*numPoints); // allocating device memory for points
	cudaMalloc((void **)&check.Ml[0], sizeof(sfixn)*numPoints); // allocating device memory for leaves of left  subproduct tree
	cudaMalloc((void **)&check.Mr[0], sizeof(sfixn)*numPoints); // allocating device memory for leaves of right subproduct tree
	cudaMemcpy(pointsGPU, points, sizeof(sfixn)*numPoints, cudaMemcpyHostToDevice); //coping the points in device memory
	cudaThreadSynchronize();

	leavesSubproductTree<<<(int)ceil((double)( numPoints/2.0)/(double)Tmax) , Tmax>>>(check.Ml[0], pointsGPU, numPoints/2, 0); // creating leaves of left  subproduct tree
	leavesSubproductTree<<<(int)ceil((double)( numPoints/2.0)/(double)Tmax) , Tmax>>>(check.Mr[0], pointsGPU, numPoints/2, 1); // creating leaves of right subproduct tree
	cudaThreadSynchronize();
	
	subProductTree( k, p); // creating  subproduct tree
	subInvTree(k, p); // creating  subinverse tree

	cudaThreadSynchronize();


	return check;
}








