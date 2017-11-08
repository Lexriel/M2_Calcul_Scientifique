/*
Author: Sardar Haque Email: haque.sardar@gmail.com

*/


#include "../include/fastPolyEvaluation.h"

using namespace std;





void fastEvaluation(int k, sfixn p, sfixn *M1, sfixn *M2, sfixn *Fpoly, int check)
{
	int i, j;

	/* creating subproduct tree*/
	
	sfixn *Mgpu[MAX_LEVEL], *A, *B, *C, *D, *E, *F, *G;
	sfixn *MinvGpu[MAX_LEVEL];	
	
	int polyLengthCurrent = 2;
	int polyOnLayerCurrent = 1L << (k-2);
	int polyLengthNext, polyOnLayerNext;
	
	cudaMalloc((void **)&Mgpu[0], sizeof(sfixn)*polyLengthCurrent*polyOnLayerCurrent);
	cudaThreadSynchronize();

	cudaMemcpy(Mgpu[0], M1, sizeof(sfixn)*polyLengthCurrent*polyOnLayerCurrent, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();

	int threadsForAmul, mulInThreadBlock, blockNo;
	int L = 1L << (k-2);
	int l;
	sfixn w, winv, ninv;
	
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(i = 1; i < (k-1); ++i)
	{
		if(i <= plainMulLimit)
		{	

			//
			//cout<<i<<" "<<polyLengthCurrent<<" "<<polyOnLayerCurrent<<" "<<polyOnLayerCurrent*polyLengthCurrent;
			//		
			polyOnLayerNext = polyOnLayerCurrent/2;
			polyLengthNext = 2*polyLengthCurrent -1;

			 //
                        //cout<<" "<<polyLengthNext<<" "<<polyOnLayerNext<<" "<<polyOnLayerNext*polyLengthNext<<endl;
                        //


			threadsForAmul = 2*polyLengthCurrent;
                        mulInThreadBlock = (int)floor((double)Tmul/(double)threadsForAmul);
                        blockNo = (int)ceil( ((double) polyOnLayerCurrent/(double) mulInThreadBlock)*0.5  );

					
			cudaMalloc((void **)&Mgpu[i], sizeof(sfixn)*polyOnLayerNext*polyLengthNext);
			cudaThreadSynchronize();
			listPlainMulGpu<<<blockNo, Tmul>>>(Mgpu[i-1], Mgpu[i], polyLengthCurrent, polyOnLayerCurrent, threadsForAmul, mulInThreadBlock, p);
			cudaThreadSynchronize();
					
			if(i == plainMulLimit)
			{
				cudaMalloc((void **)&Mgpu[i+1], sizeof(sfixn)*(polyOnLayerNext)*(polyLengthNext-1));
				cudaThreadSynchronize();
				cudaMalloc((void **)&MinvGpu[i], sizeof(sfixn)*(polyOnLayerNext)*(polyLengthNext-1));
                                cudaThreadSynchronize();

				blockNo = (int)ceil((double)(polyOnLayerNext)/(double)(Tinv));
				listPolyinv<<<blockNo, Tinv>>>(Mgpu[i], MinvGpu[i], polyOnLayerNext, p);


				copyMgpu<<<polyOnLayerNext ,(polyLengthNext -1)>>>(Mgpu[i+1], Mgpu[i], polyLengthNext);
				cudaThreadSynchronize();
				cudaFree(Mgpu[i]);
				Mgpu[i] = Mgpu[i+1];				
			}			
			polyLengthCurrent = polyLengthNext;
			polyOnLayerCurrent = polyOnLayerNext;				
		}		
		else
		{
			l = 1L << (i-1);
			cudaMalloc((void **)&A,    sizeof(sfixn)*L);
			cudaMalloc((void **)&B,    sizeof(sfixn)*L);			
			C = Mgpu[i-1];
			cudaMemcpy(A,   C,     sizeof(sfixn)*L,     cudaMemcpyDeviceToDevice);	
			cudaMemcpy(B, &(C[l]), sizeof(sfixn)*(L-l), cudaMemcpyDeviceToDevice);	
			zeroInbetween<<<(int)ceil((double)L/(double)(Tmax*2)), Tmax>>>(A, B, L/2, l );

			//
			//cout<<i<<" "<<l<<" "<<L<<endl;
		 	//printGPUArray(C, L);
 			//printGPUArray(A, L);
			//printGPUArray(B, L);
			//
			cudaMalloc((void **)&Mgpu[i], sizeof(sfixn)*L);
			allZero<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>(Mgpu[i], L);	

			//
			 //printGPUArray(Mgpu[i], L);
			//
			pointAdd2<<<(int)ceil((double)(L/2)/(double)Tmax) , Tmax>>>(Mgpu[i] , Mgpu[i-1], l, L/2, p);	
			//
                         //printGPUArray(Mgpu[i], L);
                        //

			
			w = primitive_root(i, p);
			l = 1L << (k-i -2);
			//
			//cout<<i<<" "<<w<<" "<<l<<endl;
			//
			list_stockham_dev(A, l, i, w, p);
			list_stockham_dev(B, l, i, w, p);
			pointMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( A, B, L, p);

			winv = inv_mod(w, p);
			list_stockham_dev(A, l, i, winv, p);

			w = (1L << i);
			ninv = inv_mod(w, p);			
			scalarMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( A, ninv, L, p);	
			pointAdd<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( Mgpu[i], A, L, p);	
		

			
			cudaFree(A);
			cudaFree(B);		
		}
	}


	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float outerTime;
        cudaEventElapsedTime(&outerTime, start, stop);
	cout<<outerTime/1000.0<<" seconds for subproductree and the subInversetree of plain arithmatic level."<<endl;

	
	/* creating subinverse tree*/
	

	cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);	
	
	 L = 1L << (k-2); 
	//for(i =  plainMulLimit; i < (k-2); ++i)
	for(i =  plainMulLimit; i < (k-2); ++i)
	{
		
		
	
		
		l =  1L << (i);
		j = 1L << (k-2-i);
	        cudaMalloc((void **)&A, sizeof(sfixn)*L);
	       	blockNo = (int)ceil((double)(L)/(double)(Tmax));
	        listReversePoly<<<blockNo, Tmax>>>(A, MinvGpu[i], l , j);
		
		cudaMalloc((void **)&B, sizeof(sfixn)*L);
		listCpLdZeroPoly<<<blockNo, Tmax>>>(B, A ,l , j);
		
		cudaMalloc((void **)&C, sizeof(sfixn)*L);
		cudaMemcpy(C,   Mgpu[i],     sizeof(sfixn)*L,     cudaMemcpyDeviceToDevice);

		//
                //cout<<"creating subInverse tree at: "<<i+1<<" the number of poly is: "<<j<<endl;
		//cout<<"Reverse of Minv["<<i<<"]: "<<endl;
                //printGPUArray(A, L);
		//cout<<"Reverse of Minv["<<i<<"] excluding leading coefficient: "<<endl;
		//printGPUArray(B, L);
		//cout<<"M["<<i<<"]: "<<endl;
		//printGPUArray(C, L);
                //

		w =  primitive_root(i, p);
		//
		//cout<<"2^i th root of unity is: "<<w<<endl;
		//
		list_stockham_dev(A, j, i, w, p);
		list_stockham_dev(C, j, i, w, p);
		pointMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( A, C, L, p);
	        winv = inv_mod(w, p);	
		list_stockham_dev(A, j, i, winv, p);
                ninv = inv_mod(l, p);
	        scalarMul<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( A, ninv, L, p);
			
		pointAdd<<<(int)ceil((double)L/(double)Tmax) , Tmax>>>( A, B, L, p);
		

 		blockNo = (int)ceil((double)(L)/(double)(Tmax));
                listReversePoly<<<blockNo, Tmax>>>(B, A, l , j);		
		allNeg<<<blockNo, Tmax >>>(B, L, p);
		//
		//cout<<"T:"<<endl;
		//printGPUArray(B, L);
		//
		
		cudaMalloc((void **)&D, sizeof(sfixn)*L*2);
		cudaMalloc((void **)&E, sizeof(sfixn)*L*2);
		cudaMalloc((void **)&F, sizeof(sfixn)*L*2);
		cudaMalloc((void **)&G, sizeof(sfixn)*L*2);

		
		blockNo = (int)ceil((double)(L*2)/(double)(Tmax));
	        listPolyDegInc<<<blockNo, Tmax>>>(MinvGpu[i], D, l,   j, 2*l);
		listPolyDegInc<<<blockNo, Tmax>>>(B, E, l,   j, 2*l);
		cudaMemcpy(F,  D,  sizeof(sfixn)*L*2,  cudaMemcpyDeviceToDevice);	
		
		
	
		//
		//cout<<"Minv["<<i<<"] padded by zeros:"<<endl;
		//printGPUArray(D, 2*L);
		//cout<<"T padded by zeros"<<endl;
		//printGPUArray(E, 2*L);	
		//cout<<"Minv["<<i<<"] padded by zeros"<<endl;
                //printGPUArray(F, 2*L);

		//
		w =  primitive_root(i+1, p);
		list_stockham_dev(E, j, i+1, w, p);
                list_stockham_dev(D, j, i+1, w, p);
                pointMul<<<(int)ceil((double)(L*2)/(double)Tmax) , Tmax>>>( E, D, 2*L, p);
                winv = inv_mod(w, p);
                list_stockham_dev(E, j, i+1, winv, p);
                ninv = inv_mod(l*2, p);
                scalarMul<<<(int)ceil((double)(L*2)/(double)Tmax) , Tmax>>>( E, ninv, L*2, p);


		listCpUpperCuda<<<(int)ceil((double)(L)/(double)Tmax) , Tmax>>>( F, E, L, l);
		//
		//cout<<"Minv["<<i<<"] with one step Newton iteration:"<<endl;
		//printGPUArray(F, 2*L);
		//	

		cudaMemcpy(G, &(F[2*l]), sizeof(sfixn)*(2*L-2*l), cudaMemcpyDeviceToDevice);
		zeroInbetween<<<(int)ceil((double)L/(double)(Tmax)), Tmax>>>(F, G, L, 2*l );	

		j = j/2;
		w =  primitive_root(i+2, p);
		list_stockham_dev(F, j, i+2, w,   p);
		list_stockham_dev(G, j, i+2, w,   p);
		pointMul<<<(int)ceil((double)(L*2)/(double)Tmax) , Tmax>>>( F, G, L*2, p);
		winv = inv_mod(w, p);
		list_stockham_dev(F, j, i+2, winv, p);
		ninv = inv_mod(l*4, p);
		scalarMul<<<(int)ceil((double)(L*2)/(double)Tmax) , Tmax>>>( F, ninv, 2*L, p);
		
		//
		//printGPUArray(F, 2*L);
		//

		
		cudaMalloc((void **)&MinvGpu[i+1], sizeof(sfixn)*L);
	
		listCpLowerCuda<<<(int)ceil((double)(L)/(double)Tmax) , Tmax>>>( MinvGpu[i+1], F, L, 2*l);
		cudaThreadSynchronize();
		

 		//
                //printGPUArray(MinvGpu[i+1], L);
                //


		cudaFree(A);
		cudaFree(B);
		cudaFree(C);
		cudaFree(D);
		cudaFree(E);
		cudaFree(F);
		cudaFree(G);

	}	

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&outerTime, start, stop);
        cout<<outerTime/1000.0<<" seconds for the subInversetree of FFT level."<<endl;

	/* fast evaluation */

	sfixn *FpolyGpu[MAX_LEVEL];
	cudaMalloc((void **)&FpolyGpu[k-1], sizeof(sfixn)*(1L << (k-1)));
	cudaMemcpy(FpolyGpu[k-1], Fpoly, sizeof(sfixn)*(1L << (k-1)), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&FpolyGpu[k-2], sizeof(sfixn)*(1L << (k-2)));

	int lm, linv, lf;
	sfixn *H, *I, *J;


	//for(i = k-2; i >= 0; --i)
	//{
		i = k-2;
		j = 1L << (k-2-i);			
		lm = 1L << (i);
		linv = lm;
		lf = 1L << (i+1);
		//
                printGPUArray(FpolyGpu[k-1], lf);
                //
		cudaMalloc((void **)&H, sizeof(sfixn)*(lf));
		blockNo = (int)ceil((double)(lf*j)/(double)(Tmax));
		listReversePoly<<<blockNo, Tmax>>>(H, FpolyGpu[k-1], lf , j);
		//
                printGPUArray(H, lf);
                //

		cudaMalloc((void **)&I, sizeof(sfixn)*(lf));
 		cudaMemcpy(I, &(H[lm]), sizeof(sfixn)*(lm), cudaMemcpyDeviceToDevice);
		zeroInbetween<<<(int)ceil((double)lf/(double)(Tmax*2)), Tmax>>>(H, I, lf/2, lm );

		cudaMalloc((void **)&J, sizeof(sfixn)*(lf));

		blockNo = (int)ceil((double)(lf *j)/(double)(Tmax));
                listPolyDegInc<<<blockNo, Tmax>>>(MinvGpu[i], J, linv,   j, lf);


		//
                printGPUArray(H, lf);
		printGPUArray(I, lf);
		printGPUArray(J, lf); 
		//


 		w =  primitive_root(i+1, p);
                list_stockham_dev(H, j, i+1, w, p);
                list_stockham_dev(I, j, i+1, w, p);
 		list_stockham_dev(J, j, i+1, w, p);


                pointMul<<<(int)ceil((double)(lf)/(double)Tmax) , Tmax>>>( H, J, lf, p);
		pointMul<<<(int)ceil((double)(lf)/(double)Tmax) , Tmax>>>( I, J, lf, p);
                winv = inv_mod(w, p);
                list_stockham_dev(H, j, i+1, winv, p);
		list_stockham_dev(I, j, i+1, winv, p);
                ninv = inv_mod(lf, p);
                scalarMul<<<(int)ceil((double)(lf)/(double)Tmax) , Tmax>>>( H, ninv, lf, p);
	        scalarMul<<<(int)ceil((double)(lf)/(double)Tmax) , Tmax>>>( I, ninv, lf, p);

		//
                printGPUArray(H, lf);
		 printGPUArray(I, lf);
                //


	
	        list2wayCp<<<(int)ceil((double)(linv )/(double)Tmax) , Tmax>>>( H, I, linv, linv, p);

		//
		printGPUArray(H, lf);
		//
		
		cudaFree(H);
		cudaFree(I);
		cudaFree(J);
	//} 



	struct status trees;


	/* Copy the subproduct tree */
	
		
	
	for(i = 0; i < (k-1) ; ++i)	
	{
		polyLengthCurrent = 1 << (i);
		polyOnLayerCurrent = 1 << (k-i-2);
		if(i < plainMulLimit) ++polyLengthCurrent;

		trees.M[i] = new sfixn [polyLengthCurrent*polyOnLayerCurrent];
	

		cudaMemcpy(trees.M[i] , Mgpu[i], sizeof(sfixn)*polyLengthCurrent*polyOnLayerCurrent, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		cudaFree(Mgpu[i]);
		
		//
		//cout<<"copying subproduct tree: "<<i<<endl;
		//	
	}
	
 	for(i =  plainMulLimit; i < (k-1) ; ++i)
        {
                polyLengthCurrent = 1 << (i);
                polyOnLayerCurrent = 1 << (k-i-2);
                
		//
                //cout<<"copying subinverse tree: "<<i<<endl;
                //

                trees.InvM[i] = new sfixn [polyLengthCurrent*polyOnLayerCurrent];

                cudaMemcpy(trees.InvM[i] ,  MinvGpu[i], sizeof(sfixn)*polyLengthCurrent*polyOnLayerCurrent, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                cudaFree(MinvGpu[i]);
		

        }
	
	return trees;	
}




