#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <time.h>

using namespace std;

const int N = 32;
const int thread = 32;
const int block = N/thread;
#define size (32*32)

__global__ void matMult(int *A, int *B, int *C){
	__shared__ int a[size];
	__shared__ int b[size];
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	int temp = 0;
	for(int i = 0; i<block; i++){
		a[blockDim.y * threadIdx.y + threadIdx.x] = A[(row * N) + (i * blockDim.x) + threadIdx.x];
		b[blockDim.y * threadIdx.y + threadIdx.x] = B[(i * blockDim.y * N) + (N * threadIdx.y) + col];
		__syncthreads();
		for(int j = 0; j<blockDim.x ;j++){
			temp+=a[blockDim.y * threadIdx.y + j]*b[threadIdx.x + blockDim.y*j];
		}
		__syncthreads();
	}
	C[row * N + col] = temp;
}

void matrixInit(int *A){
	srand(time(NULL));
	for(int i = 0; i< N*N; i++){
		A[i] = rand()%3 -1;
	}
}

void directMultCheck(int *A, int *B, int *C){
	int temp;
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			temp = 0;
			for(int k = 0; k<N; k++){
				temp+=A[i*N + k]*B[k*N + j];
			}
			assert(temp == C[i*N + j]);
		}
	}
}

void printer(int *A, int *B, int *C){
	for(int i=0; i<N*N; i++){
		printf("%d\t", A[i]);
		if((i+1)%N == 0){
			printf("\n");
		}
	}
	printf("\n\n");
	for(int i=0; i<N*N; i++){
		printf("%d\t", B[i]);
		if((i+1)%N == 0){
			printf("\n");
		}
	}
	printf("\n\n");
	for(int i=0; i<N*N; i++){
		printf("%d\t", C[i]);
		if((i+1)%N == 0){
			printf("\n");
		}
	}
}

int main(){
	clock_t start, end;
	double time_taken_m, time_taken_d;
	int *A, *B, *C;
	
	cudaMallocManaged(&A, N * N * sizeof(int));
	cudaMallocManaged(&B, N * N * sizeof(int));
	cudaMallocManaged(&C, N * N * sizeof(int));
	
	matrixInit(A);
	matrixInit(B);
	
	dim3 THREAD(thread, thread);
	dim3 BLOCK(block, block);
	

	start = clock();
	matMult<<<BLOCK, THREAD>>>(A, B, C);
	cudaDeviceSynchronize();
	end = clock();
	
	time_taken_m = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("\nTotal time taken: %f\n", time_taken_m);
	//printer(A, B, C);
	
	/*
	start = clock();
	directMultCheck(A, B, C);
	cout << "\nCORRECT CALCULATION" << endl;
	end = clock();
	
	time_taken_d = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("\nTotal time taken: %f\n", time_taken_d);
	*/
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	
	return 0;
}