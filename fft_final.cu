#include <stdio.h>
#include <cstdlib>
#include <cuComplex.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>

#define M_PI acos(-1.0)

__global__ void fft(cuDoubleComplex *A, int n, int i, double power){
	int block = (blockDim.x)*(blockIdx.x);
	//printf("%d\n", block);
	int real_j = (n*i) + (((1 << (i+1))*(threadIdx.x + block)));
	int pow2i = 1 << i;
	for(int p=0; p<pow2i*2; p++){
		cuDoubleComplex ad;
		if(p<pow2i){
			ad = cuCadd(A[real_j+p], cuCmul(make_cuDoubleComplex(cos(power*(double)(p)), -1*sin(power*(double)(p))), A[real_j + pow2i + p]));
		}
		else{
			ad = cuCsub(A[real_j+p%pow2i], cuCmul(make_cuDoubleComplex(cos(power*(double)(p%pow2i)), -1*sin(power*(double)(p%pow2i))), A[real_j + pow2i + p%pow2i]));
		}

		A[((i+1)*n) + (pow2i*2)*(threadIdx.x + block) + p] = make_cuDoubleComplex(ad.x, ad.y);
	}
	__syncthreads();
}

int logg(int n, int b){
	return log(n)/log(b);
}

int arrInit(cuDoubleComplex *A, int n){
	srand(time(NULL));
	for(int i = 0; i<n; i++){
		double cr = (double)(rand() % 3) - 1.0;
		double ci = (double)(rand() % 3) - 1.0;
		A[i] = make_cuDoubleComplex(cr, ci);
	}
	return 0;
}

__global__ void stateInit(curandState *state, unsigned long seed){
	int i = 1024*blockIdx.x + threadIdx.x;
	curand_init(seed, i, 0, &state[i]);
}

__global__ void arrInitMP(cuDoubleComplex *A, curandState *state){
	int i = 1024*blockIdx.x + threadIdx.x;
	float num[2];
	curandState localState = state[i];

    num[0] = curand_uniform_double(&localState);
	num[1] = curand_uniform_double(&localState);
	cuDoubleComplex c = make_cuDoubleComplex((float)(((int)(num[0]*10000))%10 - 5), (float)(((int)(num[1]*10000))%10 - 5));
	
	state[i] = localState;
	A[1024*blockIdx.x + threadIdx.x] = c;
	//printf("start 2");
}

int customArr(cuDoubleComplex *A, int n){
	//double temp[16] = {5.0, 7.0, 5.0, 2.0, 4.0, 6.0, 9.0, 9.0, 2.0, 2.0, 1.0, 4.0, 1.0, 3.0, 6.0, 9.0};
	double temp[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	for(int i =0; i<n; i++){
		double cr = (double)(i+1);
		cr = temp[i];
		double ci = 0.0;
		A[i] = make_cuDoubleComplex(cr, ci);
		//printf("%f %f\n", cuCreal(A[i]), cuCimag(A[i]));
	}
	return 0;
}

void printer(cuDoubleComplex *A, int n){
	for(int i=0; i<n; i++){
		printf("%f %f\n", cuCreal(A[i]), cuCimag(A[i]));
	}
}

void printerFinal(cuDoubleComplex *A, int size, int n){
	for(int i=size-n; i<size; i++){
		printf("%f %f\n", cuCreal(A[i]), cuCimag(A[i]));
	}
}

int bitSwitch(int x,int logn){
	int reverse_num = 0;
	for(int i = 0; i<logn; i++){
		reverse_num <<=1;
		reverse_num |= (x&1);
		x >>= 1;
	}
	return reverse_num;
}

__global__ void bitSwitchMP(cuDoubleComplex *B, cuDoubleComplex *A, int logn){
	int xt = blockIdx.x;
	int reverse_num = 0;
	for(int i = 0; i<logn; i++){
		reverse_num <<=1;
		reverse_num |= (xt&1);
		xt >>= 1;
	}
	A[blockIdx.x] = make_cuDoubleComplex(B[reverse_num].x, B[reverse_num].y);
}

int main(){
	clock_t start, end;
	double time_taken_m;
	int N = 1 << 3;
	
	int size = N*logg(N, 2) + N;
	
	curandState *allStates;
	cuDoubleComplex *B, *A;
	
	cudaMallocManaged(&allStates, N*sizeof(curandState));
	cudaMallocManaged(&B, size*sizeof(cuDoubleComplex));
	cudaMallocManaged(&A, size*sizeof(cuDoubleComplex));
	
	
	int bl = N/(1 << 10);
	if(bl == 0){
		bl = 1;
	}
	dim3 TH(1024, 1);
	dim3 BL(bl, 1);
	
	stateInit<<<BL, TH>>>(allStates, time(NULL));
	cudaDeviceSynchronize();
	
	arrInitMP<<<BL, TH>>>(B, allStates);
	cudaDeviceSynchronize();
	
	//arrInit(B, N);
	//customArr(B, N);
	
	//printer(B, N);
	
	bitSwitchMP<<<N, 1>>>(B, A, logg(N, 2));
	cudaDeviceSynchronize();
	
	start = clock();
	
	for(int i=0; i<logg(N, 2); i++){
		double power = 2.0*M_PI/(1 << (i+1));
		
		int thread = (int)((double)(N)/(1 << (i+1)));
		int block = (thread/(1 << 10));
		if(block == 0){
			block = 1;
		}
		if(block > 1){
			thread = 1 << 10;
		}
		
		//printf("block = %d, thread = %d\n", block, thread);
		
		dim3 BLOCK(block, 1);
		dim3 THREAD(thread, 1);
		
		fft<<<BLOCK, THREAD>>>(A, N, i, power);
		cudaDeviceSynchronize();
		
		
		//printer(A, size);
	}
	end = clock();
	
	//printf("\n\n\n");
	
	time_taken_m = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("\nTotal time taken: %f\n", time_taken_m);
	
	//printerFinal(A, size, N);
	
	cudaFree(A);
	cudaFree(B);
	
}