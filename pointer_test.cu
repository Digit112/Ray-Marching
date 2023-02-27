#include <stdio.h>

#include "utility/vec.hpp"

using foo_t = int (*) ();

__device__ int ret_1() {
	return 1;
}

__device__ foo_t p_ret_1 = ret_1;

__global__ void call(foo_t foo) {
	printf("%d\n", foo());
}

int main() {
	printf("Hello Ray Marching!\n");
	cudaError_t err;
	
	foo_t ret_1_k = ret_1;
    err = cudaMemcpyFromSymbol(&ret_1_k, p_ret_1, sizeof(foo_t));
	printf("%s\n", cudaGetErrorName(err));
	
	call<<<1, 1>>>(ret_1_k);
	
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();
	printf("%s\n", cudaGetErrorName(err));
	
	return 0;
}