#include<stdio.h>
__global__ void func(void){
	printf("Hello world del bloque %d del thread %d!\n", blockIdx.x, threadIdx.x);
}
int main(void){
	func<<<3,3>>>(); //3 bloques de 3 threads cada uno
	cudaDeviceSynchronize();
	printf("Hola del cpu thread\n");
	return 0;
}

