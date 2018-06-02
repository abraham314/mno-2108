#include<stdio.h>
main(){

	int *apuntador;
	int variable1 = -10;
	int variable2 = 5;
	apuntador = &variable1;
	printf("apuntador[0]: %d\n", apuntador[0]);
	apuntador = apuntador + 2;
	apuntador = &variable2;
	apuntador = apuntador - 2;
	printf("apuntador[2] : %d\n", apuntador[2]);

}
