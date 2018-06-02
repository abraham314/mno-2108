#include<stdio.h>
main(){
	int arreglo[3];
	int *apuntador, *apuntador2;
	apuntador = arreglo;
	apuntador2 = arreglo+3;
	printf("apuntador2 - apuntador: %ld\n", apuntador2-apuntador); // El resultado es la "distancia" entre cada uno de los apuntadores
}
