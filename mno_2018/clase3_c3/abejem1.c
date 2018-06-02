#include<stdio.h>
main(){
	int *apuntador;
	int	variable = -5;
	apuntador = &variable;
	printf("Valor de variable: %d\n", variable);
	printf("Valor de apuntador[0]: %d\n",apuntador[0]);
}
