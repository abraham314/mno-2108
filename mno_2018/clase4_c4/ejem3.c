#include<stdio.h>
main(){
	char *arreglo_str[4];
	arreglo_str[0] = "Hola,\n";
	arreglo_str[1] = "este es un\n";
	arreglo_str[2] = "ejemplo\t";
	arreglo_str[3] = "de un arreglo de apuntadores\n";
	printf("arreglo_str[0] : %s\n", arreglo_str[0]);
	printf("*arreglo_str : %s\n", *arreglo_str);
	printf("**arreglo_str: %c\n", **arreglo_str);
}
