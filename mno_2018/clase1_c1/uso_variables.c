#include<stdio.h>
/*Definición y declaración  e inicialización de variables*/
main(){
	char variable_char; //8 bits, entera
	int variable_int; //32 bits, entera
	float variable_float = -2.0; //32 bits, punto flotante. Esta línea representa 
								//definición e inicialización de variable_float.
	double variable_double; //64 bits, punto flotante
	variable_char = 'b';
	variable_int = 3;
	variable_double = 5.0;

	//Imprimir los valores
	printf("Valor de variable char: %c", variable_char);
}
