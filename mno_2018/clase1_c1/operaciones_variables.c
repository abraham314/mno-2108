#include<stdio.h>
main(){
	int variable_int1, variable_int2;
	double variable_double1, variable_double2, variable_double3;
	//Inicialización de variables:
	variable_int1 = 3;
	variable_int2 = -1;
	variable_double1 = 5.0;
	variable_double2 = -3.0;
	variable_double3 = 0.6789281736281947;
	variable_int1 = variable_int1/variable_int2;
	printf("Variable entera divida por -1: %d\n", variable_int1);
	variable_double1 = variable_double1/variable_double2;
	printf("Variable double1 entre variable double2: %1.9f\n", variable_double1);
	//Notación exponencial
	printf("Variable double 1 entre variable double 2 notación exponencial: %1.9e\n", variable_double1);

}
