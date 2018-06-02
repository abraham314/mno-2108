#include<stdio.h>
main(){
	int variable_int;
	double variable_double = -5291.485;
	printf("variable double=%0.5f\n", variable_double);
	variable_int = variable_double;
	printf("variable_int = %d\n",variable_int);
	printf("variable_double+variable_int=%f\n",variable_double+variable_int);
}
