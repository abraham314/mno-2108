#include<stdio.h>
main(){
		int variable;
		variable=10;
		printf("Variable antes de while: %d\n", variable);
		while(variable-1){
			printf("valor de variable = %d\n", variable);
			--variable;
		}
		printf("Fin de while\n");
		printf("Variable después del while: %d\n",variable);
	}
