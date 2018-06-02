#include<stdio.h>
main(){
	int variable1 = -5;
	int variable2;

	switch(variable1){
		case 0:
			variable2 = -variable1;
			printf("Primer caso se cumplió, variable2=%d\n",variable2);
		case -10:
			variable2 = variable1;
			printf("Segundo caso se cumplió, variable2=%d\n",variable2);
		default:
		variable2 = variable1*10;
		printf("Caso default se cumplió, variable2=%d\n",variable2);

	}
}
