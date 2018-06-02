#include<stdio.h>
main(){
	int variable1=-437, variable2;
	if(variable1<=10)printf("Primer if: variable1 menor o igual a 10\n");
	if(variable1>10){
		printf("Segundo if: variable1 mayor a 10\n");//no es necesario el uso de llaves
	}
	else
		printf("Segundo if: variable1 menor o igual que 10\n");

	printf("Otra forma es con ?:\n");

	variable2=(variable1 <= 10)?variable1:0;
	printf("Usando signo ? para asignar el valor de variable2: %d\n",variable2);

}
