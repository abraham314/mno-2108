#include<stdio.h>
#define LIMITE_INFERIOR 0//esto es como variales globales fuera del main
#define LIMITE_SUPERIOR 5
main(){
	int contador;
	double variable1=3485.7, variable2=-4.01;
	printf("variable1 = %4.2f \t variable2 = %1.2f\n", variable1,variable2);
	printf("limite inferior: %d\n",LIMITE_INFERIOR);
	printf("limite superior: %d\n",LIMITE_SUPERIOR);
	printf("Iteracion \t variable1 \t variable2 \t División variable1 entre variable2\n");
	for(contador=LIMITE_INFERIOR; contador < LIMITE_SUPERIOR; contador = contador+1){
		printf("%d \t \t %4.1f \t \t %2.2f \t \t %1.3f\n", contador, variable1,variable2,variable1/variable2);
		variable2/=2;
	}
}
