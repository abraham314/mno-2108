#include<stdio.h>
main(){
	int variable = 5;
	int *p = &variable; //inicializamos al apuntador p al definirlo y declararlo
	printf("Address de variable int: %p\n", &variable);
	printf("Address de apuntador: %p\n", p);
	printf("Accedemos al objeto con *p: %d\n", *p);
}
