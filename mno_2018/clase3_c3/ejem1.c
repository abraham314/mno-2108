#include<stdio.h>
main(){
	int variable;
	int *p;
	p = &variable; //inicializamos al apuntador p
	printf("Address de variable int: %p\n", &variable);
	printf("Address de apuntador: %p\n", p);
}
