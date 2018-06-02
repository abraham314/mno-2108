#include<stdio.h>
main(){
	int i,j;
	i=3;
	printf("Valor de i inicial :%d\n",i);
	i++; //i = i+1;
	printf("Valor de i con i++: %d\n", i);
	i=3;
	//++i; //i = i+1;
	printf("Valor de i con i++: %d\n", ++i);
	i=12;
	printf("Valor de i antes de dar valor a j: %d\n", i);
	j=i++ + 5;
	printf("Valor de i después de dar valor a j con i++: %d\n", i);
	printf("Valor de j con i++: %d\n", j);
	i=12;
	printf("Valor de i antes de dar valor a j: %d\n", i);
	j=++i + 5;
	printf("Valor de i después de dar valor a j con ++i: %d\n", i);
	printf("Valor de j con i++: %d\n", j);
	i=0;
	printf("Valor de i antes de dar valor a j: %d\n", i);
	j = i-- + 5;
	printf("Valor de i con i--: %d\n", i);
	printf("Valor de j = i--+5: %d\n", j);
	i=0;
	printf("Valor de i antes de dar valor a j: %d\n", i);
	j = --i + 5;
	printf("Valor de i con --i: %d\n", i);
	printf("Valor de j = --i+5: %d\n", j);
}
