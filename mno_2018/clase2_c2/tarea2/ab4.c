#include<stdio.h>

void main(void){
  
    int c;
    int n;

   while((c=getchar())!=EOF){

        if(c=='\n'){
          
            ++n;
        }
   }


  printf("lineas: %d\n",n);
  printf("\n");


}
