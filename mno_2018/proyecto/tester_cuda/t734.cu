#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusolverRf.h"

#define TEST_PASSED  0
#define TEST_FAILED  1

int main (void){
    /* matrix A */
    int n;
    int nnzA;
    int *Ap=NULL;
    int *Ai=NULL;
    double *Ax=NULL;
    int *d_Ap=NULL;
    int *d_Ai=NULL;
    double *d_rAx=NULL;
    /* matrices L and U */
    int nnzL, nnzU;
    int *Lp=NULL;
    int *Li=NULL;
    double* Lx=NULL;
    int *Up=NULL;
    int *Ui=NULL;
    double* Ux=NULL;
    /* reordering matrices */
    int *P=NULL;
    int *Q=NULL;
    int * d_P=NULL;
    int * d_Q=NULL;
    /* solution and rhs */
    int nrhs; //# of rhs for each system (currently only =1 is supported)
    double *d_X=NULL;
    double *d_T=NULL;
    /* cuda */
    cudaError_t cudaStatus;
    /* cuolverRf */
    cusolverRfHandle_t gH=NULL;
    cusolverStatus_t status;
    /* host sparse direct solver */
    /* ... */
    /* other variables */
    int tnnzL, tnnzU;
    int *tLp=NULL;
    int *tLi=NULL;
    double *tLx=NULL;
    int *tUp=NULL;
    int *tUi=NULL;
    double *tUx=NULL;
    clock_t t1, t2;



    /* ASSUMPTION: recall that we are solving a set of linear systems
       A_{i} x_{i} = f_{i}  for i=0,...,k-1
       where the sparsity pattern of the coefficient matrices A_{i}
       as well as the reordering to minimize fill-in and the pivoting
       used during the LU factorization remain the same. */


    /* Step 1: solve the first linear system (i=0) on the host,
               using host sparse direct solver, which involves
               full LU factorization and solve. */
    /* ... */


    /* Step 2: interface to the library by extracting the following
               information from the first solve:
               a) triangular factors L and U
               b) pivoting and reordering permutations P and Q
               c) also, allocate all the necessary memory */
    /* ... */


    /* Step 3: use the library to solve subsequent (i=1,...,k-1) linear systems
    a) the  library setup (called only once) */
    //create handle
    status = cusolverRfCreate(&gH);
    if (status != CUSOLVER_STATUS_SUCCESS){
        printf ("[cusolverRf status %d]\n",status);
        return TEST_FAILED;
    }

    //set fast mode
    status = cusolverRfSetResetValuesFastMode(gH,CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
    if (status != CUSOLVER_STATUS_SUCCESS){
        printf ("[cusolverRf status %d]\n",status);
        return TEST_FAILED;
    }


    //assemble internal data structures (you should use the coeffcient matrix A
    //corresponding to the second (i=1) linear system in this call)
    t1 = clock();
    status = cusolverRfSetupHost(n, nnzA, Ap, Ai, Ax,
                               nnzL, Lp, Li, Lx, nnzU, Up, Ui, Ux, P, Q, gH);
    cudaStatus = cudaDeviceSynchronize();
    t2 = clock();
    if ((status != CUSOLVER_STATUS_SUCCESS) || (cudaStatus != cudaSuccess)) {
        printf ("[cusolverRf status %d]\n",status);
        return TEST_FAILED;
    }
    printf("cusolverRfSetupHost time = %f (s)\n", (t2-t1)/(float)CLOCKS_PER_SEC);

    //analyze available parallelism
    t1 = clock();
    status = cusolverRfAnalyze(gH);
    cudaStatus = cudaDeviceSynchronize();
    t2 = clock();
    if ((status != CUSOLVER_STATUS_SUCCESS) || (cudaStatus != cudaSuccess)) {
        printf ("[cusolverRf status %d]\n",status);
        return TEST_FAILED;
    }
    printf("cusolverRfAnalyze time = %f (s)\n", (t2-t1)/(float)CLOCKS_PER_SEC);

    /* b) The  library subsequent (i=1,...,k-1) LU re-factorization
          and solve (called multiple times). */
    int k = 2;
    for (int i=1; i<k; i++){
        //LU re-factorization
        t1 = clock();
        status = cusolverRfRefactor(gH);
        cudaStatus = cudaDeviceSynchronize();
        t2 = clock();
        if ((status != CUSOLVER_STATUS_SUCCESS) || (cudaStatus != cudaSuccess)) {
            printf ("[cusolverRF status %d]\n",status);
            return TEST_FAILED;
        }
        printf("cuSolverReRefactor time = %f (s)\n", (t2-t1)/(float)CLOCKS_PER_SEC);

        //forward and backward solve
        t1 = clock();
        status = cusolverRfSolve(gH, d_P, d_Q, nrhs, d_T, n, d_X, n);
        cudaStatus = cudaDeviceSynchronize();
        t2 = clock();
        if ((status != CUSOLVER_STATUS_SUCCESS) || (cudaStatus != cudaSuccess)) {
            printf ("[cusolverRf status %d]\n",status);
            return TEST_FAILED;
        }
        printf("cusolverRfSolve time = %f (s)\n", (t2-t1)/(float)CLOCKS_PER_SEC);

        // extract the factors (if needed)
        status = cusolverRfExtractSplitFactorsHost(gH, &tnnzL, &tLp, &tLi, &tLx,
                                                &tnnzU, &tUp, &tUi, &tUx);
        if(status != CUSOLVER_STATUS_SUCCESS){
            printf ("[cusolverRf status %d]\n",status);
            return TEST_FAILED;
        }
        /*
        //print
        int row, j;
        printf("printing L\n");
        for (row=0; row<n; row++){
            for (j=tLp[row]; j<tLp[row+1]; j++){
                printf("\%d,\%d,\%f\n",row,tLi[j],tLx[j]);
            }
        }
        printf("printing U\n");
        for (row=0; row<n; row++){
            for (j=tUp[row]; j<tUp[row+1]; j++){
                printf("\%d,\%d,\%f\n",row,tUi[j],tUx[j]);
            }
        }
        */

        /* perform any other operations based on the solution */
        /* ... */

        /* check if done */
        /* ... */

        /* proceed to solve the next linear system */
        // update the coefficient matrix using reset values
        // (assuming that the new linear system, in other words,
        //  new values are already on the GPU in the array d_rAx)
        t1 = clock();
        status = cusolverRfResetValues(n,nnzA,d_Ap,d_Ai,d_rAx,d_P,d_Q,gH);
        cudaStatus = cudaDeviceSynchronize();
        t2 = clock();
        if ((status != CUSOLVER_STATUS_SUCCESS) || (cudaStatus != cudaSuccess)) {
            printf ("[cusolverRf status %d]\n",status);
            return TEST_FAILED;
        }
        printf("cusolverRf_reset_values time = %f (s)\n", (t2-t1)/(float)CLOCKS_PER_SEC);
    }

    /* free memory and exit */
    /* ... */
    return TEST_PASSED;
}

