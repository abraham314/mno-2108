#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<math.h>

#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include "global.h"
/********/
/* MAIN */
/********/
int main(){

        // --- gesvd only supports Nrows >= Ncols
        // --- column major memory ordering

        const long long Nrows = 200*200;
        const long long Ncols = 232;

        // --- cuSOLVE input/output parameters/arrays
        int work_size = 0;
        int *devInfo;                   gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));

        // --- CUDA solver initialization
        cusolverDnHandle_t solver_handle;
        cusolverDnCreate(&solver_handle);

        // --- Setting the host, Nrows x Ncols matrix
        float *h_A = (float *)malloc(Nrows * Ncols * sizeof(float));
        for(int j = 0; j < Nrows; j++)
                for(int i = 0; i < Ncols; i++)
                        h_A[j + i*Nrows] = (i + j*j) * sqrt((float)(i + j));

        // --- Setting the device matrix and moving the host matrix to the device
        float *d_A;                     gpuErrchk(cudaMalloc(&d_A,              Nrows * Ncols * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice));

        // --- host side SVD results space
        float *h_U = (float *)malloc(Nrows * Nrows     * sizeof(float));
        float *h_V = (float *)malloc(Ncols * Ncols     * sizeof(float));
        float *h_S = (float *)malloc(min(Nrows, Ncols) * sizeof(float));

        // --- device side SVD workspace and matrices
        float *d_U;                     gpuErrchk(cudaMalloc(&d_U,      Nrows * Nrows     * sizeof(float)));
        float *d_V;                     gpuErrchk(cudaMalloc(&d_V,      Ncols * Ncols     * sizeof(float)));
        float *d_S;                     gpuErrchk(cudaMalloc(&d_S,      min(Nrows, Ncols) * sizeof(float)));


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // --- CUDA SVD initialization
        cusolveSafeCall(cusolverDnSgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
        float *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));

        // --- CUDA SVD execution
//      cusolveSafeCall(cusolverDnSgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo));
        cusolveSafeCall(cusolverDnSgesvd(solver_handle, 'A', 'N', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, /*d_V*/NULL, Ncols, work, work_size, NULL, devInfo));
        int devInfo_h = 0;      gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) std::cout   << "Unsuccessful SVD execution\n\n";

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Elapsed time(ms): %f\n", milliseconds);

        // --- Moving the results from device to host
        gpuErrchk(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_V, d_V, Ncols * Ncols     * sizeof(float), cudaMemcpyDeviceToHost));
#if 0
        std::cout << "Singular values\n";
        for(int i = 0; i < min(Nrows, Ncols); i++)
                std::cout << "d_S["<<i<<"] = " << std::setprecision(15) << h_S[i] << std::endl;

        std::cout << "\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n";
        for(int j = 0; j < Nrows; j++) {
                printf("\n");
                for(int i = 0; i < Nrows; i++)
                        printf("U[%i,%i]=%f\n",i,j,h_U[j*Nrows + i]);
        }

        std::cout << "\nRight singular vectors - For y = A * x, the columns of V span the space of x\n";
        for(int i = 0; i < Ncols; i++) {
                printf("\n");
                for(int j = 0; j < Ncols; j++)
                        printf("V[%i,%i]=%f\n",i,j,h_V[j*Ncols + i]);
        }
#endif
        cusolverDnDestroy(solver_handle);
