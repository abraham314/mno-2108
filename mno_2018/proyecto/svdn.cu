# include <time.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <cuda_runtime.h>
# include <cublas_v2.h>
# include <cusolverDn.h>
# define BILLION 1000000000 L ;
int main (int argc , char * argv [])
{
struct timespec start , stop ; // variables for timing
double accum ; // elapsed time variable
cusolverDnHandle_t cusolverH ; // cusolver handle
cublasHandle_t cublasH ; // cublas handle
cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS ;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS ;
cudaError_t cudaStat = cudaSuccess ;
const int m = 2048; // number of rows of A
const int n = 2048; // number of columns of A
const int lda = m; // leading dimension of A
// declare the factorized matrix A, orthogonal matrices U, VT
float *A, *U, *VT , *S; // and sing .val. matrix S on the host
A=( float *) malloc (lda*n* sizeof ( float ));
U=( float *) malloc (lda*m* sizeof ( float ));
VT =( float *) malloc (lda *n* sizeof ( float ));
S= ( float *) malloc (n* sizeof ( float ));
for(int i=0;i<lda*n;i++) A[i]= rand ()/( float ) RAND_MAX ;
// the factorized matrix d_A , orthogonal matrices d_U , d_VT
float *d_A , *d_U , *d_VT , *d_S; // and sing .val. matrix d_S
int * devInfo ; // on the device
float *d_work , * d_rwork ; // workspace on the device
float *d_W; // auxiliary device array (d_W = d_S* d_VT )
int lwork = 0;
int info_gpu = 0; // info copied from device to host
const float h_one = 1;
const float h_minus_one = -1;
// create cusolver and cublas handle
cusolver_status = cusolverDnCreate (& cusolverH );
cublas_status = cublasCreate (& cublasH );
// prepare memory on the device
cudaStat = cudaMalloc (( void **)& d_A , sizeof ( float )* lda*n);
cudaStat = cudaMalloc (( void **)& d_S , sizeof ( float )*n);
cudaStat = cudaMalloc (( void **)& d_U , sizeof ( float )* lda*m);
cudaStat = cudaMalloc (( void **)& d_VT , sizeof ( float )* lda*n);
cudaStat = cudaMalloc (( void **)& devInfo , sizeof (int ));
cudaStat = cudaMalloc (( void **)& d_W , sizeof ( float )* lda*n);
cudaStat = cudaMemcpy (d_A , A, sizeof ( float )* lda*n,
cudaMemcpyHostToDevice ); // copy A- >d_A
// compute buffer size and prepare workspace
cusolver_status = cusolverDnSgesvd_bufferSize ( cusolverH ,m,n,
& lwork );
cudaStat = cudaMalloc (( void **)& d_work , sizeof ( float )* lwork );
// compute the singular value decomposition of d_A
// and optionally the left and right singular vectors :
// d_A = d_U *d_S * d_VT ; the diagonal elements of d_S 
// are the singular values of d_A in descending order
// the first min (m,n) columns of d_U contain the left sing .vec .
// the first min (m,n) cols of d_VT contain the right sing .vec .
signed char jobu = ’A’; // all m columns of d_U returned
signed char jobvt = ’A’; // all n columns of d_VT returned
clock_gettime ( CLOCK_REALTIME ,& start ); // start timer
cusolver status = cusolverDnSgesvd (cusolverH, jobu, jobvt,
m, n, d A, lda, d S, d U, lda, d VT, lda, d work, lwork,
d rwork, devInfo);
cudaStat = cudaDeviceSynchronize ();
clock_gettime ( CLOCK_REALTIME ,& stop ); // stop timer
accum =( stop .tv_sec - start . tv_sec )+ // elapsed time
( stop . tv_nsec - start . tv_nsec )/( double ) BILLION ;
printf ("SVD time : %lf sec .\n",accum ); // print elapsed time
cudaStat = cudaMemcpy (U,d_U , sizeof ( float )* lda*m,
cudaMemcpyDeviceToHost ); // copy d_U - >U
cudaStat = cudaMemcpy (VT ,d_VT , sizeof ( float )* lda*n,
cudaMemcpyDeviceToHost ); // copy d_VT - >VT
cudaStat = cudaMemcpy (S,d_S , sizeof ( float )*n,
cudaMemcpyDeviceToHost ); // copy d_S - >S
cudaStat = cudaMemcpy (& info_gpu , devInfo , sizeof (int) ,
cudaMemcpyDeviceToHost ); // devInfo - > info_gpu
printf (" after gesvd : info_gpu = %d\n", info_gpu );
// multiply d_VT by the diagonal matrix corresponding to d_S
cublas_status = cublasSdgmm ( cublasH , CUBLAS_SIDE_LEFT ,n,n,
d_VT , lda , d_S , 1 , d_W , lda ); // d_W =d_S * d_VT
cudaStat = cudaMemcpy (d_A ,A, sizeof ( float )* lda*n,
cudaMemcpyHostToDevice ); // copy A- >d_A
// compute the difference d_A -d_U *d_S * d_VT
cublas_status = cublasSgemm_v2 ( cublasH , CUBLAS_OP_N , CUBLAS_OP_N ,
m, n, n, & h_minus_one ,d_U , lda , d_W , lda , &h_one , d_A , lda );
float dR_fro = 0.0; // variable for the norm
// compute the norm of the difference d_A -d_U *d_S * d_VT
cublas_status = cublasSnrm2_v2 ( cublasH ,lda*n,d_A ,1 ,& dR_fro );
printf ("|A - U*S*VT| = %E \n", dR_fro ); // print the norm
// free memory
cudaFree (d_A );
cudaFree (d_S );
cudaFree (d_U );
cudaFree ( d_VT );
cudaFree ( devInfo );
cudaFree ( d_work );
cudaFree ( d_rwork );
cudaFree (d_W );
cublasDestroy ( cublasH );
cusolverDnDestroy ( cusolverH );
cudaDeviceReset ();
return 0;
}
