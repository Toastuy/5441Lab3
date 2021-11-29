#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define MATRIX_DIM 4096

unsigned long serial_part_2();
unsigned long cuda_part_2();
__global__ void Device_Part_2(float *A, float *B, float *C, int dim);


int main() {
  unsigned long flops = 0;
  time_t serialStart;
  time_t serialFinish;


  // Perform serial version
  time(&serialStart);
  flops = serial_part_2();
  time(&serialFinish);

  printf("%d\n", serialFinish - serialStart);

  flops = cuda_part_2();

  return 0;
}

unsigned long serial_part_2() {
  float **A, **B, **C;
  float *row_ptr_helper_A;
  float *row_ptr_helper_B;
  float *row_ptr_helper_C;
  A = (float **) malloc(MATRIX_DIM * sizeof(float *) + MATRIX_DIM * MATRIX_DIM * sizeof(float));
  B = (float **) malloc(MATRIX_DIM * sizeof(float *) + MATRIX_DIM * MATRIX_DIM * sizeof(float));
  C = (float **) malloc(MATRIX_DIM * sizeof(float *) + MATRIX_DIM * MATRIX_DIM * sizeof(float));

  row_ptr_helper_A = (float *)(A + MATRIX_DIM);
  row_ptr_helper_B = (float *)(B + MATRIX_DIM);
  row_ptr_helper_C = (float *)(C + MATRIX_DIM);

  // Point row pointers to appropriate locations
  for (int i = 0; i < MATRIX_DIM; i++) {
    A[i] = (row_ptr_helper_A + MATRIX_DIM * i);
    B[i] = (row_ptr_helper_B + MATRIX_DIM * i);
    C[i] = (row_ptr_helper_C + MATRIX_DIM * i);
  }

  unsigned long flops = 0;
  // Initialize matrices for serial version
  for (int i = 0; i < MATRIX_DIM; i++) {
    for (int j = 0; j < MATRIX_DIM; j++) {
      A[i][j] = (float) rand() / RAND_MAX + 1;
      B[i][j] = (float) rand() / RAND_MAX + 1;
      flops++;
    }
  }

  float sum = 0;
  for (int i = 0; i < MATRIX_DIM; i++) {
    for (int j = 0; j < MATRIX_DIM; j++) {
      for (int k = 0; k < MATRIX_DIM; k++) {
        sum += A[i][k] * B[k][j];
        flops++;
      }
      C[i][j] = sum;
      sum = 0;
    }
  }

  free(A);
  free(B);
  free(C);
  return flops;
}

unsigned long cuda_part_2() {
  unsigned long flops = 0;
  float *d_A, *d_B, *d_C;
  float *h_C = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));
  int numBlocks = 8;
  int threadsPerBlock = 512;

  curandState *random;

  size_t matrix_mem_size;
  matrix_mem_size = MATRIX_DIM * MATRIX_DIM * sizeof(float *);

  cudaMalloc((void**) &d_A, matrix_mem_size);
  cudaMalloc((void**) &d_B, matrix_mem_size);
  cudaMalloc((void**) &d_C, matrix_mem_size);


  dim3 dimGrid(numBlocks);
  dim3 dimBlock(threadsPerBlock);

  Device_Part_2<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, MATRIX_DIM);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize())
  cudaMemcpy(h_C, d_C, matrix_mem_size, cudaMemcpyDeviceToHost);
  
  printf("%f\n", h_C[24]);
  return flops;
}

__global__ void Device_Part_2(float *A, float *B, float *C, int dim) {
    
    float sum;

    // TODO: Init with CuRand
    // int i = threadIdx.x;
    // for (int j = blockIdx.x; j < dim; j += gridDim.x) {
    //   A[i * dim + j] = 1.5;
    //   B[i * dim + j] = 1.7;
    // }


    int row = threadIdx.x + blockIdx.x * blockDim.x;

    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        A[row * dim + k] = 1.5;
        B[k * dim + j] = 1.7;
      }
    }


    // TODO: Adjust for 4096 dim
    for (int j = 0; j < dim; j++) {
      sum = 0;
      for (int k = 0; k < dim; k++) {
        sum += A[row * dim + k] * B[k * dim + j];
      }
      C[row * dim + j] = sum;
    }

}