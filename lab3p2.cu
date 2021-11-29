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
__global__ void Initialize_Arrays_Part_2(float *A, float *B, int dim);


int main() {
  unsigned long flops = 0;
  time_t start;
  time_t finish;


  // Perform serial version
  time(&start);
  //flops = serial_part_2();
  time(&finish);

  printf("Flops: %lu\n", flops);

  time(&start);
  flops = cuda_part_2();
  time(&finish);

  printf("Flops: %lu\n", flops);

  return 0;
}

unsigned long serial_part_2() {
  float *A, *B, *C;
  A = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));
  B = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));
  C = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));

  unsigned long flops = 0;
  // Initialize matrices for serial version
  for (int i = 0; i < MATRIX_DIM; i++) {
    for (int j = 0; j < MATRIX_DIM; j++) {
      A[i * MATRIX_DIM + j] = (float) rand() / RAND_MAX + 1;
      B[i * MATRIX_DIM + j] = (float) rand() / RAND_MAX + 1;
      flops++;
    }
  }

  float sum = 0;
  for (int i = 0; i < MATRIX_DIM; i++) {
    for (int j = 0; j < MATRIX_DIM; j++) {
      for (int k = 0; k < MATRIX_DIM; k++) {
        sum += A[i * MATRIX_DIM + k] * B[k * MATRIX_DIM + j];
        flops++;
      }
      C[i * MATRIX_DIM + j] = sum;
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
  float *h_A = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));
  float *h_B = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));
  float *h_C = (float *) malloc(MATRIX_DIM * MATRIX_DIM * sizeof(float *));
  int numBlocks = 256;
  int threadsPerBlock = 256;

  size_t matrix_mem_size;
  matrix_mem_size = MATRIX_DIM * MATRIX_DIM * sizeof(float *);

  gpuErrchk(cudaMalloc((void**) &d_A, matrix_mem_size));
  gpuErrchk(cudaMalloc((void**) &d_B, matrix_mem_size));
  gpuErrchk(cudaMalloc((void**) &d_C, matrix_mem_size));


  dim3 dimGrid(numBlocks, numBlocks);
  dim3 dimBlock(threadsPerBlock);

  // Initialize arrays on GPU
  Initialize_Arrays_Part_2<<< dimGrid, dimBlock >>>(d_A, d_B, MATRIX_DIM);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_A, d_A, matrix_mem_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_B, d_B, matrix_mem_size, cudaMemcpyDeviceToHost));

  // Pass initialized array and output array to matrix multiply funciton
  Device_Part_2<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, MATRIX_DIM);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_C, d_C, matrix_mem_size, cudaMemcpyDeviceToHost));
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  printf("%f\n", h_C[22]);
  return flops;
}

__global__ void Initialize_Arrays_Part_2(float *A, float *B, int dim) {
    // Get global linearized thread ID
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    // Get specific element assigned to thread
    int column = threadId % dim;
    int row = threadId / dim;

    // Each thread initializes a single element in each array
    A[row * dim + column] = 1.2;
    B[row * dim + column] = 1.3;
}

__global__ void Device_Part_2(float *A, float *B, float *C, int dim) {

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    // Get specified row and column item to compute
    int column = threadId % dim;
    int row = threadId / dim;

    // Multiply each row and column element to get computed C value
    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += A[row * dim + j] * B[j * dim + column];
    }
    C[row * dim + column] = sum;

}