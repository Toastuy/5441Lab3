#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Constants
const int MATRIXSIZE = 4000;

// Serial P1
unsigned long serialP1(){

    double fMatrix [MATRIXSIZE][MATRIXSIZE];
    unsigned long numOfSerialFlops = 0;
    int count = 0;

    for(int i = 0; i < MATRIXSIZE; i++){
        for(int j = 0; j < MATRIXSIZE; j++){
            // Initialize every value of the matrix
            fMatrix[i][j] = (double) (i + j) / (double) MATRIXSIZE;
        }
    }

    for (int i = 0; i < MATRIXSIZE; i += 4){
        // Check for our edge case first
        if (i == MATRIXSIZE - 4){
            count += 2;
            printf("%f\n", fMatrix[i][1]);
            printf("%f\n", fMatrix[MATRIXSIZE - 1][1]);
        } 
        // If I'm not an edge case, just print the element and move on
        else {
            count++;
            printf("%f\n", fMatrix[i][1]);
        }
    }
    printf("Total # Of Items: %d \n", count);
    return numOfSerialFlops;
}

__global__ void cudaP1(double *A, int matrixDimension){

    // TODO: Remove, checking to make sure I'm on the GPU at Owens
    printf("Hello from the GPU!");

    // Get our thread ID
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    // Get specific element assigned to thread
    int column = threadId % matrixDimension;
    int row = threadId / matrixDimension;

    // Initialize each element of the array to our equations
    // (i + j) / (double) 4000
    A[row][column] = ((row + column) / (double) 4000);
}

unsigned long parallelP1(){

    // Allocation
    // Flops
    unsigned long numOfParallelFlops = 0;

    // Spaces in memory
    double *d_A;
    double *h_A = (double *) malloc(MATRIXSIZE * MATRIXSIZE * sizeof(double *));

    // Cuda memory block stuff
    int numOfBlocks = 4;
    int threadsPerBlock = 512;

    // Declare the size of our matrix
    size_t matrixMemSize;
    matrixMemSize = MATRIXSIZE * MATRIXSIZE * sizeof(double *);

    // Malloc the spot for our array on the device
    gpuErrchk(cudaMalloc((void**) &d_A, matrixMemSize));

    // Get our dims for our cuda func call
    dim3 dimGrid(numOfBlocks);
    dim3 dimBlock(threadsPerBlock);

    // Call our cuda function to initialize the array and do our work
    cudaP1<<<dimGrid, dimBlock>>>(d_A, MATRIXSIZE);
    
    // Error checking goes here

    // Copy our memory back over to the host
    gpuErrchk(cudaMemcpy(h_A, d_A, matrixMemSize, cudaMemcpyDeviceToHost));

    // Free our memory
    gpuErrchk(cudaFree(d_A));

    // Print the second element of every fourth row
    for (int i = 0; i < MATRIXSIZE; i += 4){
        // Check for our edge case first
        if (i == MATRIXSIZE - 4){
            printf("%f\n", h_A[i][1]);
            printf("%f\n", h_A[MATRIXSIZE - 1][1]);
        } 
        // If I'm not an edge case, just print the element and move on
        else {
            printf("%f\n", h_A[i][1]);
        }
    }

    return numOfParallelFlops;
}

int main(){

    // Timing and flop stuff
    unsigned long serialFlops = 0;
    unsigned long parallelFlops = 0;
    time_t start;
    time_t finish;

    // Serial Way
    time(&start);
    serialFlops = serialP1();
    time(&finish);
    printf("Number of Serial Flops: %lu\n", serialFlops);

    // Parallel Way
    time(&start);
    parallelFlops = parallelP1();
    time(&finish);
    printf("Number of Parallel Flops: %lu\n", parallelFlops);
    
}