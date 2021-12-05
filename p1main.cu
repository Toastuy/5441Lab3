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

    // Initialize every value of the matrix
    for(int i = 0; i < MATRIXSIZE; i++){
        for(int j = 0; j < MATRIXSIZE; j++){
            fMatrix[i][j] = (double) (i + j) / (double) MATRIXSIZE;
            numOfSerialFlops++;
        }
    }

    // Do our work
    for (int i = 0; i < MATRIXSIZE; i++){
        for (int j = 0; j < MATRIXSIZE; j++){
            // Redundant, can set for loop boundaries but this is self documenting
            // Check for our border conditions
            if (i != 0 && j != MATRIXSIZE - 1){
                fMatrix[i][j] = fMatrix[i - 1][j + 1] + fMatrix[i][j + 1];
                numOfSerialFlops++;
            }
        }
    }

    // This is for printing the serial values, not needed by project
    // for (int i = 0; i < MATRIXSIZE; i += 4){
    //     // Check for our edge case first
    //     if (i == MATRIXSIZE - 4){
    //         count += 2;
    //         printf("%f\n", fMatrix[i][1]);
    //         printf("%f\n", fMatrix[MATRIXSIZE - 1][1]);
    //     } 
    //     // If I'm not an edge case, just print the element and move on
    //     else {
    //         count++;
    //         printf("%f\n", fMatrix[i][1]);
    //     }
    // }
    return numOfSerialFlops;
}

__global__ void cudaP1(double *A, int matrixDimension){

    // TODO: Remove, checking to make sure I'm on the GPU at Owens

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize each element of the array to our equations
    // (i + j) / (double) 4000
    for (int i = 0; i < matrixDimension; i++) {
        A[x * matrixDimension + i] = ((x + i) / (double) 4000);
    }
}

__global__ void cudaP1_work(double *A, int matrixDimension){

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < matrixDimension; i++) {
        if (x != 0 && i != matrixDimension - 1) {
            A[x * matrixDimension + i] = A[(x - 1) * matrixDimension + i + 1] + A[x * matrixDimension + i + 1];
        }
    }
    
}

unsigned long parallelP1(){

    // Allocation
    // Flops: Initialization has MATRIXSIZE^2 amount of flops
    // Work has MATRIXSIZE^2 - (MATRIXSIZE * 2) amount of flops
    // Hard coded to avoid moving unnecessary data to our device
    unsigned long flops = (MATRIXSIZE * MATRIXSIZE) + (MATRIXSIZE * MATRIXSIZE) - (MATRIXSIZE * 2);
    unsigned long flopsPerSecond;
    cudaEvent_t start, stop, start2, stop2;
    float milliseconds = 0;
    float milliseconds2 = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    // Spaces in memory
    double *d_A;
    double *h_A = (double *) malloc(MATRIXSIZE * MATRIXSIZE * sizeof(double *));

    // Cuda memory block stuff
    int numOfBlocks = 4;
    int threadsPerBlock = 1000;

    // Declare the size of our matrix
    size_t matrixMemSize;
    matrixMemSize = MATRIXSIZE * MATRIXSIZE * sizeof(double *);

    // Malloc the spot for our array on the device
    gpuErrchk(cudaMalloc((void**) &d_A, matrixMemSize));

    // Get our dims for our cuda func call
    dim3 dimGrid(numOfBlocks);
    dim3 dimBlock(threadsPerBlock);

    // Call our cuda function to initialize the array and do our work
    // Measure the flops for this part as well, specified in write up
    cudaEventRecord(start);
    cudaP1<<<dimGrid, dimBlock>>>(d_A, MATRIXSIZE);
    cudaEventRecord(stop);

    printf("Post init check\n");

    // Copy our memory back over to the host
    gpuErrchk(cudaMemcpy(h_A, d_A, matrixMemSize, cudaMemcpyDeviceToHost));

    // Print the second element of every fourth row
    for (int i = 0; i < MATRIXSIZE; i+= 4){
        
        // i is row, multiply it by matrix size to get out access stride
        // add one to get the 2nd column
        printf("%lf\n", h_A[(i * MATRIXSIZE) + 1]);
        
        if (i == 3996){
            printf("%lf\n", h_A[((i * MATRIXSIZE) + 3) + 1]);
        }
    }

    // Call our cuda function to do our work
    cudaEventRecord(start2);
    cudaP1_work<<<dimGrid, dimBlock>>>(d_A, MATRIXSIZE);
    cudaEventRecord(stop2);

    // Timing stuff
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    milliseconds += milliseconds2;
    
    // Error checking goes here

    printf("Post work check\n");

    // Copy our memory back over to the host
    gpuErrchk(cudaMemcpy(h_A, d_A, matrixMemSize, cudaMemcpyDeviceToHost));

    // Print the second element of every fourth row
    for (int i = 0; i < MATRIXSIZE; i+= 4){
        
        // i is row, multiply it by matrix size to get out access stride
        // add one to get the 2nd column
        printf("%lf\n", h_A[(i * MATRIXSIZE) + 1]);
        
        if (i == 3996){
            printf("%lf\n", h_A[((i * MATRIXSIZE) + 3) + 1]);
        }
    }

    fprintf(stderr, "Total Time for CUDA in S: %f\n", milliseconds/1000);
    flopsPerSecond = flops / (milliseconds / 1000);
    
    // Free our memory
    gpuErrchk(cudaFree(d_A));

    return flopsPerSecond;
}

int main(){

    // Timing and flop stuff
    unsigned long serialFlops = 0;
    unsigned long serialFlopsPerSecond = 0;
    unsigned long parallelFlopsPerSecond = 0;
    double serialTime = 0.0;
    clock_t start;
    clock_t finish;

    // Serial Way
    start = clock();
    serialFlops = serialP1();
    finish = clock();
    
    serialTime = (double) (finish - start) / CLOCKS_PER_SEC;

    // Parallel Way
    parallelFlopsPerSecond = parallelP1();
    fprintf(stderr, "Total Time for Serial in S: %lf\n", serialTime);
    serialFlopsPerSecond = serialFlops / serialTime;
    fprintf(stderr, "Number of Serial Flops Per Second: %lu\n", serialFlopsPerSecond);
    fprintf(stderr, "Number of Parallel Flops Per Second: %lu\n", parallelFlopsPerSecond);
    
}
