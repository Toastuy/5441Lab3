#include <stdio.h>

// Constants
const int MATRIXSIZE = 4000;

// Serial P1
int serialP1(){
    double fMatrix [MATRIXSIZE][MATRIXSIZE];
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
    return 1;
}

__global__ void parallelP1(double *x, double *y, int matSize){
    // This will init array and compute all the calculations
    // Then copy the device memory over to host memory to read from
    printf("Hello from the GPU!");

    // Matrix
    double fMatrix [matSize][matSize];

    // Initialize memory
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

}

int main(){

    // Serial Way
    serialP1();

    // Parallel Way

    // Allocation
    double *x, *y;
    double *d_x, *d_y;
    x = new double(MATRIXSIZE * MATRIXSIZE);
    y = new double(MATRIXSIZE * MATRIXSIZE);

    // Thread Hierarchy
    int nblocks = 4;
    int tpb = 512;

    // Allocate device memory
    size_t memSize;
    memSize = MATRIXSIZE * MATRIXSIZE * sizeof(double);
    cudaMalloc((void**) &d_x, memSize);
    cudaMalloc((void**) &d_y, memSize);

    // Initialize memory to device
    // TODO: Do this in GPU
    //cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_y, y, memSize, cudaMemcpyHostToDevice);

    // First we need to build our dims
    dim3 dimGrid(nblocks);
    dim3 dimBlock(tpb);

    // Launch Kernal
    parallelP1<<<dimGrid, dimBlock>>>(d_x, d_y, MATRIXSIZE);

    cudaMemcpy(x, d_x, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, memSize, cudaMemcpyDeviceToHost);

}