#include "cuda_runtime.h"
#include <stdio.h>
#include <chrono>
#include <string>
#include <cstdio>
#include <math.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void InplaceTranspose(int *mat, int mat_dim, int cells_per_tile_x, int tiles_per_grid_x) {

    int temp;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int init_i = x/tiles_per_grid_x * cells_per_tile_x;
    int init_j = x%tiles_per_grid_x * cells_per_tile_x;
        
    int count = 0;
    for(int i = 0; i < cells_per_tile_x; i++) {
        for(int j = 0; j < cells_per_tile_x; j++) {

            if ((i+init_i)*mat_dim + (j+init_j) < mat_dim*mat_dim && (j+init_j)*mat_dim + (i+init_i) < mat_dim*mat_dim && (i + init_i) > (j + init_j)) {
                count++;
                temp = mat[(i+init_i)*mat_dim + (j+init_j)];
                mat[(i+init_i)*mat_dim + (j+init_j)] = mat[(j+init_j)*mat_dim + (i+init_i)];
                mat[(j+init_j)*mat_dim + (i+init_i)] = temp;
            }
        }
    }
}

__host__ void Transpose(int blocks, int threads) {
    int n_blocks = blocks;
    int threads_pb = threads;

    fprintf(stderr, "Test %d block with %d threads\n", n_blocks, threads_pb);

    int* device_mat;

    // Creating matrix
    int* host_mat;

    // Copy of matrix for checking result
    int* host_mat_copy;

    // Total mat dim and tile dim 
    int mat_dim = 1024;
    
    // Used for workload distribution in kernel
    int cells_per_tile_x = (int)ceil(sqrt((double)(mat_dim*mat_dim)/(n_blocks * pow(floor(sqrt(threads_pb)), 2))));
    int tiles_per_grid_x = ceil((double)mat_dim/cells_per_tile_x);

    host_mat = new int[mat_dim*mat_dim];
    host_mat_copy = new int[mat_dim*mat_dim];

    // Initializing matrix to rand values 0-1000
    srand(time(NULL));
    for (int i = 0; i < mat_dim*mat_dim; i++) {
        host_mat[i] = rand() % 1000 + 1;
        host_mat_copy[i] = host_mat[i];
    }

    // Allocate on device 
    size_t memSize = mat_dim*mat_dim*sizeof(int);
    gpuErrchk(cudaMalloc((void**)&device_mat, memSize));

    // Initialize on device
    gpuErrchk(cudaMemcpy(device_mat, host_mat, memSize, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimGrid(n_blocks);
    dim3 dimBlock(threads_pb);
    
    // Set up events for kernel execution timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start event
    cudaEventRecord(start);

    InplaceTranspose<<<dimGrid, dimBlock>>>(device_mat, mat_dim, cells_per_tile_x, tiles_per_grid_x);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Stop event
    cudaEventRecord(stop);

    // Get transpose time in ms
    cudaEventSynchronize(stop);
    float transposeTime = 0;
    cudaEventElapsedTime(&transposeTime, start, stop);

    fprintf(stderr, "Transpose time: %f ms\n", transposeTime);

    // Retrieve results
    gpuErrchk(cudaMemcpy(host_mat, device_mat, memSize, cudaMemcpyDeviceToHost));
    
    // Checking result
    int total = 0;
    int correct = 0;
    for (int i = 0; i < mat_dim; i++) {
        for(int j = 0; j < mat_dim; j++) {
            std::string match = "false";
            total++;
            if (host_mat_copy[i * mat_dim + j] == host_mat[j * mat_dim + i]) {
                match = "true";
                correct++;
            }
            else {
                // fprintf(stderr, "Element (%d, %d) does NOT map to tranposed element (%d, %d)\n", i, j, j, i);
            }
        }
    }

    fprintf(stderr, "Transpose: %d Correct Elements / %d Total Elements\n", correct, total);
    fprintf(stderr, "---------------------------------------------------------------------------\n");

    cudaFree(device_mat);
}

int main() {
    
    Transpose(1, 32);
    Transpose(1, 1024);
    Transpose(2, 1024);
    Transpose(64, 1024);

    return 0;
}