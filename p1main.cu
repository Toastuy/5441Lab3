#include <stdio.h>

// Constants
const int MATRIXSIZE = 4000;

// Serial P1
// TODO: Remove Debug Print Statements
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

int main(){

    // Serial Way
    serialP1();

    // Parallel Way
    // Pointers for our host and device
    //int *h_a;
    int *d_a;
    size_t memSize;

    memSize = 30;
    cudaMalloc((void**) &d_a, memSize);

    // parallel

}