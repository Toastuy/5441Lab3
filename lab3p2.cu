#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MATRIX_DIM 12
#define NUM_THREADS 10

unsigned long serial_part_2(float **A, float **B, float **C);
void cuda_part_2();


int main() {
  unsigned long flops = 0;
  time_t serialStart;
  time_t serialFinish;

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

  // Perform serial version
  time(&serialStart);
  flops = serial_part_2(A, B, C);
  time(&serialFinish);
  printf("FLOPS: %d\n C[00]: %f\n", flops, C[0][0]);

  printf("%d\n", serialFinish - serialStart);

  free(A);
  free(B);
  free(C);

  return 0;
}

unsigned long serial_part_2(float **A, float **B, float **C) {
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
  return flops;
}

void cuda_part_2() {

}