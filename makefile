all: part3

part3: part3.cu
		nvcc -o test part3.cu