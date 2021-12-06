all: part1 part2

part1: p1main.cu
	nvcc -o lab3p1_dmilasky_milasky.3 p1main.cu

part2: lab3p2.cu
	nvcc -0 lab3p2_dmilasky_milasky.3 lab3p2.cu