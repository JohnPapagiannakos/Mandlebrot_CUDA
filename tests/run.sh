#--- requires cuda v. 11.0 or later (in order to add -std=c++17)---.

nvcc --gpu-architecture=sm_75 -lcudart -lcuda -lcublas ../include/cudaoperators.cu -O3 -c -o ../bin/cudaoperators.o

nvcc -L/usr/local/cuda-11.2/lib64 --gpu-architecture=sm_75 -lcudart -lcuda -std=c++17\
     -I../include ../bin/cudaoperators.o test_julia.cu -Xcompiler "-O3 -march=native -mtune=native" -o ../bin/test_julia
