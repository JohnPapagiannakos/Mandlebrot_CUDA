#--- requires cuda v. 11.0 or later (in order to add -std=c++17)---.

nvcc --gpu-architecture=sm_75 -lcudart -lcuda -lcublas ../include/operators.cu -O3 -c -o ../bin/operators.o

nvcc -L/usr/local/cuda-11.2/lib64 --gpu-architecture=sm_75 -lcudart -lcuda -std=c++17\
     -I../include ../bin/operators.o test_julia.cpp -Xcompiler "-O3 -march=native -mtune=native" -o ../bin/test_julia
