cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
enable_language(CUDA)
project(mandlebrot)

option(USE_CUDA "Use CUDA" ON)

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)

set(CUDA_INCLUDE_DIRS            /usr/local/cuda-11.2/include/ ${CUDA_INCLUDE_DIRS})


SET(INCLUDE_DIR                  "${CMAKE_CURRENT_LIST_DIR}/include")
SET(BIN_DIR                      "${CMAKE_CURRENT_LIST_DIR}/bin")


SET(TEST_SRC_DIR                 ${CMAKE_CURRENT_LIST_DIR})
# SET(TEST_SRC_FILES               ${TEST_SRC_DIR}/CUDA/test_julia.cu)

SET(TEST_HEADER_FILES            ${INCLUDE_DIR})
INCLUDE_DIRECTORIES(             ${TEST_HEADER_FILES})

set(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS} -std=c++17")
SET(CMAKE_CXX_FLAGS_RELEASE      "-O3 -fopenmp -funroll-loops -march=native -mtune=native ")
set(CMAKE_CXX_FLAGS_DEBUG        "-g")

add_subdirectory(include)

add_subdirectory(tests/CUDA)

add_subdirectory(tests/CPU)

add_subdirectory(tests/OpenGL)

add_subdirectory(tests/compute_and_plot)