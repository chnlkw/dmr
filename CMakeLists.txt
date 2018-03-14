cmake_minimum_required(VERSION 3.5)
project(dmr)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    set(USE_CUDA ON)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -DCUDA -lineinfo)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-extended-lambda)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70)
    #    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --ptxas-options=-v)
    #    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount 64)
endif ()

#configure_file(config.h.in config.h @ONLY)

#aux_source_directory(arraygpu lib_src)
#MESSAGE("lib_src = ${lib_src}")

add_subdirectory(arraygpu)

add_executable(main main.cpp)
target_link_libraries(main engine)

include_directories(arraygpu/src)
include_directories(arraygpu/logging)
include_directories(arraygpu/cuda)
#include_directories(arraygpu/include)
cuda_add_library(dmr_kernel Kernels.cu)

add_library(dmr dmr.cpp)
target_link_libraries(dmr dmr_kernel engine)
target_include_directories(dmr
        PUBLIC .
)

target_link_libraries(main dmr_kernel)
configure_file(arraygpu/logging/logging.conf logging.conf)
