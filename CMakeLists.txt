cmake_minimum_required(VERSION 3.5)
project(dmr)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(CUDA)
if (CUDA_FOUND)
    add_definitions("-DUSE_CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -DCUDA -lineinfo)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70)
    #    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --ptxas-options=-v)
    #    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount 64)
endif ()

include_directories(include)
include_directories(logging)
include_directories(arraygpu)

aux_source_directory(arraygpu lib_src)
MESSAGE("lib_src = ${lib_src}")

add_library(dmr ${lib_src} logging/easylogging++.cc)

add_executable(main main.cpp)

target_link_libraries(main dmr)

if (CUDA_FOUND)
    cuda_add_library(dmr_kernel arraygpu/Kernels.cu)
    target_link_libraries(main dmr_kernel)
endif ()

configure_file(logging/logging.conf logging.conf)
