//
// Created by chnlkw on 1/18/18.
//

#include "Algorithm.h"

//void shuffle_by_idx_gpu_ff(float *dst, const float *src, const size_t *idx, size_t size) {
////    shuffle_by_idx_kernel<T, TOff> << < (size + 31 / 32) * 32, 32 >> > (dst, src, idx, size);
//    CUDA_CHECK();
//
//    int dev;
//    cudaGetDevice(&dev);
//    printf("run kernel %d\n", dev);
//    int numBlock = (size + 31) / 32;
//    shuffle_by_idx_kernel_ff << < numBlock, 32 >> > (dst, src, idx, size);
//    CUDA_CHECK();
//    printf("fin kernel\n");
//}

template void shuffle_by_idx_gpu<float, size_t>(float *dst, const float *src, const size_t *idx, size_t size);
template void shuffle_by_idx_gpu<uint32_t, size_t>(uint32_t *dst, const uint32_t *src, const size_t *idx, size_t size);

//template __global__ void shuffle_by_idx_kernel<float, size_t>(float *dst, const float *src, const size_t *idx, size_t size);
