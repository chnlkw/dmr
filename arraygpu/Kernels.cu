//
// Created by chnlkw on 1/23/18.
//

#include "Kernels.h"

template<class T, class TOff>
__global__ void shuffle_by_idx_kernel(T *dst, const T *src, const TOff *idx, size_t size) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        dst[i] = src[idx[i]];
};

template<class T, class TOff>
void shuffle_by_idx_gpu(T *dst, const T *src, const TOff *idx, size_t size, cudaStream_t stream) {

    shuffle_by_idx_kernel << < (size + 31) / 32, 32, 0, stream >> > (dst, src, idx, size);
}

template void
shuffle_by_idx_gpu<float, size_t>(float *dst, const float *src, const size_t *idx, size_t size, cudaStream_t stream);

template void
shuffle_by_idx_gpu<unsigned int, size_t>(unsigned int *dst, const unsigned int *src, const size_t *idx, size_t size,
                                         cudaStream_t stream);

template void shuffle_by_idx_gpu<unsigned int, unsigned int>(unsigned int *, unsigned int const *, unsigned int const *,
                                                             unsigned long, cudaStream_t);

template<class T>
__global__ void gpu_add_kernel(T *c, const T *a, const T *b, size_t size) { // c[i] = a[i] + b[i]
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

template<class T>
void gpu_add<>(T *c, const T *a, const T *b, size_t size, cudaStream_t stream) { // c[i] = a[i] + b[i]
    gpu_add_kernel << < (size + 31) / 32, 32, 0, stream >> > (c, a, b, size);
}

template void gpu_add<int>(int *, const int *, const int *, size_t, cudaStream_t stream);
