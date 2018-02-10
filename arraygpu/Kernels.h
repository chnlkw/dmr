//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_KERNELS_H
#define DMR_KERNELS_H

#include <cstdio>

template<class T, class TOff>
extern void shuffle_by_idx_gpu(T *dst, const T *src, const TOff *idx, size_t size, cudaStream_t stream = 0);

template<class T>
extern void gpu_add(T *c, const T *a, const T *b, size_t size, cudaStream_t stream); // c[i] = a[i] + b[i]


#endif //DMR_KERNELS_H
