//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_KERNELS_H
#define DMR_KERNELS_H

#include <cstdio>

template<class T, class TOff>
extern void shuffle_by_idx_gpu(T *dst, const T *src, const TOff *idx, size_t size);


#endif //DMR_KERNELS_H
