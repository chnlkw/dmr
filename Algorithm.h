//
// Created by chnlkw on 1/18/18.
//

#ifndef DMR_ALGORITHM_H
#define DMR_ALGORITHM_H

#include "arraygpu/Array.h"

template<class T, class TOff>
void shuffle_by_idx_gpu(T *dst, const T *src, const TOff *idx, size_t size);

//void shuffle_by_idx_gpu_ff(float *dst, const float *src, const size_t *idx, size_t size);

namespace Algorithm {

template<class V1, class V2, class V3>
void ShuffleByIdx(V1, const V2 &src, const V3 &idx);

template<class T, class TOff>
void ShuffleByIdx(Array<T> &dst, const Array<T> &src, const Array<TOff> &idx) {
    int device = dst.GetDevice();
    assert(device == src.GetDevice());
    assert(device == idx.GetDevice());
    size_t size = dst.size();
    assert(size == src.size());
    assert(size == idx.size());

    if (device < 0) { //CPU
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] = src[idx[i]];
        }
    } else {
        CUDA_CALL(cudaSetDevice, device);
        shuffle_by_idx_gpu(dst.data(), src.data(), idx.data(), size);
    }
}

template<class T, class TOff>
void ShuffleByIdx(std::vector<T> &dst, const std::vector<T> &src, const std::vector<TOff> &idx) {
    size_t size = dst.size();
    assert(size == src.size());
    assert(size == idx.size());

    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = src[idx[i]];
    }

}

template<class V>
V Renew(const V &in, size_t count = 0);

template<class T>
std::vector<T> Renew(const std::vector<T> &in, size_t count) {
    return std::vector<T>(count);
}

template<class T>
Array<T> Renew(const Array<T> &in, size_t count) {
    return in.Renew(count);
}

}

#endif //DMR_ALGORITHM_H
