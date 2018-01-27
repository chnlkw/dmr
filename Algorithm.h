//
// Created by chnlkw on 1/18/18.
//

#ifndef DMR_ALGORITHM_H
#define DMR_ALGORITHM_H

#include "Context.h"
#include "Array.h"
#include "Data.h"
#include "Kernels.h"
#include "DataCopy.h"

//void shuffle_by_idx_gpu_ff(float *dst, const float *src, const size_t *idx, size_t size);

namespace Algorithm {

template<class V1, class V2, class V3>
void ShuffleByIdx(DevicePtr device, V1 &, const V2 &, const V3 &);

template<class T, class TOff>
void ShuffleByIdx(DevicePtr device, Array<T> &dst, const Array<T> &src, const Array<TOff> &idx) {
    int dev_id = dst.GetDevice();
    assert(dev_id == src.GetDevice());
    assert(dev_id == idx.GetDevice());
    size_t size = dst.size();
    assert(size == src.size());
    assert(size == idx.size());

//    printf("ShuffleByIdx size=%d, dev=%d\n", size, dev_id);

    if (dev_id < 0) { //CPU
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] = src[idx[i]];
        }
    } else {
        CUDA_CALL(cudaSetDevice, dev_id);
        shuffle_by_idx_gpu(dst.data(), src.data(), idx.data(), size);
        CUDA_CHECK();
    }
}

template<class T, class TOff>
void ShuffleByIdx(DevicePtr device, std::vector<T> &dst, const std::vector<T> &src, const std::vector<TOff> &idx) {
    size_t size = dst.size();
    assert(size == src.size());
    assert(size == idx.size());

    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = src[idx[i]];
    }
}

template<class T, class TOff>
void ShuffleByIdx(DevicePtr device, Data<T> &dst, const Data<T> &src, const Data<TOff> &idx) {
    ShuffleByIdx(device,
                 *dst.GetFrom(device),
                 *src.GetFrom(device),
                 *idx.GetFrom(device)
    );
}

template<class V1, class V2, class V3>
void ShuffleByIdx(V1 &dst, const V2 &src, const V3 &idx) {
    ShuffleByIdx(GetDevice(dst), dst, src, idx);
};

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

template<class T>
Data<T> Renew(const Data<T> &in, size_t count) {
    return Data<T>(count, in.DeviceCurrent());
}

template<class V>
void Copy(const V &src, size_t src_off, V &dst, size_t dst_off, size_t count);

template<class T>
void Copy(const std::vector<T> &src, size_t src_off, std::vector<T> &dst, size_t dst_off, size_t count) {
    std::copy(src.begin() + src_off, src.begin() + src_off + count, dst.begin() + dst_off);
}


template<class T>
void Copy(const Data<T> &src, size_t src_off, Data<T> &dst, size_t dst_off, size_t count) {
//    std::cout << "copy " << src.ToString() << " to " << dst.ToString() << std::endl;
    DataCopy(dst.begin() + dst_off, dst.DeviceCurrent()->Id(),
             src.begin() + src_off, src.DeviceCurrent()->Id(),
             count * sizeof(T));
//    std::copy(src.begin() + src_off, src.begin() + src_off + count, dst.begin() + dst_off);
}

}

#endif //DMR_ALGORITHM_H
