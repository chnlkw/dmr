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
