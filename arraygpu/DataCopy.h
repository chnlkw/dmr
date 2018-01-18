//
// Created by chnlkw on 11/21/17.
//

#ifndef LDA_DATACOPY_H
#define LDA_DATACOPY_H

#include "cuda_utils.h"
#include <map>

extern std::map<int, std::map<int, bool>> data_copy_p2p;

void DataCopy(void *dst_ptr, int dst_device, void *src_ptr, int src_device, size_t bytes);

void DataCopyAsync(void *dst_ptr, int dst_device, void *src_ptr, int src_device, size_t bytes, cudaStream_t stream);

void DataCopyInitP2P();

#endif //LDA_DATACOPY_H
