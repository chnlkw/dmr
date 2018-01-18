//
// Created by chnlkw on 1/18/18.
//

#ifndef DMR_CONTEXT_H
#define DMR_CONTEXT_H

#include "Allocator.h"

class Context {
private:
    int device_;
    cudaStream_t stream_;
    MultiDeviceAllocator allocator_;
public:
    Context();

    void SetDevice(int device_id);

    int GetDevice() const;

    void SetStream(cudaStream_t stream = 0);

    cudaStream_t GetStream() const;

    AllocatorBase* GetAllocator();

//    void *Alloc(size_t size);

//    void Free(void *ptr);
};

extern Context g_context;

#endif //DMR_CONTEXT_H
