//
// Created by chnlkw on 1/18/18.
//

#include "Context.h"

Context::Context() :
        device_(-1),
        allocator_([](int device) -> AllocatorBase * { return new CudaAllocator(device); }) {
}

int Context::GetDevice() const {
    return device_;
}

void Context::SetDevice(int device_id) {
    device_ = device_id;
}

void Context::SetStream(cudaStream_t stream) {
    stream_ = stream;
}

cudaStream_t Context::GetStream() const {
    return stream_;
}

AllocatorBase *Context::GetAllocator() {
    return allocator_.GetAllocatorByDevice(device_);
}

//void *Context::Alloc(size_t size) {
//    return allocator_.Alloc(size, device_);
//}
//
//void Context::Free(void *ptr) {
//    allocator_.Free(ptr);
//}

Context g_context;