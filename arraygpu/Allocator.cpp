//
// Created by chnlkw on 1/16/18.
//

#include "Allocator.h"

int UnifiedAllocator::GetDevice() {
    return device_;
}

void UnifiedAllocator::SetDevice(int device_id) {
    device_ = device_id;
}

UnifiedAllocator::UnifiedAllocator() :
        device_(-1),
        allocator_([](int device) { return AllocatorPtr(new CudaAllocator(device)); }) {
}

void *UnifiedAllocator::Alloc(size_t size) {
    return allocator_.Alloc(size, device_);
}

void UnifiedAllocator::Free(void *ptr) {
    allocator_.Free(ptr);
}
