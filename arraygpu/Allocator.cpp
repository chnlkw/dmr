//
// Created by chnlkw on 1/16/18.
//

#include "Allocator.h"

#define LG(x) CLOG(x, "Allocator")

#ifdef USE_CUDA

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

#endif

CudaPreAllocator::CudaPreAllocator(int device, size_t pre_alloc_bytes, size_t align) :
        CudaAllocator(device),
        size_(pre_alloc_bytes),
        allocated_(0),
        align_(align) {

    if (device_ < 0) {
        CUDA_CALL(cudaMallocHost, &ptr_, size_);
    } else {
        CUDA_CALL(cudaSetDevice, device_);
        CUDA_CALL(cudaMalloc, &ptr_, size_);
    }
}

CudaPreAllocator::~CudaPreAllocator() {
    if (allocated_ > 0) {
        fprintf(stderr, "[WARN] dangling pointer to CudaPreAllocater, allocated size = %lu\n", allocated_);
    }
    if (device_ < 0) {
        CUDA_CALL(cudaFreeHost, ptr_);
    } else {
        CUDA_CALL(cudaSetDevice, device_);
        CUDA_CALL(cudaFree, ptr_);
    }
}

void *CudaPreAllocator::Alloc(size_t size) {
    auto align_up = [this](size_t off) {
        off += align_ - 1;
        off &= ~(align_ - 1);
        return off;
    };
    off_t off = 0;
    for (auto p : m_) {
        auto beg = p.first;
        auto end = p.first + p.second;
        if (off + size > beg) {
            off = align_up(end);
        } else {
            break;
        }
    }
    if (off + size > size_) {
        std::ostringstream os;
        os << "CudaPreAllocator :: not enough memory when allocating " << size << " remain " << size_ - allocated_;
        throw std::runtime_error(os.str().c_str());
    }
    m_.emplace(off, size);
    allocated_ += size;
    LG(INFO) << "CurePreAllocator: Alloc=" << size << " allocated=" << allocated_ << " remain=" << size_ - allocated_;
    return (char *) ptr_ + off;
}

void CudaPreAllocator::Free(void *ptr) {
    off_t off = (char *) ptr - (char *) ptr_;
    auto it = m_.find(off);
    if (it == m_.end()) {
        std::ostringstream os;
        os << "CudaPreAllocator :: Free pointer not found ptr=" << ptr << " off = " << off;
        throw std::runtime_error(os.str().c_str());
    }
    allocated_ -= it->second;
    LG(INFO) << "CurePreAllocator: free=" << it->second << " allocated=" << allocated_ << " remain=" << size_ - allocated_;
    m_.erase(it);
}
