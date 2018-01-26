//
// Created by chnlkw on 11/20/17.
//

#ifndef LDA_ALLOCATOR_H
#define LDA_ALLOCATOR_H

#include <map>
#include <iostream>
#include <memory>
#include <deque>
#include <sstream>

#include "cuda_utils.h"
#include "defs.h"

class AllocatorBase {

public:
    AllocatorBase() {}

    virtual ~AllocatorBase() {}

    virtual void *Alloc(size_t size) = 0;

    virtual void Free(void *ptr) = 0;
};

class CPUAllocator : public AllocatorBase {
public:
    void *Alloc(size_t size) {
        return malloc(size);
    }

    void Free(void *ptr) {
        free(ptr);
    }
};

class MultiDeviceAllocator {
    struct Meta {
        size_t size;
        int device;
    };
    std::map<int, AllocatorPtr> device_allocator_;
    std::map<void *, Meta> meta_;

public:
    using F_AllocatorFactory = std::function<AllocatorPtr(int)>;
private:
    F_AllocatorFactory allocator_factory_;
public:

    MultiDeviceAllocator(F_AllocatorFactory allocator_factory) :
            allocator_factory_(allocator_factory) {}

    AllocatorPtr GetAllocatorByDevice(int device) {
        auto it = device_allocator_.find(device);
        if (it == device_allocator_.end()) {
            it = device_allocator_.emplace(device, AllocatorPtr(allocator_factory_(device))).first;
        }
        return it->second;
    }

    void *Alloc(size_t size, int device = 0) {
        void *ptr = GetAllocatorByDevice(device)->Alloc(size);
        meta_.emplace(ptr, Meta{size, device});
        return ptr;
    }

    void Free(void *ptr) {
        auto it = meta_.find(ptr);
        if (it == meta_.end()) {
            std::cerr << "free error : " << ptr << std::endl;
            abort();
        }
        int device = it->second.device;
        GetAllocatorByDevice(device)->Free(ptr);
        meta_.erase(it);
    }

};

class UnifiedAllocator : public AllocatorBase {
private:
    int device_;
    MultiDeviceAllocator allocator_;
public:
    UnifiedAllocator();

    void Init();

    void SetDevice(int device_id);

    int GetDevice();

    void *Alloc(size_t size) override;

    void Free(void *ptr) override;
};

template<class Impl>
class AllocatorMetric {
    struct Meta {
        size_t size;
    };
    int device_;
    size_t maxsize_;
    size_t allocated_;
    std::map<void *, Meta> meta_;
    Impl impl_;

public:
    AllocatorMetric(int device, size_t maxsize) :
            device_(device),
            maxsize_(maxsize),
            allocated_(0),
            impl_(device) {}

    void *Alloc(size_t size) {
        allocated_ += size;
        void *ptr = impl_.Alloc(size);
        meta_.emplace(ptr, Meta{size});
        return ptr;
    }

    void Free(void *ptr) {
        auto it = meta_.find(ptr);
        if (it == meta_.end()) {
            return;
        }
        allocated_ -= it->second.size;
        impl_.Free(ptr);
    }

    int GetDevice() {
        return device_;
    }

};

class DummyAllocator : public AllocatorBase {
    char *ptr;

public:
    DummyAllocator(int device) : ptr((char *) 0x1) {

    }

    virtual void *Alloc(size_t size) override {
        std::cout << "Alloc " << size << " = " << (void *) ptr << std::endl;
        return ptr++;
    }

    virtual void Free(void *ptr) override {
        std::cout << "Free " << ptr << std::endl;
    }

};

class CudaAllocator : public AllocatorBase {
protected:
    int device_;
public:
    CudaAllocator(int device) : device_(device) {
    }

    virtual void *Alloc(size_t size) override {
        void *ptr;
        if (device_ < 0) {
            CUDA_CALL(cudaMallocHost, &ptr, size);
        } else {
            CUDA_CALL(cudaSetDevice, device_);
            CUDA_CALL(cudaMalloc, &ptr, size);
        }
        return ptr;
    }

    virtual void Free(void *ptr) override {
        if (device_ < 0) {
            CUDA_CALL(cudaFreeHost, ptr);
        } else {
            CUDA_CALL(cudaSetDevice, device_);
            CUDA_CALL(cudaFree, ptr);
        }
    }

    int DeviceId() const { return device_; }
};

class CudaPreAllocator : public CudaAllocator {
    size_t size_;
    size_t allocated_;
    size_t align_;
    void *ptr_;
    std::map<off_t, size_t> m_;
public:
    CudaPreAllocator(int device, size_t pre_alloc_bytes, size_t align = 4096) :
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

    ~CudaPreAllocator() {
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

    void *Alloc(size_t size) override {
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
        return (char *) ptr_ + off;
    }

    void Free(void *ptr) override {
        off_t off = (char *) ptr - (char *) ptr_;
        auto it = m_.find(off);
        if (it == m_.end()) {
            std::ostringstream os;
            os << "CudaPreAllocator :: Free pointer not found ptr=" << ptr << " off = " << off;
            throw std::runtime_error(os.str().c_str());
        }
        allocated_ -= it->second;
        m_.erase(it);
    }
};

class Allocator : public MultiDeviceAllocator {
    static AllocatorPtr factory(int device) { return AllocatorPtr(new CudaAllocator(device)); }

public:
    Allocator() : MultiDeviceAllocator(factory) {
    }
};

class PreAllocator : public MultiDeviceAllocator {
//    static AllocatorBase *factory(int device) {
//
//    }

public:
    PreAllocator(size_t sz, size_t align = 64) : MultiDeviceAllocator(
            [sz, align](int device) -> AllocatorPtr {
                if (device >= 0)
                    return AllocatorPtr(new CudaPreAllocator(device, sz, align));
                else
                    return AllocatorPtr(new CudaAllocator(device));
            }) {

    }
};

#endif //LDA_ALLOCATOR_H
