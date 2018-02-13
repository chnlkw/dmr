//
// Created by chnlkw on 1/16/18.
//

#include "Array.h"

ArrayBase::ArrayBase(size_t bytes)
        : allocator_(Device::Current()->GetAllocator()),
          device_(Device::Current()->Id()) {
    Allocate(bytes);
}

ArrayBase::ArrayBase(const ArrayBase &that) :
        allocator_(Device::Current()->GetAllocator()),
        device_(Device::Current()->Id()) {
    Allocate(that.bytes_);
    CopyFrom(that);
}

ArrayBase::ArrayBase(void *ptr, size_t bytes) : //copy from cpu ptr
        allocator_(Device::Current()->GetAllocator()),
        device_(Device::Current()->Id()) {
    Allocate(bytes);
    DataCopy(this->ptr_, this->device_, ptr, -1, this->bytes_);
}

ArrayBase::ArrayBase(ArrayBase &&that) :
        allocator_(that.allocator_),
        bytes_(that.bytes_),
        ptr_(that.ptr_),
        device_(that.device_),
        owned_(that.owned_) {
    that.ptr_ = nullptr;
    that.owned_ = false;
}

ArrayBase::ArrayBase(const ArrayBase &that, size_t off, size_t bytes)
        : allocator_(nullptr),
          bytes_(bytes),
          ptr_((char *) that.ptr_ + off),
          device_(that.device_),
          owned_(false) {
}

ArrayBase::~ArrayBase() {
    Free();
}

void ArrayBase::Free() {
    if (owned_) {
        assert(ptr_ != nullptr);
        allocator_->Free(ptr_);
        ptr_ = nullptr;
        owned_ = false;
    }
    bytes_ = 0;
}

void ArrayBase::Allocate(size_t bytes) {
//    Free();
    bytes_ = bytes;
    if (bytes > 0) {
        owned_ = true;
        ptr_ = allocator_->Alloc(bytes);
    } else {
        owned_ = false;
        ptr_ = nullptr;
    }
//    printf("reallocate ptr %p bytes = %lu\n", ptr_, bytes);
}

void ArrayBase::CopyFrom(const ArrayBase &that, bool check_size_equal) {
    size_t bytes = std::min(this->bytes_, that.bytes_);
    printf("Copy %d -> %d\n", that.device_, this->device_, bytes);
    if (check_size_equal)
        assert(this->bytes_ == that.bytes_);
    DataCopy(this->ptr_, this->device_, that.ptr_, that.device_, bytes);
}

void ArrayBase::CopyFromAsync(const ArrayBase &that, cudaStream_t stream, bool check_size_equal) {
    size_t bytes = std::min(this->bytes_, that.bytes_);
    printf("CopyAsync %d -> %d\n", that.device_, this->device_, bytes);
    if (check_size_equal)
        assert(this->bytes_ == that.bytes_);
    DataCopyAsync(this->ptr_, this->device_, that.ptr_, that.device_, bytes, stream);
}

ArrayBase::ArrayBase(AllocatorPtr allocator, int device, size_t bytes)
        : allocator_(std::move(allocator)),
          device_(device) {
    Allocate(bytes);
}
