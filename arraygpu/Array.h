//
// Created by chnlkw on 11/21/17.
//

#ifndef LDA_ARRAY_H_H
#define LDA_ARRAY_H_H

#include "defs.h"
#include "Allocator.h"
#include "Device.h"
#include "DataCopy.h"

class ArrayBase {
protected:
    AllocatorPtr allocator_;
    int device_;
    size_t bytes_;
    void *ptr_;
    bool owned_;

    void allocate(size_t bytes) {
        bytes_ = bytes;
        if (bytes) {
            owned_ = true;
            ptr_ = allocator_->Alloc(bytes);
        } else {
            owned_ = false;
            ptr_ = nullptr;
        }
    }

public:

    ArrayBase(size_t bytes)
            : allocator_(Device::Current()->GetAllocator()),
              device_(Device::Current()->Id()) {
        allocate(bytes);
    }

    ArrayBase(AllocatorPtr allocator, int device, size_t bytes)
            : allocator_(allocator),
              device_(device) {
        allocate(bytes);
    }

    ArrayBase(const ArrayBase &that) :
            allocator_(Device::Current()->GetAllocator()),
            device_(Device::Current()->Id()) {
        allocate(that.bytes_);
        CopyFrom(that);
    }

    ArrayBase(void *ptr, size_t bytes) : //copy from cpu ptr
            allocator_(Device::Current()->GetAllocator()),
            device_(Device::Current()->Id()) {
        allocate(bytes);
        DataCopy(this->ptr_, this->device_, ptr, -1, this->bytes_);
    }

    ArrayBase(ArrayBase &&that) :
            allocator_(that.allocator_),
            bytes_(that.bytes_),
            ptr_(that.ptr_),
            device_(that.device_),
            owned_(that.owned_) {
        that.ptr_ = nullptr;
        that.owned_ = false;
    }

    ArrayBase(const ArrayBase &that, size_t off, size_t bytes)
            : allocator_(nullptr),
              bytes_(bytes),
              ptr_((char *) that.ptr_ + off),
              device_(that.device_),
              owned_(false) {
    }

    ~ArrayBase() {
        if (owned_) {
            assert(ptr_ != nullptr);
            allocator_->Free(ptr_);
            ptr_ = nullptr;
            owned_ = false;
        }
    }

    ArrayBase Renew(size_t bytes) const {
        return {allocator_, device_, bytes};
    }

    void CopyFrom(const ArrayBase &that) {
        assert(this->bytes_ == that.bytes_);
        DataCopy(this->ptr_, this->device_, that.ptr_, that.device_, this->bytes_);
    }

    void CopyFromAsync(const ArrayBase &that, cudaStream_t stream) {
        assert(this->bytes_ == that.bytes_);
        DataCopyAsync(this->ptr_, this->device_, that.ptr_, that.device_, this->bytes_, stream);
    }

    int GetDevice() const {
        return device_;
    }

    size_t GetBytes() const {
        return bytes_;
    }
};

template<class T>
class Array : public ArrayBase {
    size_t count_;
public:

    using value_type = T;

    explicit Array(size_t count = 0) :
            ArrayBase(count * sizeof(T)),
            count_(count) {
    }

    Array(AllocatorPtr allocator, int device, size_t count = 0)
            : ArrayBase(allocator, device, count * sizeof(T)),
              count_(count) {
    }

//    Array(AllocatorBase *allocator, size_t count) : // need allocated
//            ArrayBase(allocator, count * sizeof(T)),
//            count_(count) {
//    }

//    Array(MultiDeviceAllocator &allocator, T *ptr, size_t count, int device) : // not allocated
//            ArrayBase(allocator, ptr, count * sizeof(T), device),
//            count_(count) {
//    }

    Array(Array<T> &&that) :
            ArrayBase(std::move(that)),
            count_(that.count_) {
        that.count_ = 0;
    }

//    Array &operator=(Array<T> &&that) {
//        count_ = that.count_;
//        that.count_ = 0;
//        ArrayBase::operator=(std::move(that));
//        return *this;
//    }

    Array(const std::vector<T> &that) :
            ArrayBase((void *) that.data(), that.size() * sizeof(T)),
            count_(that.size()) {
    }

    Array(const Array<T> &that) :
            ArrayBase(that),
            count_(that.count_) {
    }

    T *data() {
        return reinterpret_cast<T *>(ptr_);
    }

    const T *data() const {
        return reinterpret_cast<const T *>(ptr_);
    }

    T *begin() {
        return reinterpret_cast<T *>(ptr_);
    }

    const T *begin() const {
        return reinterpret_cast<const T *>(ptr_);
    }

    T *end() {
        return begin() + count_;
    }

    const T *end() const {
        return begin() + count_;
    }

    size_t size() const {
        return count_;
    }

    Array<T> CopyTo(int device) {
        Array<T> that(allocator_, count_, device);
        that.CopyFrom(*this);
        return that;
    }

    Array<T> Renew(size_t count) const {
        return {allocator_, device_, count};
    }

    Array<T> Slice(size_t beg, size_t end) {
        assert(beg < end && end <= count_);
        T *ptr = this->data() + beg;
        size_t count = end - beg;
        return Array<T>(allocator_, ptr, count, device_);
    }

//    Array<T> &operator=(const Array<T> &that) {
//        CopyFrom(that);
//    }

    const T &operator[](ssize_t idx) const {
        return data()[idx];
    }

    T &operator[](ssize_t idx) {
        return data()[idx];
    }

    Array<T> operator[](std::pair<ssize_t, ssize_t> range) {
        return Slice(range.first, range.second);
    }
};

struct array_constructor_t {
    template<class T, class ...Args>
    static Array<T> Construct(Args &&... args) {
        return {std::forward<Args>(args)...};
    }
};

#endif //LDA_ARRAY_H_H
