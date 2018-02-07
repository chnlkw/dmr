//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_DATA_H
#define DMR_DATA_H

#include "Array.h"

class DataBase {
    struct State {
        std::weak_ptr<TaskBase> task;
        size_t bytes = 0;
        std::map<DevicePtr, ArrayBasePtr> replicas;
        std::map<DevicePtr, ArrayBasePtr> invalids;

        ArrayBasePtr ReadAt(const DevicePtr &dev, cudaStream_t stream);

        ArrayBasePtr
        WriteAt(const DevicePtr &dev, cudaStream_t stream, bool keep_old, size_t cur_bytes);
    };

    mutable State last_state_;
    mutable std::deque<State> states_;

    void *data_ = nullptr;

public:

    explicit DataBase(size_t bytes = 0) {
        last_state_.bytes = bytes;
    }

    void ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    void *Read(DevicePtr dev) {
        Wait();
        data_ = last_state_.ReadAt(dev, 0)->data();
        CUDA_CALL(cudaStreamSynchronize, 0);
        return data_;
    }

    void *Write(DevicePtr dev, size_t bytes) {
        Wait();
        data_ = last_state_.WriteAt(dev, 0, false, bytes)->data();
        CUDA_CALL(cudaStreamSynchronize, 0);
        return data_;
    }

    void *Write(DevicePtr dev) { return Write(dev, last_state_.bytes); }

    void WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes_);

    void WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    void ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    void *ReadWrite(DevicePtr dev) {
        data_ = last_state_.WriteAt(dev, 0, true, last_state_.bytes)->data();
        CUDA_CALL(cudaStreamSynchronize, 0);
        return data_;
    }

    void Wait();
};

template<class T>
class Data : public DataBase {
private:
//    size_t count_;
//    mutable DevicePtr cur_device_;
//    mutable DevicePtr master_device_;
//    mutable T *beg_, *end_;
//    mutable std::map<DevicePtr, ArrayPtr<T>> replicas_;
//    mutable std::map<DevicePtr, ArrayPtr<T>> invalids_;

    //add policy

public:

    using value_type = T;

    explicit Data() {}


    Data(size_t count, DevicePtr device = Device::Current()) :
            count_(count), beg_(nullptr), end_(nullptr) {
//        replicas_[device] = CreateArrayAt(device, count_);
        last_state_.WriteAt(device, count);
        SetPointers(device);
    }

    Data(const std::vector<T> &vec) :
            count_(vec.size()) {
        DevicePtr device = Device::Current();
        ArrayPtr<T> array(new Array<T>(vec));
        replicas_.emplace(device, array);
        SetPointers(device);
    }

    bool HeldBy(DevicePtr device) const {
        return replicas_.count(device);
    }

    void CopyTo(DevicePtr device) const {
        if (HeldBy(device))
            return;
        assert(replicas_.size() > 0);
        ArrayPtr<T> from = replicas_.begin()->second;

        ArrayPtr<T> to;
        if (invalids_.count(device)) {
            to = invalids_[device];
            invalids_.erase(device);
        } else {
            to = CreateArrayAt(device, count_);
        }
        to->CopyFrom(*from);
        replicas_[device] = to;
        SetPointers(device);
    }

    void MoveTo(DevicePtr device) {
        if (HeldBy(device))
            return;
        CopyTo(device);
        InvalidOthers(device);
        SetPointers(device);
    }

    void Use(DevicePtr device = Device::Current()) {
        GetFrom(device);
        InvalidOthers(device);
        SetPointers(device);
    }

    ArrayPtr<T> GetFrom(DevicePtr device) {
        if (replicas_.size() > 0) {
            CopyTo(device);
        } else {
            replicas_[device] = CreateArrayAt(device, count_);
        }
        InvalidOthers(device);
        return replicas_[device];
    }

    auto GetFrom(DevicePtr device) const {
        CopyTo(device);
        return std::const_pointer_cast<const Array<T>>(replicas_[device]);
    }

    void CleanInvalids() {
        invalids_.clear();
    }

    DevicePtr DeviceCurrent() const {
//        printf("DeviceCurrent = %d\n", cur_device_->Id());
        return cur_device_;
    }

    size_t size() const {
        return count_;
    }

    const T *begin() const { return beg_; }

    T *begin() { return beg_; }

    const T *end() const { return end_; }

    T *end() { return end_; }

    T &operator[](size_t i) { return beg_[i]; }

    const T &operator[](size_t i) const { return beg_[i]; }

    std::string ToString() const {
        std::ostringstream os;
        os << "Data(" << "count=" << count_ << ", dev=" << cur_device_->Id() << "[" << beg_ << "," << end() << "])";
        return os.str();
    }

private:

    void SetPointers(DevicePtr device) const {
//        beg_ = (T *) GetFrom(device)->data();
        beg_ = last_state_.replicas[device]->data();
        end_ = beg_ + count_;
        cur_device_ = device;
    }

    void InvalidOthers(DevicePtr device) {
        for (auto it = replicas_.begin(); it != replicas_.end();) {
            if (it->first != device) {
                invalids_.emplace(*it);
                it = replicas_.erase(it);
            } else {
                ++it;
            }
        }
    }

    static auto CreateArrayAt(DevicePtr device, size_t count) {
        return ArrayPtr<T>(new Array<T>(device->GetAllocator(), device->Id(), count));
    }
};

namespace std {
template<class T>
std::string to_string(const Data<T> &v) { return v.ToString(); }

template<class T>
std::string to_string(const std::vector<T> &v) {
    std::ostringstream os;
    os << "(" << v.size() << " : ";
    for (auto x : v) os << x << ",";
    os << ") ";
    return os.str();
}
}

struct data_constructor_t {
    template<class T, class ...Args>
    static Data<T> Construct(Args &&... args) {
        return {std::forward<Args>(args)...};
    }
};

#endif //DMR_DATA_H
