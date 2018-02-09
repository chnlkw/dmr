//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_DATA_H
#define DMR_DATA_H

#include "Array.h"

class DataBase {
protected:
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

//    void *data_ = nullptr;

public:

    DataBase() {

    }

    ArrayBasePtr ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr Read(DevicePtr dev) {
        Wait();
        ArrayBasePtr ret = last_state_.ReadAt(dev, 0);
        CUDA_CALL(cudaStreamSynchronize, 0);
        return ret;
    }

    ArrayBasePtr Write(DevicePtr dev, size_t bytes) {
        Wait();
        ArrayBasePtr ret = last_state_.WriteAt(dev, 0, false, bytes);
        CUDA_CALL(cudaStreamSynchronize, 0);
        return ret;
    }

    ArrayBasePtr Write(DevicePtr dev) { return Write(dev, last_state_.bytes); }

    ArrayBasePtr WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes);

    ArrayBasePtr WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr ReadWrite(DevicePtr dev) {
        ArrayBasePtr ret = last_state_.WriteAt(dev, 0, true, last_state_.bytes);
        CUDA_CALL(cudaStreamSynchronize, 0);
        return ret;
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


    Data(size_t count, DevicePtr device = Device::Current()) {
//        replicas_[device] = CreateArrayAt(device, count_);
        Write(device, count * sizeof(T));
    }

    Data(const std::vector<T> &vec, DevicePtr device = Device::Current()) {
        size_t bytes = vec.size() * sizeof(T);
        void *ptr = Write(device, bytes)->data();
        DataCopy(ptr, device->Id(), vec.data(), -1, bytes);
    }

    ArrayPtr<T> Read(DevicePtr dev = Device::Current()) {
        return std::static_pointer_cast<Array<T>>(DataBase::Read(dev));
    }

    ArrayPtr<T> ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
        return std::static_pointer_cast<Array<T>>(DataBase::ReadAsync(task, dev, stream));
    }

    ArrayPtr<T> WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
        return std::static_pointer_cast<Array<T>>(DataBase::WriteAsync(task, dev, stream, bytes));
    }

    ArrayPtr<T> WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
        return std::static_pointer_cast<Array<T>>(DataBase::WriteAsync(task, dev, stream));
    }

    ArrayPtr<T> ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
        return std::static_pointer_cast<Array<T>>(DataBase::ReadWriteAsync(task, dev, stream));
    }

    ArrayPtr<T> Write(DevicePtr dev, size_t bytes) {
        return std::static_pointer_cast<Array<T>>(DataBase::Write(dev, bytes));
    }

    ArrayPtr<T> Write(DevicePtr dev = Device::Current()) {
        return std::static_pointer_cast<Array<T>>(DataBase::Write(dev));
    }

//    bool HeldBy(DevicePtr device) const {
//        return replicas_.count(device);
//    }

//    void CopyTo(DevicePtr device) const {
//        if (HeldBy(device))
//            return;
//        assert(replicas_.size() > 0);
//        ArrayPtr<T> from = replicas_.begin()->second;
//
//        ArrayPtr<T> to;
//        if (invalids_.count(device)) {
//            to = invalids_[device];
//            invalids_.erase(device);
//        } else {
//            to = CreateArrayAt(device, count_);
//        }
//        to->CopyFrom(*from);
//        replicas_[device] = to;
//        SetPointers(device);
//    }

//    void MoveTo(DevicePtr device) {
//        if (HeldBy(device))
//            return;
//        CopyTo(device);
//        InvalidOthers(device);
//        SetPointers(device);
//    }

//    void Use(DevicePtr device = Device::Current()) {
//        GetFrom(device);
//        InvalidOthers(device);
//        SetPointers(device);
//    }

//    ArrayPtr<T> GetFrom(DevicePtr device) {
//        if (replicas_.size() > 0) {
//            CopyTo(device);
//        } else {
//            replicas_[device] = CreateArrayAt(device, count_);
//        }
//        InvalidOthers(device);
//        return replicas_[device];
//    }

//    auto GetFrom(DevicePtr device) const {
//        CopyTo(device);
//        return std::const_pointer_cast<const Array<T>>(replicas_[device]);
//    }

//    void CleanInvalids() {
//        invalids_.clear();
//    }

//    DevicePtr DeviceCurrent() const {
//        printf("DeviceCurrent = %d\n", cur_device_->Id());
//        return cur_device_;
//    }

    size_t size() const {
        return last_state_.bytes / sizeof(T);
    }

//    const T *begin() const { return (T *) data_; }

//    T *begin() { return (T *) data_; }

//    const T *end() const { return begin() + size(); }

//    T *end() { return begin() + size(); }

//    T &operator[](size_t i) { return begin()[i]; }

//    const T &operator[](size_t i) const { return begin()[i]; }

    std::string ToString() const {
        std::ostringstream os;
//        os << "Data(" << "count=" << size() << ", dev=" << "[" << begin() << "," << end() << "])";
        os << "Data(" << "count=" << size();
        return os.str();
    }

private:
//
//    void SetPointers(DevicePtr device) const {
////        beg_ = (T *) GetFrom(device)->data();
//        beg_ = last_state_.replicas[device]->data();
//        end_ = beg_ + count_;
//        cur_device_ = device;
//    }
//
//    void InvalidOthers(DevicePtr device) {
//        for (auto it = replicas_.begin(); it != replicas_.end();) {
//            if (it->first != device) {
//                invalids_.emplace(*it);
//                it = replicas_.erase(it);
//            } else {
//                ++it;
//            }
//        }
//    }
//
//    static auto CreateArrayAt(DevicePtr device, size_t count) {
//        return ArrayPtr<T>(new Array<T>(device->GetAllocator(), device->Id(), count));
//    }

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
