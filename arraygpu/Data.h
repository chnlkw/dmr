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

    size_t NumStates() const {
        return states_.size();
    }

    ArrayBasePtr ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr Read(DevicePtr dev) const;

    ArrayBasePtr Write(DevicePtr dev, size_t bytes);

    ArrayBasePtr Write(DevicePtr dev) { return Write(dev, last_state_.bytes); }

    ArrayBasePtr WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes);

    ArrayBasePtr WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr ReadWrite(DevicePtr dev);

    void Wait() const;
};

template<class T>
class Data : public DataBase {
private:
    //add policy

public:

    using value_type = T;

    Data() {}

    Data(size_t count, DevicePtr device = Device::Current()) {
        Write(device, count * sizeof(T));
    }

    Data(const std::vector<T> &vec, DevicePtr device = Device::Current()) {
        size_t bytes = vec.size() * sizeof(T);
        void *ptr = Write(device, bytes)->data();
        DataCopy(ptr, device->Id(), vec.data(), -1, bytes);
    }

    ArrayPtr<T> Read(DevicePtr dev = Device::Current()) const {
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

    size_t size() const {
        return last_state_.bytes / sizeof(T);
    }

    std::string ToString() const {
        std::ostringstream os;
        os << "Data(" << "count=" << size();
        return os.str();
    }

private:
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
