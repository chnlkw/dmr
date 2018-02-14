//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_DATA_H
#define DMR_DATA_H

#include "Array.h"

class DataBase {
protected:
    struct State {
        size_t bytes = 0;
        std::map<DevicePtr, ArrayBasePtr> replicas;
        std::map<DevicePtr, ArrayBasePtr> invalids;

        ArrayBasePtr ReadAt(const DevicePtr &dev, cudaStream_t stream);

        ArrayBasePtr WriteAt(const DevicePtr &dev, cudaStream_t stream, bool keep_old, size_t cur_bytes);
    };

    mutable State last_state_;
    mutable std::deque<std::weak_ptr<TaskBase>> tasks_scheduled_;

    friend class Engine;

    void AddTask(const TaskPtr &t) {
        tasks_scheduled_.push_back(t);
    }

public:

    DataBase() = default;

    DataBase(const DataBase &) = delete;

    size_t Bytes() const {
        return last_state_.bytes;
    }

    size_t NumTasks() const {
        return tasks_scheduled_.size();
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
class Data : public std::shared_ptr<DataBase> {
private:
    mutable T *data_ = nullptr;

    //add policy

public:
    Data() : std::shared_ptr<DataBase>(new DataBase()) {
    }

    explicit Data(size_t count, DevicePtr device = Device::Current()) : std::shared_ptr<DataBase>(new DataBase()) {
        Write(device, count * sizeof(T));
    }

    explicit Data(const std::vector<T> &vec, DevicePtr device = Device::Current()) : std::shared_ptr<DataBase>(new DataBase()) {
        size_t bytes = vec.size() * sizeof(T);
//        void *ptr = get()->Write(device, bytes)->data();
        Write(device, bytes);
        if (bytes > 0)
            DataCopy(data(), device->Id(), vec.data(), -1, bytes);
    }

    using value_type = T;

    const Array<T> &Read(DevicePtr dev = Device::Current()) const {
        const Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->Read(dev));
        data_ = (T *) ret.data();
        return ret;
    }

    const Array<T> &ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) const {
        data_ = nullptr;
        return *std::static_pointer_cast<Array<T>>(get()->ReadAsync(task, dev, stream));
    }

    Array<T> &WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
        data_ = nullptr;
        return *std::static_pointer_cast<Array<T>>(get()->WriteAsync(task, dev, stream, bytes));
    }

    Array<T> &WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
        data_ = nullptr;
        return *std::static_pointer_cast<Array<T>>(get()->WriteAsync(task, dev, stream));
    }

    Array<T> &ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
        data_ = nullptr;
        return *std::static_pointer_cast<Array<T>>(get()->ReadWriteAsync(task, dev, stream));
    }

    Array<T> &Write(DevicePtr dev, size_t bytes) {
        Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->Write(dev, bytes));
        data_ = ret.data();
        return ret;
    }

    Array<T> &Write(DevicePtr dev = Device::Current()) {
        Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->Write(dev));
        data_ = ret.data();
        return ret;
    }

    size_t size() const {
        return get()->Bytes() / sizeof(T);
    }

    T *data() { return data_; }

    const T *data() const { return data_; }

    const T &operator[](ssize_t idx) const { return data()[idx]; }

    T &operator[](ssize_t idx) { return data()[idx]; }

    std::string ToString() const {
        std::ostringstream os;
        const T *a = Read(Device::CpuDevice()).data();
        os << "Data(" << "ptr=" << a << " count=" << size() << ":";
        for (size_t i = 0; i < size(); i++)
            os << a[i] << ',';
        os << ")";

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
    os << "vector(" << v.size() << " : ";
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
