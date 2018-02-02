//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_DATA_H
#define DMR_DATA_H

#include "Array.h"

class DataBase {
    std::set<TaskBase *> depend_tasks_;
public:
    const std::set<TaskBase *> &DependTasks() const {
        return depend_tasks_;
    }

    void UpdateByTask(TaskBase* t) {
        depend_tasks_.clear();
        depend_tasks_.insert(t);
    }
};

template<class T>
class Data : public DataBase {
private:
    size_t count_;
    mutable DevicePtr cur_device_;
    mutable T *beg_, *end_;
    mutable std::map<DevicePtr, ArrayPtr<T>> replicas_;
    mutable std::map<DevicePtr, ArrayPtr<T>> invalids_;

public:

    using value_type = T;

    explicit Data() :
            count_(0), beg_(nullptr), end_(nullptr) {
    }

    Data(size_t count, DevicePtr device = Device::Current()) :
            count_(count), beg_(nullptr), end_(nullptr) {
        replicas_[device] = CreateArrayAt(device, count_);
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
        beg_ = (T *) GetFrom(device)->data();
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
