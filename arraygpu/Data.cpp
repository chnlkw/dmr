//
// Created by chnlkw on 2/6/18.
//

#include "Data.h"

ArrayBasePtr DataBase::State::ReadAt(const DevicePtr &dev, cudaStream_t stream) {
    if (replicas.count(dev) == 0) {
        ArrayBasePtr arr;
        assert(!replicas.empty());
        ArrayBasePtr from = replicas.begin()->second;
        if (invalids.count(dev)) {
            arr = invalids[dev];
            invalids.erase(dev);
        } else {
            arr = std::make_shared<ArrayBase>(dev->GetAllocator(), dev->Id(), bytes);
        }
        arr->CopyFromAsync(*from, stream);
        replicas[dev] = arr;
    }
    assert(replicas[dev]->GetBytes() == bytes);
    return replicas[dev];
}

ArrayBasePtr DataBase::State::WriteAt(const DevicePtr &dev, cudaStream_t stream, bool keep_old, size_t cur_bytes) {
    assert(cur_bytes > 0);
    bytes = cur_bytes;
    // invalid other replicas
    for (auto it = replicas.begin(); it != replicas.end();) {
        if (it->first != dev) {
            invalids.emplace(*it);
            it = replicas.erase(it);
        } else {
            ++it;
        }
    }
    if (replicas.count(dev)) {
        if (replicas[dev]->GetBytes() < bytes) {
            auto arr = std::make_shared<ArrayBase>(dev->GetAllocator(), dev->Id(), bytes);
            if (keep_old)
                arr->CopyFromAsync(*replicas[dev], stream, false);
            replicas[dev] = arr;
        }
    } else {
        replicas[dev] = std::make_shared<ArrayBase>(dev->GetAllocator(), dev->Id(), bytes);
    }
    assert(replicas[dev]->GetBytes() >= bytes);
    return replicas[dev];
}

void DataBase::ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
    last_state_.ReadAt(dev, stream);
    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, true, last_state_.bytes);
    last_state_.task = task;
    states_.push_back(last_state_);
    data_ = arr->data();
}

void DataBase::ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
    ArrayBasePtr arr = last_state_.ReadAt(dev, stream);
    last_state_.task = task;
    states_.push_back(last_state_);
    data_ = arr->data();
}

void DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
    assert(bytes > 0);
    last_state_.task = task;
    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, false, bytes);
    states_.push_back(last_state_);
    data_ = arr->data();
}

void DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
    WriteAsync(task, dev, stream, last_state_.bytes);
}

