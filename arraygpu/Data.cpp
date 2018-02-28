//
// Created by chnlkw on 2/6/18.
//

#include "Data.h"
#include <cassert>

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

ArrayBasePtr DataBase::ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
    last_state_.ReadAt(dev, stream);
    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, true, last_state_.bytes);
    return arr;
}

ArrayBasePtr DataBase::ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
//    tasks_writing_.clear();
//    tasks_reading_.push_back(task);
    ArrayBasePtr arr = last_state_.ReadAt(dev, stream);
    return arr;
}

ArrayBasePtr DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
//    tasks_reading_.clear();
//    tasks_writing_.push_back(task);
    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, false, bytes);
    return arr;
}

ArrayBasePtr DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
    return WriteAsync(task, dev, stream, last_state_.bytes);
}

ArrayBasePtr DataBase::Read(DevicePtr dev) const {
    Wait();
    ArrayBasePtr ret = last_state_.ReadAt(dev, 0);
    CUDA_CALL(cudaStreamSynchronize, 0);
    return ret;
}

ArrayBasePtr DataBase::Write(DevicePtr dev, size_t bytes) {
    Wait();
    ArrayBasePtr ret = last_state_.WriteAt(dev, 0, false, bytes);
    CUDA_CALL(cudaStreamSynchronize, 0);
    return ret;
}

ArrayBasePtr DataBase::ReadWrite(DevicePtr dev) {
    ArrayBasePtr ret = last_state_.WriteAt(dev, 0, true, last_state_.bytes);
    CUDA_CALL(cudaStreamSynchronize, 0);
    return ret;
}

const std::vector<std::weak_ptr<TaskBase>> &DataBase::RegisterTask(const TaskPtr &t, bool read_only) {
    tasks_scheduled_.push_back(t);
    if (read_only) {
        if (writing) {
            writing = false;
            last_reading_.clear();
        }
        last_reading_.push_back(t);
        return last_writing_;
    } else {
        if (!writing) {
            writing = true;
            last_writing_.clear();
        }
        last_writing_.push_back(t);
        return last_reading_;
    }
}

