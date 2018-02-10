//
// Created by chnlkw on 11/28/17.
//

#include "Engine.h"

std::unique_ptr<Engine> Engine::engine;

void TaskBase::WaitFinish() {
    while (!finished) {
        if (!engine_.Tick()) {
            throw std::runtime_error("Task unfinished, while engine ends");
        }
    }
}

void DataBase::Wait() const {
    // Wait all tasks finish
    for (auto &s : states_) {
        if (auto t = s.task.lock()) {
            t->WaitFinish();
        }
    }
    states_.clear();
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

