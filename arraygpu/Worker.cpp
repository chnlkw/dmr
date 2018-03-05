//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"
#include "Engine.h"

CPUWorker::CPUWorker() : WorkerBase(Engine::GetCPUDevice()) {}

std::vector<TaskPtr> CPUWorker::GetCompleteTasks() {
    std::vector<TaskPtr> ret;
    for (TaskPtr t : tasks_) {
        t->Run(this);
        ret.push_back(t);
    }
    tasks_.clear();
    return ret;
}

GPUWorker::GPUWorker(GPUDevice *gpu) :
        WorkerBase(gpu) {
    CLOG(INFO, "Worker") << "Create GPU Worker with device = " << gpu->Id();
    CUDA_CALL(cudaSetDevice, gpu->Id());
    CUDA_CALL(cudaStreamCreate, &stream_);
}

void GPUWorker::RunTask(TaskPtr t) {
    GPUDevice &gpu = *static_cast<GPUDevice *>(device_);
    CUDA_CALL(cudaSetDevice, gpu.Id());
    cudaEvent_t e;
    if (events_unused_.size() > 0) {
        e = events_unused_.back();
        events_unused_.pop_back();
    } else {
        CUDA_CALL(cudaEventCreate, &e);
    }

    CLOG(INFO, "Worker") << stream_ << " " << *this << " Run Task " << t->Name();
    t->Run(this);
    CUDA_CALL(cudaEventRecord, e, stream_);
    queue_.emplace_back(e, t);
}
