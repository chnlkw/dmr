//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"
#include "Data.h"
#include "Engine.h"

#define LG(x) CLOG(x, "Worker")

CPUWorker::CPUWorker(CPUDevice *cpu) : WorkerBase(cpu) {}

std::vector<TaskPtr> CPUWorker::GetCompleteTasks() {
    std::vector<TaskPtr> ret;
    for (TaskPtr t : tasks_) {
        CPUTask *cputask = dynamic_cast<CPUTask *>(t.get());
        if (!cputask)
            cputask = t->GetCPUTask();
        CLOG(INFO, "Worker") << *this << " Run Task " << *t;
        if (cputask) {
            t->PrepareData(device_, 0);
            CUDA_CALL(cudaStreamSynchronize, 0);

            (*cputask)(this);
        } else
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
    auto gpu = dynamic_cast<GPUDevice *>(device_);
    assert(gpu);
    CUDA_CALL(cudaSetDevice, gpu->Id());
    cudaEvent_t e;
    if (!events_unused_.empty()) {
        e = events_unused_.back();
        events_unused_.pop_back();
    } else {
        CUDA_CALL(cudaEventCreate, &e);
    }

    auto gputask = dynamic_cast<GPUTask *>(t.get());
    if (!gputask)
        gputask = t->GetGPUTask();
    CLOG(INFO, "Worker") << stream_ << " " << *this << " Run Task " << t->Name() << " gputask_ptr " << gputask;
    if (gputask) {
        for (auto &m : t->GetMetas()) {
            if (m.is_read_only) {
                m.data->ReadAsync(t, device_, stream_);
            } else {
                m.data->WriteAsync(t, device_, stream_);
            }
        }
        (*gputask)(this);
    } else
        t->Run(this);
    CUDA_CALL(cudaEventRecord, e, stream_);
    queue_.emplace_back(e, t);
}
