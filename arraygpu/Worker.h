//
// Created by chnlkw on 2/1/18.
//

#ifndef DMR_WORKER_H
#define DMR_WORKER_H

#include "Task.h"
#include "cuda_utils.h"

class WorkerBase {
public:
    virtual void RunTask(TaskPtr t) = 0;

    virtual std::vector<TaskPtr> GetCompleteTasks() = 0;

    virtual bool Empty() const = 0;

    virtual DevicePtr Device() const = 0;
};

class CPUWorker : public WorkerBase {
    std::deque<TaskPtr> tasks_;
public:
    CPUWorker() {}

    void RunTask(TaskPtr t) override {
        tasks_.push_back(t);
    }

    bool Empty() const override { return tasks_.empty(); }

    std::vector<TaskPtr> GetCompleteTasks() override {
        std::vector<TaskPtr> ret;
        for (TaskPtr t : tasks_) {
            t->Run(this);
            ret.push_back(t);
        }
        tasks_.clear();
        return ret;
    }

    DevicePtr Device() const override {
        return Device::CpuDevice();
    }

};

class GPUWorker : public WorkerBase {
    std::shared_ptr<GPUDevice> gpu_;
    cudaStream_t stream_;

    std::vector<cudaEvent_t> events_unused_;
    std::deque<std::pair<cudaEvent_t, TaskPtr>> queue_;

public:
    GPUWorker(std::shared_ptr<GPUDevice> gpu) :
            gpu_(gpu) {
        printf("gpu worker id = %d\n", gpu->Id());
        CUDA_CALL(cudaSetDevice, gpu->Id());
        CUDA_CALL(cudaStreamCreate, &stream_);
    }

    virtual bool Empty() const override {
        return queue_.empty();
    }

    cudaStream_t Stream() const {
        return stream_;
    }

    DevicePtr Device() const override {
        return gpu_;
    }

private:

    void RunTask(TaskPtr t) override {
        CUDA_CALL(cudaSetDevice, gpu_->Id());
        cudaEvent_t e;
        if (events_unused_.size() > 0) {
            e = events_unused_.back();
            events_unused_.pop_back();
        } else {
            CUDA_CALL(cudaEventCreate, &e);
        }

//        for (DataBasePtr d : t->GetInputs())
//            d->ReadAsync(t, gpu_, stream_);
//        for (DataBasePtr d : t->GetOutputs())
//            d->WriteAsync(t, gpu_, stream_);

        t->Run(this);
        CUDA_CALL(cudaEventRecord, e, stream_);
        queue_.emplace_back(e, t);
    }

    std::vector<TaskPtr> GetCompleteTasks() override {
#ifdef USE_CUDA
        std::vector<TaskPtr> ret;
        if (Empty())
            return ret;

        while (true) {
            TaskPtr t;
            cudaEvent_t e;
            std::tie(e, t) = queue_.front();
            cudaError_t err = cudaEventQuery(e);
            if (err == cudaSuccess) {
                queue_.pop_front();
                events_unused_.push_back(e);
                ret.push_back(t);
                break;
            } else if (err == cudaErrorNotReady) {
                continue;
            } else {
                CUDA_CHECK();
            }
        }
        return ret;
#else
        return {};
#endif
    }

};


#endif //DMR_WORKER_H
