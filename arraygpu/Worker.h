//
// Created by chnlkw on 2/1/18.
//

#ifndef DMR_WORKER_H
#define DMR_WORKER_H

#include "defs.h"
#include "cuda_utils.h"
#include <deque>

class WorkerBase : public el::Loggable {
protected:
    DevicePtr device_;
public:
    WorkerBase(DevicePtr d) : device_(d) {
    }

    virtual void RunTask(TaskPtr t) = 0;

    virtual std::vector<TaskPtr> GetCompleteTasks() = 0;

    virtual bool Empty() const = 0;

    DevicePtr Device() const {
        return device_;
    }
    virtual void log(el::base::type::ostream_t &os) const;
};


class CPUWorker : public WorkerBase {
    std::deque<TaskPtr> tasks_;
public:
    CPUWorker();

    void RunTask(TaskPtr t) override {
        tasks_.push_back(t);
    }

    bool Empty() const override { return tasks_.empty(); }

    std::vector<TaskPtr> GetCompleteTasks() override;

};

class GPUWorker : public WorkerBase {
    cudaStream_t stream_;

    std::vector<cudaEvent_t> events_unused_;
    std::deque<std::pair<cudaEvent_t, TaskPtr>> queue_;

public:
    GPUWorker(GPUDevice* gpu);

    virtual bool Empty() const override {
        return queue_.empty();
    }

    cudaStream_t Stream() const {
        return stream_;
    }

private:

    void RunTask(TaskPtr t) override;

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
