//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_TASK_H
#define DMR_TASK_H

#include <memory>
#include <set>
#include <list>
#include <queue>

#include "defs.h"
#include "Data.h"

class TaskBase {
    TaskBase(const TaskBase &) = delete;

    std::vector<DataBasePtr> inputs_;
    std::vector<DataBasePtr> outputs_;
    std::vector<DevicePtr> device_prefered_;
    std::set<WorkerPtr> worker_prefered_;

    friend class Engine;

public:
    virtual ~TaskBase() {}

    template<class Worker>
    void Run(Worker *t) {
        RunWorker(t);
    }

    const std::vector<DataBasePtr> &GetInputs() const {
        return inputs_;
    }

    const std::vector<DataBasePtr> &GetOutputs() const {
        return outputs_;
    }

    TaskBase &Prefer(WorkerPtr w) {
        worker_prefered_.insert(w);
        return *this;
    }

    virtual void Run(CPUWorker *) { throw std::runtime_error("not implemented on CPUWorker"); };

    virtual void Run(GPUWorker *) { throw std::runtime_error("not implemented on GPUWorker"); };

protected:

    TaskBase() {}

    void AddInput(DataBasePtr data) {
        inputs_.push_back(data);
    }

    void AddOutput(DataBasePtr data) {
        outputs_.push_back(data);
    }

};

#endif //DMR_TASK_H
