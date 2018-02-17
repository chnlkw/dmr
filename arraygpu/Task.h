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

class TaskBase : public std::enable_shared_from_this<TaskBase>, public el::Loggable {
    TaskBase(const TaskBase &) = delete;

    std::vector<DataBasePtr> inputs_;
    std::vector<DataBasePtr> outputs_;
    std::vector<DevicePtr> device_prefered_;
    std::set<WorkerPtr> worker_prefered_;
    bool finished = false;
    Engine &engine_;

    friend class Engine;

    std::string name_;

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

    void WaitFinish();

    virtual std::string Name() const { return name_; }

    virtual void log(el::base::type::ostream_t &os) const {
        os << "[Task]" << Name();
        if (finished) os << " Finished";
    }


protected:
    TaskBase(Engine &engine, std::string name = "Task") :
            engine_(engine), name_(name) {}

    void AddInput(DataBasePtr data) {
        inputs_.push_back(data);
    }

    void AddOutput(DataBasePtr data) {
        outputs_.push_back(data);
    }

    void Finish() {
        finished = true;
    }

};


#endif //DMR_TASK_H
