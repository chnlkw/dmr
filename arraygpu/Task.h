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

//enum Flag {
//    Default = 0,
//    Shared = 1,
//    Exclusive = 2
//};


class TaskBase : public std::enable_shared_from_this<TaskBase>, public el::Loggable {
    TaskBase(const TaskBase &) = delete;

    struct Meta : public el::Loggable {
        DataBasePtr data;
        bool read_only = true;
        int priority = 0;

        Meta(DataBasePtr d, bool b, int p) :
                data(d),
                read_only(b),
                priority(p) {}

        bool operator<(const Meta &that) const {
            return priority > that.priority;
        }

        virtual void log(el::base::type::ostream_t &os) const {
            os << "[Meta] "
               << data << " "
               << (read_only ? "R " : "W ")
               << priority << ". ";
        }
    };

    std::vector<Meta> metas_;
    bool finished = false;
    Engine &engine_;

    friend class Engine;

    std::string name_;

public:
    virtual ~TaskBase();

//    template<class Worker>
//    void Run(Worker *t) {
//        RunWorker(t);
//    }

    const auto &GetMetas() {
        std::sort(metas_.begin(), metas_.end());
        return metas_;
    }
//    const std::vector<DataBasePtr> &GetInputs() const {
//        return inputs_;
//    }
//
//    const std::vector<DataBasePtr> &GetOutputs() const {
//        return outputs_;
//    }

//    TaskBase &Prefer(WorkerPtr w) {
//        worker_prefered_.insert(w);
//        return *this;
//    }

    virtual void Run(CPUWorker *) { throw std::runtime_error("not implemented on CPUWorker"); };

    virtual void Run(GPUWorker *) { throw std::runtime_error("not implemented on GPUWorker"); };

    void WaitFinish();

    virtual std::string Name() const { return name_; }

    virtual void log(el::base::type::ostream_t &os) const {
        os << "[Task]" << Name();
        if (finished) os << " Finished";
    }

    bool IsFinished() const {
        return finished;
    }

protected:
    TaskBase(Engine &engine, std::string name = "nonamed task") :
            engine_(engine), name_(name) {}

    void AddInput(DataBasePtr data, int priority = 1) {
        metas_.push_back(Meta{data, true, priority});
    }

    void AddOutput(DataBasePtr data, int priority = 2) {
        metas_.push_back(Meta{data, false, priority});
    }

    void Finish() {
        finished = true;
    }

};


#endif //DMR_TASK_H
