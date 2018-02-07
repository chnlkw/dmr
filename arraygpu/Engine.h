//
// Created by chnlkw on 11/28/17.
//

#ifndef LDA_ENGINE_H
#define LDA_ENGINE_H

#include "cuda_utils.h"
#include <deque>
#include <map>

#include "Worker.h"

class Engine {
    struct Node {
        size_t in_degree = 0;
        std::vector<TaskPtr> next_tasks_;
    };
    std::map<TaskPtr, Node> tasks_;

    struct DataNode {
        TaskPtr writer;
        std::vector<TaskPtr> readers;
    };
    std::map<DataBasePtr, DataNode> data_;
    std::set<WorkerPtr> workers_;
    std::vector<TaskPtr> ready_tasks_;

public:
    Engine(std::set<WorkerPtr> workers) :
            workers_(std::move(workers)) {
    }

    TaskBase &AddTask(TaskPtr task) {
        for (auto d : task->GetInputs()) {
            DataNode &data = data_[d];
            if (data.writer)
                AddEdge(data.writer, task);
            data.readers.push_back(task);
        }
        for (auto d : task->GetOutputs()) {
            DataNode &data = data_[d];
            if (data.writer && data.readers.empty()) {
                fprintf(stderr, "Data written twice but no readers between them\n");
            }
            for (auto r : data.readers)
                AddEdge(r, task);
            data.readers.clear();
            data.writer = task;
        }
        CheckTaskReady(task);
        return *task;
    }

    template<class Task, class... Args>
    TaskBase &AddTask(Args... args) {
        auto t = std::make_shared<Task>(*this, args...);
        return AddTask(t);
    };

    bool Tick() {
        std::cout << "Tick " << std::endl;
        size_t empty_workers_ = 0;
        for (auto w : workers_) {
            std::cout << "workers " << w.get() << std::endl;
            if (w->Empty()) {
                empty_workers_++;
                continue;
            }
            auto tasks = w->GetCompleteTasks();
            for (auto &t : tasks) {
                FinishTask(t);
            }
        }
        if (ready_tasks_.empty() && empty_workers_ == workers_.size()) {
            return false; // Finish
        }
        for (auto t : ready_tasks_) {
            ChooseWorker(t)->RunTask(t);
        }
        ready_tasks_.clear();
        return true;
    }

    WorkerPtr ChooseWorker(TaskPtr t) {
        if (t->worker_prefered_.size()) {
            return *(t->worker_prefered_.begin());
        } else {
            return *workers_.begin();
        }
    }

private:
    void AddEdge(TaskPtr src, TaskPtr dst) {
        tasks_[src].next_tasks_.push_back(dst);
        tasks_[dst].in_degree++;
    }

    void CheckTaskReady(TaskPtr task) {
        if (tasks_[task].in_degree == 0)
            ready_tasks_.push_back(task);
    }

    void FinishTask(TaskPtr task) {
        for (auto t : tasks_[task].next_tasks_) {
            --tasks_[t].in_degree;
            CheckTaskReady(t);
        }
        task->Finish();
        tasks_.erase(task);
    }
};

void TaskBase::WaitFinish() {
    while (!finished) {
        if (!engine_.Tick()) {
            throw std::runtime_error("Task unfinished, while engine ends");
        }
    }
}

void DataBase::Wait() {
    // Wait all tasks finish
    for (auto &s : states_) {
        if (auto t = s.task.lock()) {
            t->WaitFinish();
        }
    }
    states_.clear();
}

#if 0

struct TaskContext {
    int pipeline_id = -1;
    int device = -1;
    cudaStream_t stream = nullptr;
};

struct GpuTaskBase {
    TaskContext context;

    GpuTaskBase(int device) {
        context.device = device;
    }

    GpuTaskBase(const GpuTaskBase &) = delete;

    virtual void Tick() = 0;
};

template<class State, class F>
struct GpuTask : GpuTaskBase {
    State state_;
    F f_;

    GpuTask(int start_device, State state, F f) :
            GpuTaskBase(start_device), state_(state), f_(f) {
    }

    virtual void Tick() override {
        context.device = f_(state_, context);
    }

};

template<class State, class F>
std::unique_ptr<GpuTask<State, F>> make_gpu_task(int start_device, State state, F f) {
    return std::unique_ptr<GpuTask<State, F>>(new GpuTask<State, F>(start_device, state, f));
};


struct TaskEngine {
    int device_;
    cudaStream_t stream_;
    std::vector<cudaEvent_t> events_unused_;
    std::deque<std::pair<cudaEvent_t, int>> queue_;

    TaskEngine(int device, int capacity = -1) : device_(device) {
        cudaSetDevice(device_);
        cudaStreamCreate(&stream_);
    }

    TaskEngine(const TaskEngine &) = delete;

    TaskEngine(TaskEngine &&that) :
            device_(that.device_),
            stream_(that.stream_),
            events_unused_(std::move(that.events_unused_)),
            queue_(std::move(that.queue_)) {
        that.device_ = 0;
        that.stream_ = 0;
        that.events_unused_.clear();
        that.queue_.clear();
    }

    ~TaskEngine();

    void Push(int id);

    bool Empty() { return queue_.empty(); }

    int Pop();
    int Poll();
};

using GPUTaskPtr = std::unique_ptr<GpuTaskBase>;

class GpusPipeline {
    std::vector<TaskEngine> dev_engines_;
    std::map<int, GPUTaskPtr> tasks_;
    bool flag_empty;
    int seq;
    int cur_dev;

    GpusPipeline(const GpusPipeline &) = delete;

    void RunTask(int seq);

public:
    GpusPipeline(size_t num_devices);

    GpusPipeline(GpusPipeline &&that);

    void AddGpuTask(GPUTaskPtr ptr);

    bool Tick();

    void Run();
};

class GpusEngine {
    std::vector<GpusPipeline> gpus_pipelines_;
    size_t num_stream_;
    size_t cur_id_;

public:
    GpusEngine(size_t num_stream, size_t num_devices);

    void AddGpuTask(GPUTaskPtr ptr);

    void Run();
};

#endif

#endif //LDA_ENGINE_H
