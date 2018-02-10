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

    static std::unique_ptr<Engine> engine;

    Engine(std::set<WorkerPtr> workers) :
            workers_(std::move(workers)) {
    }

public:

    static void Create(std::set<WorkerPtr> workers) {
        engine.reset(new Engine(std::move(workers)));
    }

    static void Finish() {
        engine.reset();
    }

    static Engine &Get() {
        if (!engine)
            throw std::runtime_error("Engine::Create() must be called before ::Get()");
        return *engine;
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
            WorkerPtr w = ChooseWorker(t);
            w->RunTask(t);
        }
        ready_tasks_.clear();
        return true;
    }

    WorkerPtr ChooseWorker(TaskPtr t) {
        WorkerPtr w;
        if (t->worker_prefered_.size()) {
            w = *(t->worker_prefered_.begin());
        } else {
            w = *workers_.begin();
        }
        printf("choose worker id = %d\n", w->Device()->Id());
        return std::move(w);
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

#endif //LDA_ENGINE_H
