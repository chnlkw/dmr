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

    explicit Engine(std::set<WorkerPtr> workers);

public:

    static void Create(std::set<WorkerPtr> workers);

    static void Finish();

    static Engine &Get();

    TaskBase &AddTask(TaskPtr task);

    template<class Task, class... Args>
    TaskBase &AddTask(Args... args) {
        auto t = std::make_shared<Task>(*this, args...);
        return AddTask(t);
    };

    bool Tick();

    WorkerPtr ChooseWorker(TaskPtr t);

private:
    void AddEdge(TaskPtr src, TaskPtr dst);

    void CheckTaskReady(TaskPtr task);

    void FinishTask(TaskPtr task);
};

#endif //LDA_ENGINE_H
