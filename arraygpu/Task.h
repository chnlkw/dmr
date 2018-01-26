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

class Node {
    int id_;
    std::set<NodePtr> depends_;
    std::set<NodePtr> suffixs_;

public:
    Node(int id = 0) : id_(id) {
    }

    virtual ~Node() {}

    void AddDependence(NodePtr ptr) {
        depends_.insert(ptr);
    }
};

class TaskBase : public Node {
    std::list<DataPtr> inputs_, outputs_;
public:
    virtual void Run(DevicePtr device) = 0;
};

class Engine {
    std::set<DevicePtr> devices_;
    std::set<TaskPtr> dependent_tasks_;

public:
    Engine() {}

    void AddDevice(DevicePtr device) {
        devices_.insert(device);
    }

    void AddTask(TaskPtr task) {
        dependent_tasks_.insert(task);
    }

    void WaitTask(TaskPtr task);

    void WaitAll();

    bool Tick();
};

#endif //DMR_TASK_H
