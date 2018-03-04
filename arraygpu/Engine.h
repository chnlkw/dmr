//
// Created by chnlkw on 11/28/17.
//

#ifndef LDA_ENGINE_H
#define LDA_ENGINE_H

#include "cuda_utils.h"
#include <deque>
#include <map>
#include <set>
#include "defs.h"
#include "Device.h"
#include "DevicesGroup.h"
#include <boost/di.hpp>

namespace di = boost::di;

class Engine {
    struct Node {
        size_t in_degree = 0;
        std::vector<TaskPtr> next_tasks_;
    };
    std::map<TaskPtr, Node> tasks_;

    std::vector<WorkerPtr> workers_;
    std::vector<TaskPtr> ready_tasks_;

    std::unique_ptr<DeviceBase> cpu_device_;
    std::vector<std::unique_ptr<DeviceBase>> devices_;

private:

    static std::shared_ptr<Engine> engine;

public:

    BOOST_DI_INJECT(Engine, std::unique_ptr<CPUDevice> cpu_device, std::unique_ptr<MyDeviceGroup> g);

    static void Set(std::shared_ptr<Engine> e) { engine = e; }

    static Engine &Get();

    static DevicePtr GetCPUDevice();

    static void Finish() { engine.reset(); }

    const DevicePtr CpuDevice() const;

    const std::vector<std::unique_ptr<DeviceBase>> &GetDevices() const;

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
