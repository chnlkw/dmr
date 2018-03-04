//
// Created by chnlkw on 11/28/17.
//

#include <easylogging++.h>
#include "Engine.h"
#include "Task.h"
#include "Worker.h"
#include "Data.h"
#include "DevicesGroup.h"

#define LG(x) CLOG(x, "Engine")

void TaskBase::WaitFinish() {
    while (!finished) {
        if (!engine_.Tick()) {
            throw std::runtime_error("Task unfinished, while engine ends");
        }
    }
}

void DataBase::Wait() const {
    // Wait all tasks finish
    for (auto &s : tasks_scheduled_) {
        if (auto t = s.lock()) {
            t->WaitFinish();
        }
    }
    tasks_scheduled_.clear();
}

Engine::Engine(std::unique_ptr<::CPUDevice> cpu_device, std::unique_ptr<MyDeviceGroup> g) :
        cpu_device_(std::move(cpu_device)),
        devices_(std::move(*g)) {
    LG(INFO) << "engine created with cpu_device=" << cpu_device_.get() << " and devices.size() = " << devices_.size();
    for (auto &d : devices_) {
        for (auto &w : d->Workers())
            workers_.push_back(w.get());
    }
}

//Engine::Engine(std::unique_ptr<CPUDevice> cpu_device, std::unique_ptr<DevicesGroup> devices_group) :
//        cpu_device_(std::move(cpu_device)),
//        devices_(std::move(devices_group->FetchDevices())) {
//    LG(INFO) << "engine created with cpudevice=" << cpu_device_.get() << " and devices.size() = " << devices_.size();
//}

//Engine::Engine( const di::extension::ifactory<DeviceBase, int>& device_factory) {
//    auto d1 = device_factory.create(-1);
//    auto d2 = device_factory.create(-1);
//}

void Engine::AddEdge(TaskPtr src, TaskPtr dst) {
    if (src->finished)
        return;
    LG(INFO) << "AddEdge " << *src << " -> " << *dst;
    tasks_[src].next_tasks_.push_back(dst);
    tasks_[dst].in_degree++;
}

//Engine::Engine(std::set<WorkerPtr> workers) :
//        workers_(std::move(workers)) {
//}

bool Engine::Tick() {
    LG(INFO) << "Tick";
//    std::cout << "Tick " << std::endl;
    size_t empty_workers_ = 0;
    for (auto w : workers_) {
//        std::cout << "workers " << w.get() << std::endl;
        if (w->Empty()) {
            empty_workers_++;
            continue;
        }
        auto tasks = w->GetCompleteTasks();
        for (auto &t : tasks) {
            FinishTask(t);
            LG(INFO) << "Finish task " << *t;
        }
    }
    if (ready_tasks_.empty() && empty_workers_ == workers_.size()) {
        return false; // Finish
    }
    for (auto t : ready_tasks_) {
//        std::cout << "Engine Run Task " << std::endl;
        LG(INFO) << "Choose worker of task " << *t;
        WorkerPtr w = ChooseWorker(t);
        w->RunTask(t);

        LG(INFO) << "After run task " << *t;
        for (auto &m : t->GetMetas()) {
            LG(DEBUG) << m << " replica count = " << m.data->last_state_.replicas.size();
            for (auto dev_arr : m.data->last_state_.replicas) {
                DevicePtr dev = dev_arr.first;
                LG(DEBUG) << "\treplica at  " << *dev;
            }
        }
        LG(INFO) << "          End";

    }
    ready_tasks_.clear();
    return true;
}

WorkerPtr Engine::ChooseWorker(TaskPtr t) {
//    if (t->worker_prefered_.size()) {
//        return *(t->worker_prefered_.begin());
//    } else {
    for (auto &m : t->GetMetas()) {
        LG(DEBUG) << m << " replica count = " << m.data->last_state_.replicas.size();
        for (DevicePtr dev : m.data->DevicesPrefered()) {
            LG(DEBUG) << "\t prefer device : " << dev;
            WorkerPtr worker_choosed = nullptr;
            for (auto &w : dev->Workers()) {
//                WorkerPtr w = ww.lock();
                if (w && !worker_choosed)
                    worker_choosed = w.get();
                if (w->Empty()) {
                    LG(INFO) << "Task " << *t << " chooses worker " << *w << " by affinity and empty";
                    return w.get();
                }
            }
            if (worker_choosed) {
                LG(INFO) << "Task " << *t << " chooses worker " << *worker_choosed << " by affinity ";
                return worker_choosed;
            }
        }
    }
    assert(!workers_.empty());
    for (auto w: workers_)
        if (w->Empty()) {
            LG(INFO) << "Task " << *t << " chooses worker " << *w << " by empty";
            return w;
        }
    LG(INFO) << "Task " << *t << " chooses worker " << **workers_.begin() << " by default";
    return *workers_.begin();
//    }
}

void Engine::CheckTaskReady(TaskPtr task) {
    if (tasks_[task].in_degree == 0)
        ready_tasks_.push_back(task);
}

void Engine::FinishTask(TaskPtr task) {
    for (auto t : tasks_[task].next_tasks_) {
        --tasks_[t].in_degree;
        CheckTaskReady(t);
    }
    task->Finish();
    tasks_.erase(task);
}

TaskBase &Engine::AddTask(TaskPtr task) {
    LG(INFO) << "AddTask " << *task;
    for (auto &m : task->GetMetas()) {
        const auto &depend_tasks = m.data->RegisterTask(task, m.read_only);
        for (const auto &depend_task : depend_tasks) {
            if (!depend_task.expired())
                AddEdge(depend_task.lock(), task);
        }
    }
    CheckTaskReady(task);
    return *task;
}

const std::vector<std::unique_ptr<DeviceBase>> &Engine::GetDevices() const {
    return devices_;
}

DevicePtr Engine::GetCPUDevice() { return engine->CpuDevice(); }

Engine &Engine::Get() { return *engine; }

const DevicePtr Engine::CpuDevice() const { return cpu_device_.get(); }

std::shared_ptr<Engine> Engine::engine;

