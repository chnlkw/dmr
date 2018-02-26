//
// Created by chnlkw on 11/28/17.
//

#include <easylogging++.h>
#include "Engine.h"

#define LG(x) CLOG(x, "Engine")

std::unique_ptr<Engine> Engine::engine;

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

ArrayBasePtr DataBase::Read(DevicePtr dev) const {
    Wait();
    ArrayBasePtr ret = last_state_.ReadAt(dev, 0);
    CUDA_CALL(cudaStreamSynchronize, 0);
    return ret;
}

ArrayBasePtr DataBase::Write(DevicePtr dev, size_t bytes) {
    Wait();
    ArrayBasePtr ret = last_state_.WriteAt(dev, 0, false, bytes);
    CUDA_CALL(cudaStreamSynchronize, 0);
    return ret;
}

ArrayBasePtr DataBase::ReadWrite(DevicePtr dev) {
    ArrayBasePtr ret = last_state_.WriteAt(dev, 0, true, last_state_.bytes);
    CUDA_CALL(cudaStreamSynchronize, 0);
    return ret;
}

void Engine::AddEdge(TaskPtr src, TaskPtr dst) {
    if (src->finished)
        return;
    LG(INFO) << "AddEdge " << *src << " -> " << *dst;
    tasks_[src].next_tasks_.push_back(dst);
    tasks_[dst].in_degree++;
}

Engine::Engine(std::set<WorkerPtr> workers) :
        workers_(std::move(workers)) {
}

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
        LOG(INFO) << "Choose worker of task " << *t;
        WorkerPtr w = ChooseWorker(t);
        w->RunTask(t);
    }
    ready_tasks_.clear();
    return true;
}

WorkerPtr Engine::ChooseWorker(TaskPtr t) {
//    if (t->worker_prefered_.size()) {
//        return *(t->worker_prefered_.begin());
//    } else {
    for (auto &m : t->GetMetas()) {
        LG(DEBUG) << m << " replicas " << m.data->last_state_.replicas.size();
        for (auto dev_arr : m.data->last_state_.replicas) {
            DevicePtr dev = dev_arr.first;
            WorkerPtr worker_choosed;
            for (auto ww : dev->Workers()) {
                WorkerPtr w = ww.lock();
                if (w && !worker_choosed)
                    worker_choosed = w;
                if (w->Empty()) {
                    LG(INFO) << "Task " << *t << " chooses worker " << *w << " by affinity and empty";
                    return w;
                }
            }
            if (worker_choosed) {
                LG(INFO) << "Task " << *t << " chooses worker " << *worker_choosed << " by affinity ";
                return worker_choosed;
            }
        }
    }
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
        if (m.read_only) {
            auto &d = m.data;
            d->AddTask(task);
            DataNode &data = data_[d];
            for (const auto &w : data.ReadBy(task))
                AddEdge(w, task);
        } else {
            auto &d = m.data;
            d->AddTask(task);
            DataNode &data = data_[d];
            for (const auto &r : data.WriteBy(task))
                AddEdge(r, task);
//        assert(!(!data.readers.empty() && !data.writers.empty()));
//        if (data.readers.size() > 0 && data.writers.empty()) {
//            for (const auto &r : data.readers)
//                AddEdge(r, task);
//        } else if (data.readers.empty() && data.writers.size() > 0) {
//            data.writers.p
//
//        }
//        if (data.writer && data.readers.empty()) {
//            LG(ERROR) << "Data written twice but no readers between them\n");
//        }
//        data.readers.clear();
//        data.writer = task;
        }
    }
    CheckTaskReady(task);
    return *task;
}

void Engine::Create(std::set<WorkerPtr> workers) {
    LG(INFO) << "Engine Create with " << workers.size() << " workers";
    engine.reset(new Engine(std::move(workers)));
}

Engine &Engine::Get() {
    if (!engine)
        throw std::runtime_error("Engine::Create() must be called before ::Get()");
    return *engine;
}

void Engine::Finish() {
    Device::UseCPU();
    engine.reset();
}
