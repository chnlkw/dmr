//
// Created by chnlkw on 11/28/17.
//

#include "Engine.h"

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
    tasks_[src].next_tasks_.push_back(dst);
    tasks_[dst].in_degree++;
}

Engine::Engine(std::set<WorkerPtr> workers) :
        workers_(std::move(workers)) {
}

bool Engine::Tick() {
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

WorkerPtr Engine::ChooseWorker(TaskPtr t) {
    WorkerPtr w;
    if (t->worker_prefered_.size()) {
        w = *(t->worker_prefered_.begin());
    } else {
        w = *workers_.begin();
    }
    printf("choose worker id = %d\n", w->Device()->Id());
    return std::move(w);
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
    for (auto &d : task->GetInputs()) {
        d->AddTask(task);
        DataNode &data = data_[d];
        if (data.writer)
            AddEdge(data.writer, task);
        data.readers.push_back(task);
    }
    for (auto &d : task->GetOutputs()) {
        d->AddTask(task);
        DataNode &data = data_[d];
        if (data.writer && data.readers.empty()) {
            fprintf(stderr, "Data written twice but no readers between them\n");
        }
        for (const auto &r : data.readers)
            AddEdge(r, task);
        data.readers.clear();
        data.writer = task;
    }
    CheckTaskReady(task);
    return *task;
}

void Engine::Create(std::set<WorkerPtr> workers) {
    engine.reset(new Engine(std::move(workers)));
}

Engine &Engine::Get() {
    if (!engine)
        throw std::runtime_error("Engine::Create() must be called before ::Get()");
    return *engine;
}

void Engine::Finish() {
    engine.reset();
}
