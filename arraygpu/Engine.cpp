//
// Created by chnlkw on 11/28/17.
//

#if 0
#include "Engine.h"
#include "All.h"

TaskEngine::~TaskEngine() {
    if (queue_.size() > 0) {
        std::cerr << "TaskEngine not clear when destroying" << std::endl;
    }
    for (cudaEvent_t e : events_unused_) {
        cudaEventDestroy(e);
    }
    events_unused_.clear();

    if (stream_)
        cudaStreamDestroy(stream_);
}

void TaskEngine::Push(int id) {
    cudaSetDevice(device_);
    cudaEvent_t e;
    if (events_unused_.size() > 0) {
        e = events_unused_.back();
        events_unused_.pop_back();
    } else {
        cudaEventCreate(&e);
    }
//    std::cout << "event record " << stream_ << ' ' << e << std::endl;
    cudaEventRecord(e, stream_);
    queue_.emplace_back(e, id);
}

int TaskEngine::Poll() {
    if (Empty())
        return -1;
    int id;
    cudaEvent_t e;
    std::tie(e, id) = queue_.front();
    cudaError_t ret = cudaEventQuery(e);
    if (ret == cudaSuccess) {
//        std::cout << "event poll " << stream_ << ' ' << e << std::endl;
        queue_.pop_front();
        events_unused_.push_back(e);
        return id;
    }
    return -2;
}

int TaskEngine::Pop() {
    if (queue_.empty())
        return -1;
    int id;
    cudaEvent_t e;
    std::tie(e, id) = queue_.front();
    queue_.pop_front();
//    std::cout << "event wait " << stream_ << ' ' << e << std::endl;
    cudaEventSynchronize(e);
    events_unused_.push_back(e);
    return id;
}

void GpusPipeline::RunTask(int seq) {
    GpuTaskBase &task = *tasks_[seq];
    if (task.context.device < 0) {
        //finish task
        tasks_.erase(seq);
        return;
    }
    TaskEngine &e = dev_engines_.at(task.context.device);
//    std::cout << "tick seq " << seq << " stream " << e.stream_ << std::endl;
    task.context.stream = e.stream_;
    task.Tick();
    e.Push(seq);
    flag_empty = false;
}

GpusPipeline::GpusPipeline(size_t num_devices) : seq(0), flag_empty(true), cur_dev(0) {
    for (int i = 0; i < num_devices; i++)
        dev_engines_.emplace_back(i);
}

GpusPipeline::GpusPipeline(GpusPipeline &&that) :
        dev_engines_(std::move(that.dev_engines_)),
        tasks_(std::move(that.tasks_)),
        flag_empty(std::move(that.flag_empty)),
        seq(that.seq),
        cur_dev(that.cur_dev) {
}

void GpusPipeline::AddGpuTask(GPUTaskPtr ptr) {
    int task_seq = seq++;
    tasks_[task_seq] = std::move(ptr);
    RunTask(task_seq);
}

bool GpusPipeline::Tick() {
    if (flag_empty)
        return flag_empty;

    flag_empty = true;

    for (int count = 0; count < dev_engines_.size(); count++) {
        TaskEngine &e = dev_engines_.at(cur_dev);

        if (!e.Empty()) {
            flag_empty = false;
            int seq = e.Poll();
            if (seq >= 0)
                RunTask(seq);
            return false;
        }
        cur_dev = (cur_dev + 1) % (int) dev_engines_.size();
    }
    return flag_empty;
}

void GpusPipeline::Run() {
    while (!Tick()) {
    }
}

GpusEngine::GpusEngine(size_t num_stream, size_t num_devices) :
        num_stream_(num_stream),
        cur_id_(0) {
    assert(num_devices > 0);
    assert(num_stream > 0);
    for (size_t i = 0; i < num_stream; i++) {
        gpus_pipelines_.emplace_back(num_devices);
    }
}

void GpusEngine::AddGpuTask(GPUTaskPtr ptr) {
    while (true) {
        size_t i = cur_id_;
        GpusPipeline &gpus_stream = gpus_pipelines_[i];
        cur_id_ = (cur_id_ + 1) % gpus_pipelines_.size();
        if (gpus_stream.Tick()) {
//            std::cout << "Add Gpu Task to Engine " << i << std::endl;
            ptr->context.pipeline_id = (int) i;
            gpus_stream.AddGpuTask(std::move(ptr));
            return;
        }
    }
}

void GpusEngine::Run() {
    bool finish = false;
    while (!finish) {
        finish = true;
        for (GpusPipeline &gpus_stream : gpus_pipelines_) {
            finish &= gpus_stream.Tick();
        }
    }

}

#endif
