//
// Created by chnlkw on 11/28/17.
//

#ifndef LDA_ENGINE_H
#define LDA_ENGINE_H

#include "cuda_utils.h"
#include <deque>
#include <map>

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

#endif //LDA_ENGINE_H
