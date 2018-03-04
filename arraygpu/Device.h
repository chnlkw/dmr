//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_DEVICE_H
#define DMR_DEVICE_H

#include <queue>
#include "defs.h"
#include "cuda_utils.h"
#include <boost/di.hpp>

class DeviceBase : public el::Loggable, public std::enable_shared_from_this<DeviceBase> {
    struct PriorityTask {
        int priority;
        TaskPtr task;

        bool operator<(const PriorityTask &that) const {
            return priority < that.priority;
        }
    };

    std::unique_ptr<AllocatorBase> allocator_;
protected:
    std::vector<std::unique_ptr<WorkerBase>> workers_;

public:
    explicit DeviceBase(std::unique_ptr<AllocatorBase> allocator);

    virtual ~DeviceBase();

    AllocatorPtr GetAllocator() {
        return allocator_.get();
    }

    const auto &Workers() const {
        return workers_;
    }

    bool Tick();

    int Id() const;

    void log(el::base::type::ostream_t &os) const override;
};

class CPUDevice : public DeviceBase {
public:
#ifdef USE_CUDA
    CPUDevice();
#else
    CPUDevice() : DeviceBase(-1, AllocatorPtr(new CPUAllocator)) { }
#endif
};

auto NumWorkersOfGPUDevices = []{};

class GPUDevice : public DeviceBase {
public:
    BOOST_DI_INJECT (GPUDevice, std::unique_ptr<CudaAllocator> allocator,(named = NumWorkersOfGPUDevices) int num_workers = 1);
};

//class Device {
//    static DevicePtr current;
//    static DevicePtr cpu;
//public:
//    static DevicePtr Current() {
//        printf("current device = %d\n", current->Id());
//        return current;
//    }

//    static DevicePtr UseCPU() { return current = cpu; }

//    static DevicePtr CpuDevice() { return cpu; }

//    static void Use(DevicePtr dev) {
//        current = dev;
//    }

//    static int NumGPUs();

//};

#endif //DMR_DEVICE_H
