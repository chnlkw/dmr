//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_DEVICE_H
#define DMR_DEVICE_H

#include <queue>
#include "defs.h"
#include "Allocator.h"

class DeviceBase : public el::Loggable {
    struct PriorityTask {
        int priority;
        TaskPtr task;

        bool operator<(const PriorityTask &that) const {
            return priority < that.priority;
        }
    };

    int id_;
    AllocatorPtr allocator_;
    std::vector<std::weak_ptr<WorkerBase>> workers_;

public:
    explicit DeviceBase(int id, AllocatorPtr allocator) :
            id_(id),
            allocator_(allocator) {
    }

    virtual ~DeviceBase() {}


    AllocatorPtr GetAllocator() {
        return allocator_;
    }

    void RegisterWorker(WorkerPtr w) {
        workers_.push_back(w);
    }

    const auto &Workers() const {
        return workers_;
    }

    bool Tick();

    int Id() const { return id_; }

    virtual void log(el::base::type::ostream_t &os) const {
        os << "Device[" << id_ << "]";
    }
};

class CPUDevice : public DeviceBase {
public:
#ifdef USE_CUDA
    CPUDevice() : DeviceBase(-1, AllocatorPtr(new CudaPreAllocator(-1, 8LU<<30))) { }
#else
    CPUDevice() : DeviceBase(-1, AllocatorPtr(new CPUAllocator)) { }
#endif
};

class GPUDevice : public DeviceBase {
public:
    GPUDevice(std::shared_ptr<CudaAllocator> allocator) :
            DeviceBase(allocator->DeviceId(), allocator) {
    }
};

class Device {
    static DevicePtr current;
    static DevicePtr cpu;
public:
    static DevicePtr Current() {
//        printf("current device = %d\n", current->Id());
        return current;
    }

    static DevicePtr UseCPU() { return current = cpu; }

    static DevicePtr CpuDevice() { return cpu; }

    static void Use(DevicePtr dev) {
        current = dev;
    }

    static int NumGPUs() {

#ifdef USE_CUDA
        int count;
        CUDA_CALL(cudaGetDeviceCount, &count);
        return count;
#else
        return 0;
#endif
    }

};

//template<class V>
//DevicePtr Device(const V &v);
//
//template<class T>
//DevicePtr Device(const std::vector<T> &v) {
//    return Device::CpuDevice();
//}
//
//template<class T>
//DevicePtr Device(const Data<T> &v) {
//    return v.DeviceCurrent();
//}

#endif //DMR_DEVICE_H
