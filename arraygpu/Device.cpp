//
// Created by chnlkw on 1/25/18.
//

#include "Device.h"
#include "Worker.h"
#include "Allocator.h"

#define LG(x) CLOG(x, "Device")

//DevicePtr Device::cpu(new CPUDevice);
//DevicePtr Device::current = Device::cpu;

//int Device::NumGPUs() {
//#ifdef USE_CUDA
//    int count;
//    CUDA_CALL(cudaGetDeviceCount, &count);
//    return count;
//#else
//    return 0;
//#endif
//}

GPUDevice::GPUDevice(std::unique_ptr<CudaAllocator> allocator, int num_workers) :
        DeviceBase(std::move(allocator)) {
    LG(INFO) << "Create GPUDevice with allocator " << GetAllocator() << " num_workers = " << num_workers << " ID = " << this->Id();
    for (int i = 0; i < num_workers; i++) {
        workers_.emplace_back(new GPUWorker(this));
    }
}

std::vector<TaskPtr> GPUDevice::GetCompleteTasks() {
    std::vector<TaskPtr> ret;
    for (auto& w : workers_) {
        auto r = w->GetCompleteTasks();
        ret.insert(ret.end(), r.begin(), r.end());
    }
    return ret;
}

void GPUDevice::RunTask(TaskPtr t) {
    running_tasks_++;
    ChooseRunnable(workers_.begin(), workers_.end())->RunTask(t);
}

DeviceBase::DeviceBase(std::unique_ptr<AllocatorBase> allocator) :
        allocator_(std::move(allocator)) {
}

DeviceBase::~DeviceBase() {}

int DeviceBase::Id() const { return allocator_->Id(); }

CPUDevice::CPUDevice() : DeviceBase(std::make_unique<CudaPreAllocator>(-1, 8LU << 30)) {}
