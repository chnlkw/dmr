//
// Created by chnlkw on 3/2/18.
//

#ifndef DMR_DEVICESGROUP_H
#define DMR_DEVICESGROUP_H

#include <boost/di/extension/injections/factory.hpp>
#include "defs.h"
#include "Allocator.h"
#include "Device.h"

namespace di = boost::di;

template<class T>
struct Generator {
    std::vector<std::unique_ptr<T>> data;

    Generator(size_t n) {
        for (size_t i = 0; i < n; i++) {
            data.push_back(std::make_unique<T>(i));
        }
    }

    auto &&operator()() {
        return std::move(data);
    }

};

class DevicesGroup {
protected:
    std::vector<std::unique_ptr<DeviceBase>> devices_;
public:
    auto &&operator()() {
        return std::move(devices_);
    }

    auto &&FetchDevices() {
        return std::move(devices_);
    }

    virtual ~DevicesGroup() {}
};

auto NumGPUInGroup = [] {};

class GPUGroup : public DevicesGroup {
public:
    BOOST_DI_INJECT(GPUGroup, (named = NumGPUInGroup)
            size_t n, const di::extension::ifactory<DeviceBase> &device_factory);
};

struct MyDeviceGroup : std::vector<std::shared_ptr<DeviceBase>> {
};

struct CPUGroupFactory {
    size_t num_cpus;

    explicit CPUGroupFactory(size_t num_cpus) : num_cpus(num_cpus) {}

    template<class TInjector, class TDependency>
    auto operator()(const TInjector &injector, const TDependency &) const {
        auto g = std::make_unique<MyDeviceGroup>();
        for (size_t i = 0; i < num_cpus; i++) {
            LOG(INFO) << "MyDeviceGroup creating CPUDevice " << i;
            g->push_back(injector.template create<std::shared_ptr<CPUDevice>>());
        }
        return std::move(g);
    }

};

struct GPUGroupFactory {
    size_t num_gpus;

    explicit GPUGroupFactory(size_t num_gpus) : num_gpus(num_gpus) {}

    template<class TInjector, class TDependency>
    auto operator()(const TInjector &injector, const TDependency &) const {
        TInjector &injector_(const_cast<TInjector &>(injector));
        auto inj = di::make_injector(std::move(injector_),
                                     di::bind<int>().named(myDeviceId).to([]() {
                                         static int seq = 0;
                                         return seq++;
                                     })
        );
        auto g = std::make_unique<MyDeviceGroup>();
        for (size_t i = 0; i < num_gpus; i++) {
            LOG(INFO) << "MyDeviceGroup creating i = " << i;
            g->emplace_back(inj.template create<GPUDevice *>()); //singleton
        }
        return std::move(g);
    }

};

struct CPUGPUGroupFactory {
    size_t num_cpus;
    size_t num_gpus;

    explicit CPUGPUGroupFactory(size_t num_cpus, size_t num_gpus) :
            num_cpus(num_cpus),
            num_gpus(num_gpus) {}

    template<class TInjector, class TDependency>
    auto operator()(const TInjector &injector, const TDependency &) const {
        TInjector &injector_(const_cast<TInjector &>(injector));
        auto inj = di::make_injector(std::move(injector_),
                                     di::bind<int>().named(myDeviceId).to([]() {
                                         static int seq = 0;
                                         return seq++;
                                     })
        );
        auto g = std::make_unique<MyDeviceGroup>();
        for (size_t i = 0; i < num_cpus; i++) {
            LOG(INFO) << "MyDeviceGroup creating CPU device " << i;
            g->emplace_back(inj.template create<CPUDevice *>()); //singleton
        }
        for (size_t i = 0; i < num_gpus; i++) {
            LOG(INFO) << "MyDeviceGroup creating GPU device " << i;
            g->emplace_back(inj.template create<GPUDevice *>()); //singleton
        }
        return std::move(g);
    }

};

#endif //DMR_DEVICESGROUP_H
