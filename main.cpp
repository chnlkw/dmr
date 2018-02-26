#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <functional>
#include <set>
#include <clock.h>

#include <All.h>
#include "dmr.h"
#include "PartitionedDMR.h"

#include "easylogging++.h"

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_int_distribution<uint32_t> dis(1, 9);
std::map<int, int> a;

auto print = [](auto &x) { std::cout << " " << x; };
auto self = [](auto x) { return x; };

namespace std {
    template<class K, class V>
    std::ostream &operator<<(std::ostream &os, const std::pair<K, V> &p) {
        os << "(" << p.first << "," << p.second << ")";
        return os;
    };
}

//template<class Idx, class Val, class Target>
//void Shuffle(Idx &&idx, Val &&val, Target &tar) {
//    for_each(view::zip(idx, val), [&tar](auto x) { tar[x.first] = x.second; });
//};


template<int dim>
class Index;

template<>
class Index<1> {
    long idx_;

public:
    Index(long i = 0) : idx_(i) {}

    template<class V>
    auto &operator()(V &v) {
        return v[idx_];
    }

    bool operator<(const Index &that) const {
        return idx_ < that.idx_;
    }
};

template<int dim>
class Index {
    long idx_;
    Index<dim - 1> next_;

public:
    Index() : idx_(0) {}

    template<class... Args>
    Index(long i, Args... j) : idx_(i), next_(j...) {}

    template<class V>
    auto &operator()(V &v) {
        return next_(v[idx_]);
    }

    bool operator<(const Index &that) const {
        return idx_ < that.idx_ || (idx_ == that.idx_ && next_ < that.next_);
    }
};

//template<class K, class V>
//struct CSR {
//    std::vector<K> keys;
//    std::vector<size_t> offs;
//    std::vector<V> values;
//
//public:
//    void Build(any_view<std::pair<K, V>> rng_kv) {
//        values = rng_kv | view::values;
//        std::vector<K> k = rng_kv | view::keys;
//        sort(view::zip(k, values));
//        auto rng_kf = view::zip(k, view::ints) |
//                      view::adjacent_filter([](auto x, auto y) { return x.first != y.first; });
//        keys = rng_kf | view::keys;
//        offs = rng_kf | view::values;
//        offs.push_back(values.size());
//    }
//
//    CSR() {}
//
//    CSR(std::vector<std::pair<K, V>> &vec) {
//        if (vec.size() == 0)
//            return;
//        values.resize(vec.size());
//        for (size_t i = 0; i < vec.size(); i++) {
//            if (i == 0 || vec[i].first != vec[i - 1].first) {
//                keys.push_back(vec[i].first);
//                offs.push_back(i);
//            }
//            values[i] = vec[i].second;
//        }
//        offs.push_back(values.size());
//    }
//
//    const std::vector<V> &Values() {
//        return values;
//    }
//
//    V *operator[](size_t idx) {
//        return values.data() + offs[idx];
//    }
//};

//void test_idx_value_1d() {
//    auto n = 10;

//    std::vector<int> val = view::ints(0, n) | view::transform([&](int i) { return dis(gen); });

//    CSR<int, int> csr;
//    csr.Build(view::zip(val, view::ints));
//    std::vector<int> pos(n);
//    Shuffle(csr.Values(), view::ints, pos);

//    std::cout << view::all(pos) << '\n';
//}

void test_dmr() {

//    std::vector<DevicePtr> gpu_devices;
//    for (int i = 0; i < Device::NumGPUs(); i++) {
//        gpu_devices.push_back(std::make_shared<GPUDevice>(std::make_shared<CudaPreAllocator>(i, 2LU << 30)));
//        gpu_devices.push_back(std::make_shared<GPUDevice>(std::make_shared<CudaAllocator>(i)));
//    }

    Device::UseCPU();

#if 0
    if (true) {
        std::vector<int> keys = {1, 3, 4, 2, 5};
        DMR<int, data_constructor_t> dmr1(keys);
        Data<float> values(keys.size());
        {
            auto v = values.Write().data();
            for (int i = 0; i < keys.size(); i++) {
                v[i] = keys[i] * 10.0f;
            }
        }
//        Device::Use(gpu_devices[0]);
        Data<float> values_out = dmr1.ShuffleValues<float>(values);
//        values_out->Wait();
//        while (Engine::Get().Tick());
        auto d_val_out = values_out.Read(Device::CpuDevice()).data();
        Device::UseCPU();
//        d_val_out.Write(Device::Current());
        {
            auto &keys = dmr1.Keys().Read();
            auto &offs = dmr1.Offs().Read();
            for (size_t i = 0; i < keys.size(); i++) {
                auto k = keys[i];
                for (int j = offs[i]; j < offs[i + 1]; j++) {
                    auto v = d_val_out[j];
                    printf("%d,%f ", k, v);
                }
            }
            printf("\n");
        }
    }
#endif

#if 1
    int N = 2;
//    DMR<uint32_t> dmr(N, M);

    std::vector<std::vector<uint32_t>> keys(N), values(N);
    for (int i = 0; i < 20; i++) {
        uint32_t k = dis(gen);
        uint32_t v = dis(gen);
        keys[i % N].push_back(k);
        values[i % N].push_back(v);
        a[k] ^= v;
    }
#if 1
    PartitionedDMR<uint32_t, data_constructor_t> dmr2(keys);

    std::vector<Data<uint32_t>> d_values;
    for (int i = 0; i < N; i++) {
        d_values.emplace_back(values[i]);
    }

    LOG(INFO) << "Shufflevalues";
    auto result = dmr2.ShuffleValues<uint32_t>(d_values);
    LOG(INFO) << "Shufflevalues OK";
#if 0
    for (int i = 0; i < 5; i++) {
        Clock clk;
        auto r = dmr2.ShuffleValues<uint32_t>(d_values);
        size_t sum = 0;
        for (auto &x : r) {
            sum += x.size() * sizeof(int);
        }
        double t = clk.timeElapsed();
        double speed = sum / t / (1LU << 30);
        printf("sum %lu bytes, time %lf seconds, speed %lf GB/s\n", sum, t, speed);
    }
#endif
    Device::UseCPU();
#else
    auto result = dmr.ShuffleValues<uint32_t>(values);
#endif

    while (Engine::Get().Tick());

    for (auto &v : result) {
        LOG(DEBUG) << "result " << v.ToString();
    }
    std::set<int> exist_keys;
    for (size_t par_id = 0; par_id < dmr2.Size(); par_id++) {
        auto keys = dmr2.Keys(par_id).Read().data();
        auto offs = dmr2.Offs(par_id).Read().data();
        auto values = result[par_id].Read().data();
        for (size_t i = 0; i < dmr2.Keys(par_id).size(); i++) {
            auto k = keys[i];
            if (exist_keys.count(k)) {
                throw std::runtime_error("same key in different partitions");
            }
            exist_keys.insert(k);
            for (int j = offs[i]; j < offs[i + 1]; j++) {
                auto v = values[j];
                a[k] ^= v;
            }
        }
    }
    for (auto x : a) {
        if (x.second != 0) {
            LOG(ERROR) << "key not match" << x.first << " " << x.second;
            abort();
        }
    }
#endif
    std::cout << "OK" << std::endl;
}

template<class T>
class TaskAdd : public TaskBase {
    Data<T> a_, b_, c_;
public:
    TaskAdd(Engine &engine, Data<T> a, Data<T> b, Data<T> c) :
            TaskBase(engine, "Add"),
            a_(a), b_(b), c_(c) {
        assert(a.size() == b.size());
        assert(a.size() == c.size());
        AddInput(a);
        AddInput(b);
        AddOutput(c);
    }

    virtual void Run(CPUWorker *cpu) override {
        LOG(INFO) << "run cpu TaskAdd";
        const T *a = a_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
        const T *b = b_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
        T *c = c_.WriteAsync(shared_from_this(), cpu->Device(), 0).data();
        for (int i = 0; i < c_.size(); i++) {
            c[i] = a[i] + b[i];
        }
    }

    virtual void Run(GPUWorker *gpu) override {
        LOG(INFO) << "run TaskAdd on gpu " << gpu->Device()->Id();
        const T *a = a_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
        const T *b = b_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
        T *c = c_.WriteAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
        gpu_add(c, a, b, c_.size(), gpu->Stream());
    }
};

void test_engine() {

    auto &engine = Engine::Get();

    auto print = [](const auto &arr) {
        printf("%p : ", &arr[0]);
        for (int i = 0; i < arr.size(); i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    };

    auto d1 = Data<int>(10);
    Array<int> &a1 = d1.Write();
    auto d2 = Data<int>(d1.size());
    Array<int> &a2 = d2.Write();
    for (int i = 0; i < d1.size(); i++) {
        a1[i] = i;
        a2[i] = i * i;
    }
    auto d3 = Data<int>(d1.size());

    print(a1);
    print(a2);
//    auto t1 = std::make_shared<TaskAdd<int>>(d1, d2, d3);
//    engine.AddTask(t1);
    engine.AddTask<TaskAdd<int>>(d1, d2, d3);

    auto d4 = Data<int>(d1.size());
    auto t2 = std::make_shared<TaskAdd<int>>(engine, d2, d3, d4);

    engine.AddTask(t2);

    while (engine.Tick());
    t2->WaitFinish();
    print(a1);
    print(a2);
    auto &a3 = d3.Read(Device::CpuDevice());
    CUDA_CHECK();
    print(a3);
    auto &a4 = d4.Read(Device::CpuDevice());
    CUDA_CHECK();
    print(a4);

}

INITIALIZE_EASYLOGGINGPP

int main() {
    el::Loggers::configureFromGlobal("logging.conf");
    LOG(INFO) << "info";
    CLOG(INFO, "Engine") << "engine info";

//    std::vector<GPUWorker> gpu_workers;
//    for (int i = 0; i < Device::NumGPUs(); i++)
//        gpu_workers.emplace_back(i);
//
//    CPUWorker cpu_worker;
    LOG(INFO) << "start";

#if USE_CUDA
    std::vector<DevicePtr> gpu_devices;
    std::vector<WorkerPtr> gpu_workers;
    for (int i = 0; i < Device::NumGPUs(); i++) {
        auto d = std::make_shared<GPUDevice>(std::make_shared<CudaPreAllocator>(i, 2LU << 30));
        gpu_devices.push_back(d);
        gpu_workers.push_back(std::make_shared<GPUWorker>(d));
        d->RegisterWorker(gpu_workers.back());
    }

    Engine::Create({gpu_workers.begin(), gpu_workers.end()});
#else
    std::vector<WorkerPtr> cpu_workers;
    cpu_workers.emplace_back(new CPUWorker());
    Engine::Create({cpu_workers.begin(), cpu_workers.end()});
#endif
    test_engine();
    test_dmr();

    for (auto d : gpu_devices) {
        Device::Use(d);
        CUDA_CHECK();
    }

    Engine::Finish();
}