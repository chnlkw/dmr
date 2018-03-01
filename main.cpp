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

void test_dmr(size_t npar, size_t num_element, int repeat) {
    LOG(INFO) << "start test dmr npar=" << npar << " num_element=" << num_element << " repeat=" << repeat;

    LOG(INFO) << "Initializing Key Value";
    num_element /= npar;
    std::vector<std::vector<uint32_t>> keys(npar), values(npar);
    for (int pid = 0; pid < npar; pid++) {
        keys[pid].resize(num_element);
        values[pid].resize(num_element);
    }
    size_t sum_keys = 0, sum_values = 0;
#pragma omp parallel reduction(+:sum_keys, sum_values)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<uint32_t> dis(1, 100);
        for (int pid = 0; pid < npar; pid++) {
            auto &k = keys[pid];
            auto &v = values[pid];
#pragma omp for
            for (int i = 0; i < k.size(); i++) {
                k[i] = dis(gen);
                v[i] = dis(gen);
                sum_keys += k[i];
                sum_values += v[i];
            }
        }
    }

    LOG(INFO) << "Initializing DMR keys";
    PartitionedDMR<uint32_t, data_constructor_t> dmr2(keys);

    LOG(INFO) << "Initializing Input values";
    std::vector<Data<uint32_t>> d_values;
    for (int i = 0; i < npar; i++) {
        d_values.emplace_back(values[i]);
    }

    LOG(INFO) << "Shufflevalues";
    auto result = dmr2.ShuffleValues<uint32_t>(d_values);
    while (Engine::Get().Tick());
    LOG(INFO) << "Shufflevalues OK";

//    for (auto &v : result) {
//        LOG(DEBUG) << "result " << v.ToString();
//    }
    size_t sum_keys_2 = 0, sum_values_2 = 0;

    LOG(INFO) << "Checking results";

    for (size_t par_id = 0; par_id < dmr2.Size(); par_id++) {
        auto keys = dmr2.Keys(par_id).Read().data();
        auto offs = dmr2.Offs(par_id).Read().data();
        auto values = result[par_id].Read().data();

#pragma omp parallel for reduction(+:sum_keys_2, sum_values_2)
        for (size_t i = 0; i < dmr2.Keys(par_id).size(); i++) {
            auto k = keys[i];
            for (int j = offs[i]; j < offs[i + 1]; j++) {
                auto v = values[j];
                sum_keys_2 += k;
                sum_values_2 += v;
            }
        }
    }
    if (sum_keys != sum_keys_2 || sum_values != sum_values_2) {
        LOG(FATAL) << "sum not match" << sum_keys << ' ' << sum_keys_2 << ' ' << sum_values << ' ' << sum_keys_2;
    }
    LOG(INFO) << "Result OK";

    LOG(INFO) << "Run benchmark ";
    for (int i = 0; i < repeat; i++) {
        Clock clk;
        for (auto &d : d_values) {
            d.Write();
        }
        auto r = dmr2.ShuffleValues<uint32_t>(d_values);
        size_t sum = 0;
        for (auto &x : r) {
            x.Read();
            sum += x.size() * sizeof(int);
        }
        double t = clk.timeElapsed();
        double speed = sum / t / (1LU << 30);
        while (Engine::Get().Tick());
        printf("sum %lu bytes, time %lf seconds, speed %lf GB/s\n", sum, t, speed);
    }
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
        const T *a = a_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
        const T *b = b_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
        T *c = c_.WriteAsync(shared_from_this(), cpu->Device(), 0).data();
        for (int i = 0; i < c_.size(); i++) {
            c[i] = a[i] + b[i];
        }
    }

    virtual void Run(GPUWorker *gpu) override {
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
//    engine.RegisterTask(t1);
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

int main(int argc, char **argv) {
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
    DataCopyInitP2P();

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
    int npar = 2;
    int nelem = 10;
    int rep = 2;
    if (argc > 1) npar = atoi(argv[1]);
    if (argc > 2) nelem = atoi(argv[2]);
    if (argc > 3) rep = atoi(argv[3]);
    test_dmr(npar, nelem, rep);

    for (auto d : gpu_devices) {
        Device::Use(d);
        CUDA_CHECK();
    }

    Engine::Finish();
}