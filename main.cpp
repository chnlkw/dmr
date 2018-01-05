#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <functional>

#include "dmr.h"
#include "PartitionedDMR.h"

#include "range/v3/all.hpp"

using namespace ranges;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_int_distribution<> dis(0, 5);
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

template<class Idx, class Val, class Target>
void Shuffle(Idx &&idx, Val &&val, Target &tar) {
    for_each(view::zip(idx, val), [&tar](auto x) { tar[x.first] = x.second; });
};


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

template<class K, class V>
struct CSR {
    std::vector<K> keys;
    std::vector<size_t> offs;
    std::vector<V> values;

public:
    void Build(any_view<std::pair<K, V>> rng_kv) {
        values = rng_kv | view::values;
        std::vector<K> k = rng_kv | view::keys;
        sort(view::zip(k, values));
        auto rng_kf = view::zip(k, view::ints) |
                      view::adjacent_filter([](auto x, auto y) { return x.first != y.first; });
        keys = rng_kf | view::keys;
        offs = rng_kf | view::values;
        offs.push_back(values.size());
    }

    CSR() {}

    CSR(std::vector<std::pair<K, V>> &vec) {
        if (vec.size() == 0)
            return;
        values.resize(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            if (i == 0 || vec[i].first != vec[i - 1].first) {
                keys.push_back(vec[i].first);
                offs.push_back(i);
            }
            values[i] = vec[i].second;
        }
        offs.push_back(values.size());
    }

    const std::vector<V> &Values() {
        return values;
    }

    V *operator[](size_t idx) {
        return values.data() + offs[idx];
    }
};

void test_idx_value_1d() {
    auto n = 10;

    std::vector<int> val = view::ints(0, n) | view::transform([&](int i) { return dis(gen); });

    CSR<int, int> csr;
    csr.Build(view::zip(val, view::ints));
    std::vector<int> pos(n);
    Shuffle(csr.Values(), view::ints, pos);

    std::cout << view::all(pos) << '\n';
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    int N = 4, M = 4;
//    DMR<uint32_t> dmr(N, M);
    PartitionedDMR<uint32_t> dmr(N, M);

    std::vector<std::vector<uint32_t>> keys(N), values(N);
    for (int i = 0; i < 10; i++) {
        uint32_t k = dis(gen);
        uint32_t v = dis(gen);
        keys[i % N].push_back(k);
        values[i % N].push_back(v);
        a[k] ^= v;
    }
    for (int i = 0; i < N; i++) {
        dmr.SetMapperKeys(i, keys[i].data(), keys[i].size());
    }
    dmr.Prepare();
    auto shuf = dmr.GetShuffler<uint32_t>();
    for (int i = 0; i < M; i++) {
        shuf.SetMapperValues(i, values[i].data(), values[i].size());
    }
    shuf.Run();
    for (auto &reducer : shuf.Reducers()) {
        auto &keys = reducer.Keys();
        auto &offs = reducer.Offs();
        auto &values = reducer.Values();
        for (size_t i = 0; i < keys.size(); i++) {
            auto k = keys[i];
            for (int j = offs[i]; j < offs[i + 1]; j++) {
                auto v = values[j];
                a[k] ^= v;
            }
        }
    }
    for (auto x : a) {
        if (x.second != 0) {
            fprintf(stderr, "key %d not match %d\n", x.first, x.second);
            abort();
        }
    }
    return 0;
}