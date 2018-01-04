#include <iostream>
#include <vector>
#include <random>
#include <map>

#include "dmr.h"

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_int_distribution<> dis(1, 10);
std::map<int, int> a;

int main() {
    std::cout << "Hello, World!" << std::endl;

    int N = 4, M = 4;
    DMR<uint32_t> dmr(N, M);

    std::vector<std::vector<uint32_t>> keys(N), values(N);
    for (int i = 0; i < 10; i++) {
        uint32_t k = dis(gen);
        uint32_t v = dis(gen);
        keys[i % N].push_back(k);
        values[i % N].push_back(v);
        a[k] ^=v;
    }
    for (int i = 0; i < N; i++) {
        dmr.SetMapperKeys(i, keys[i].data(), keys[i].size());
    }
    dmr.Prepare();
    auto shuf = dmr.GetShuffler<uint32_t>();
    for (int i = 0; i < N; i++) {
        shuf.SetMapperValues(i, values[i].data(), values[i].size());
    }
    shuf.Run();
    for (auto& reducer : shuf.Reducers()) {
        auto &keys = reducer.Keys();
        auto &offs = reducer.Offs();
        auto &values = reducer.Values();
        for (size_t i = 0; i < keys.size(); i++) {
            auto k = keys[i];
            for (int j = offs[i]; j < offs[i+1]; j++) {
                auto v = values[j];
                a[k] ^=v;
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