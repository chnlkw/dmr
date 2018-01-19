//
// Created by chnlkw on 1/5/18.
//

#ifndef DMR_PARTITIONEDDMR_H
#define DMR_PARTITIONEDDMR_H

//#include <range/v3/all.hpp>
#include "dmr.h"
#include "array_constructor.h"
#include "AlltoAllDMR.h"

#include <algorithm>

//auto max = [](auto a, auto b) { return std::max(a, b); };
auto f_key_neq = [](auto a, auto b) { return a.first != b.first; };
auto f_key_less = [](auto a, auto b) { return a.first < b.first; };

//using namespace ranges;

template<class TKey, class ArrayConstructor = vector_constructor_t>
class PartitionedDMR {
    template<class T>
    using Vector = decltype(ArrayConstructor::template Construct<T>());
public:
    using TPar = uint32_t;
    using TOff = size_t;

    struct Reducer {
        Vector<TKey> keys;
        Vector<TOff> offs;

    };

private:
    size_t size_;
    TKey max_key_;
    std::vector<DMR<TPar>> dmr1_;
    AlltoAllDMR alltoall_;
    std::vector<DMR<TKey>> dmr3_;
public:

    explicit PartitionedDMR(const std::vector<Vector<TKey>> &mapper_keys) :
            size_(mapper_keys.size()),
            dmr1_(size_),
            dmr3_(size_),
            max_key_(0) {
        Prepare(mapper_keys);
    }

    void Prepare(const std::vector<Vector<TKey>> &mapper_keys) {
        max_key_ = 0;
        for (auto &v : mapper_keys)
            max_key_ = std::accumulate(v.begin(), v.end(), max_key_, [](auto a, auto b) { return std::max(a, b); });

        // init partitioner
        size_t key_partition_size = (max_key_ + size_) / size_;
        auto partitioner = [key_partition_size](TKey k) -> TPar { return k / key_partition_size; };

        std::vector<Vector<TKey>> parted_keys;
        // local partition
        for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
            auto &keys = mapper_keys[mapper_id];
            Vector<TPar> par_id(size_);
            std::transform(keys.begin(), keys.end(), par_id.begin(), partitioner);
            DMR<TPar> dmr(par_id);
            parted_keys.push_back(dmr.ShuffleValues<TKey>(keys));
            dmr1_[mapper_id] = std::move(dmr);
        }

        // global partition
        std::vector<std::vector<size_t>> send_counts(size_);
        for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
            auto &dmr = dmr1_[mapper_id];
            Vector<size_t> counts(size_);
            for (size_t i = 0; i < dmr.Keys().size(); i++) {
                TKey k = dmr.Keys()[i];
                counts[k] = dmr.Offs()[i + 1] - dmr.Offs()[i];
            }
            send_counts[mapper_id] = std::move(counts);
        }
        alltoall_.Prepare(send_counts);
        auto results = alltoall_.ShuffleValues(parted_keys);

        // local sort
        for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
            dmr3_[mapper_id].Prepare(results[mapper_id]);
        }
    }

    template<class Vec>
    std::vector<Vec> ShuffleValues(const std::vector<Vec> &value_in) const {
        using TValue = typename Vec::value_type;
        std::vector<Vec> parted_values;
        for (size_t i = 0; i < size_; i++) {
            parted_values.push_back(dmr1_[i].ShuffleValues<TValue>(value_in[i]));
        }

        std::vector<Vec> shufed = alltoall_.ShuffleValues(parted_values);

        std::vector<Vec> ret;
        for (size_t i = 0; i < size_; i++) {
            ret.push_back(dmr3_[i].ShuffleValues<TValue>(shufed[i]));
        }

        return std::move(ret);
    }

    const Vector<TKey> &Keys(size_t i) const { return dmr3_[i].Keys(); }

    const Vector<TOff> &Offs(size_t i) const { return dmr3_[i].Offs(); }

    size_t Size() const {
        return size_;
    }
};


#endif //DMR_PARTITIONEDDMR_H
