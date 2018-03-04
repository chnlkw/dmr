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
    std::vector<DMR<TPar, ArrayConstructor>> dmr1_;
    AlltoAllDMR alltoall_;
    std::vector<DMR<TKey, ArrayConstructor>> dmr3_;

    using Partitioner = std::function<int(TKey)>;
    Partitioner partitioner_;
public:

    explicit PartitionedDMR(const std::vector<std::vector<TKey>> &mapper_keys, Partitioner partitioner) :
            size_(mapper_keys.size()),
            dmr1_(size_),
            dmr3_(size_),
            max_key_(0),
            partitioner_(partitioner) {
        Prepare(mapper_keys);
    }

    void Prepare(const std::vector<std::vector<TKey>> &mapper_keys) {

        std::vector<std::vector<TKey>> parted_keys;
        // local partition
        for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
            auto keys = mapper_keys[mapper_id];
            std::vector<TPar> par_id(keys.size());
            std::transform(keys.begin(), keys.end(), par_id.begin(), partitioner_);
            DMR<TPar> dmr(par_id);
            auto parted_key = dmr.ShuffleValues<TKey>(keys);
            parted_keys.push_back(parted_key);
            dmr1_[mapper_id] = std::move(dmr);
        }

        // global partition
        std::vector<std::vector<size_t>> send_counts(size_);
        for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
            auto &dmr = dmr1_[mapper_id];
            std::vector<size_t> counts(size_);
            for (size_t i = 0; i < dmr.Keys().size(); i++) {
                TKey k = dmr.Keys()[i];
                counts.at(k) = dmr.Offs()[i + 1] - dmr.Offs()[i];
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

    template<class Cons>
    PartitionedDMR(const PartitionedDMR<TKey, Cons> &that) :
            size_(that.Size()),
            max_key_(that.MaxKey()),
            alltoall_(that.GetAlltoallDMR()) {
        for (auto &d : that.GetDMR1())
            dmr1_.push_back(d);
        for (auto &d : that.GetDMR3())
            dmr3_.push_back(d);
    }

    template<class TValue>
    std::vector<Vector<TValue>> ShuffleValues(const std::vector<Vector<TValue>> &value_in) const {
        std::vector<Vector<TValue>> parted_values;
        for (size_t i = 0; i < size_; i++) {
            parted_values.push_back(dmr1_[i].template ShuffleValues<TValue>(value_in[i]));
        }

        auto shufed = alltoall_.ShuffleValues(parted_values);

        std::vector<Vector<TValue>> ret;
        for (size_t i = 0; i < size_; i++) {
            ret.push_back(dmr3_[i].template ShuffleValues<TValue>(shufed[i]));
        }

        return std::move(ret);
    }

    const Vector<TKey> &Keys(size_t i) const { return dmr3_[i].Keys(); }

    const Vector<TOff> &Offs(size_t i) const { return dmr3_[i].Offs(); }

    size_t Size() const {
        return size_;
    }

    TKey MaxKey() const {
        return max_key_;
    }

    const std::vector<DMR<TPar>> &GetDMR1() const { return dmr1_; }

    const AlltoAllDMR &GetAlltoallDMR() const { return alltoall_; }

    const std::vector<DMR<TKey>> &GetDMR3() const { return dmr3_; }
};


#endif //DMR_PARTITIONEDDMR_H
