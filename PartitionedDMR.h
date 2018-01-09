//
// Created by chnlkw on 1/5/18.
//

#ifndef DMR_PARTITIONEDDMR_H
#define DMR_PARTITIONEDDMR_H

//#include <range/v3/all.hpp>
#include "dmr.h"
#include "AlltoAllDMR.h"

#include <algorithm>

//auto max = [](auto a, auto b) { return std::max(a, b); };
auto f_key_neq = [](auto a, auto b) { return a.first != b.first; };
auto f_key_less = [](auto a, auto b) { return a.first < b.first; };

//using namespace ranges;

template<class TKey>
class PartitionedDMR {
public:
    using TPar = uint32_t;
    using TOff = size_t;

    struct Reducer {
        std::vector<TKey> keys;
        std::vector<TOff> offs;

        const std::vector<TKey> &Keys() const { return keys; }

        const std::vector<TOff> &Offs() const { return offs; }
    };

private:
    size_t num_mappers_;
    std::vector<std::pair<const TKey *, size_t>> mapper_keys_;
    TKey max_key_;
    std::vector<DMR<TPar>> dmr1;
    AlltoAllDMR alltoall_;
    std::vector<DMR<TKey>> dmr3;
public:

    PartitionedDMR(size_t num_mappers) :
            num_mappers_(num_mappers),
            mapper_keys_(num_mappers),
            dmr1(num_mappers),
            alltoall_(num_mappers),
            dmr3(num_mappers),
            max_key_(0) {
    }

    void SetMapperKeys(size_t mapper_id, const TKey *keys, size_t count) {
        mapper_keys_[mapper_id] = {keys, count};
        for (size_t i = 0; i < count; i++)
            max_key_ = std::max(max_key_, keys[i]);
    }

    void Prepare() {

        // init partitioner
        size_t key_partition_size = (max_key_ + num_mappers_) / num_mappers_;
        auto partitioner = [key_partition_size](TKey k) -> TPar { return k / key_partition_size; };

        // local partition
        std::vector<std::vector<TKey>> parted_keys(num_mappers_);
        for (size_t mapper_id = 0; mapper_id < num_mappers_; mapper_id++) {
            auto keys = mapper_keys_[mapper_id].first;
            size_t count = mapper_keys_[mapper_id].second;

            std::vector<TPar> par_id(count);
            std::transform(keys, keys + count, par_id.begin(), partitioner);

            auto &dmr = dmr1[mapper_id];
            dmr.SetMapperKeys(par_id.data(), par_id.size());

            auto shuf = dmr.template GetShuffler<TKey>();
            shuf.SetMapperValues(keys, count);
            shuf.Run();
            parted_keys[mapper_id] = std::move(shuf.GetReducer().Values());
        }

        // global partition
        for (size_t mapper_id = 0; mapper_id < num_mappers_; mapper_id++) {
            auto &red = dmr1[mapper_id].GetReducerIdx();
            std::vector<size_t> counts(num_mappers_);
            for (size_t i = 0; i < red.Keys().size(); i++) {
                counts[red.Keys()[i]] = red.Offs()[i + 1] - red.Offs()[i];
            }
            alltoall_.SetCounts(mapper_id, counts.data());
        }
        alltoall_.Prepare();
        auto shuf2 = alltoall_.template GetShuffler<TKey>();
        for (size_t mapper_id = 0; mapper_id < num_mappers_; mapper_id++) {
            auto &keys = parted_keys[mapper_id];
            shuf2.SetMapperValues(mapper_id, keys.data(), keys.size());
        }
        shuf2.Run();
        // local sort
        for (size_t mapper_id = 0; mapper_id < num_mappers_; mapper_id++) {
            auto keys = shuf2.Value(mapper_id);

            auto &dmr = dmr3[mapper_id];
            dmr.SetMapperKeys(keys.data(), keys.size());
        }
    }

    template<class TValue>
    struct Shuffler {

    private:
        const PartitionedDMR *dmr_;
        std::vector<std::pair<const TKey *, size_t>> mapper_values_;

        struct Reducer {
            const std::vector<TKey> &keys;
            const std::vector<TOff> &offs;
            std::vector<TValue> values;

            Reducer(const typename ::DMR<TKey>::ReducerIdx &reducer) :
                    keys(reducer.Keys()),
                    offs(reducer.Offs()),
                    values(reducer.Offs().back()) {
            }

            const std::vector<TKey> &Keys() const { return keys; }

            const std::vector<TOff> &Offs() const { return offs; }

            std::vector<TValue> &Values() { return values; }

        };

        std::vector<Reducer> reducers_;

    public:

        Shuffler(const PartitionedDMR *dmr) :
                dmr_(dmr),
                mapper_values_(dmr->num_mappers_) {
            for (auto &d : dmr_->dmr3) {
                reducers_.emplace_back(d.GetReducerIdx());
            }
        }

        void SetMapperValues(size_t mapper_id, const TValue *values, size_t count) {
            if (count != dmr_->mapper_keys_[mapper_id].second) {
                throw std::runtime_error("SetMapperValues error, count mismatch");
            }
            mapper_values_[mapper_id] = {values, count};
        }

        void Run() {
            size_t size = dmr_->num_mappers_;
            std::vector<std::vector<TKey>> parted_values(size);
            for (size_t mapper_id = 0; mapper_id < size; mapper_id++) {
                auto shuf = dmr_->dmr1[mapper_id].template GetShuffler<TValue>();
                shuf.SetMapperValues(mapper_values_[mapper_id].first, mapper_values_[mapper_id].second);
                shuf.Run();
                parted_values[mapper_id] = std::move(shuf.GetReducer().Values());
            }
            auto shuf2 = dmr_->alltoall_.template GetShuffler<TValue>();
            for (size_t mapper_id = 0; mapper_id < size; mapper_id++) {
                auto &values = parted_values[mapper_id];
                shuf2.SetMapperValues(mapper_id, values.data(), values.size());
            }
            shuf2.Run();

            for (size_t mapper_id = 0; mapper_id < size; mapper_id++) {
                auto &values = shuf2.Value(mapper_id);
                auto &dmr = dmr_->dmr3[mapper_id];

                auto shuf3 = dmr.template GetShuffler<TValue>();
                shuf3.SetMapperValues(values.data(), values.size());
                shuf3.Run();
                reducers_[mapper_id].Values() = std::move(shuf3.GetReducer().Values());
            }
//                const TValue *values = mapper_values_[mapper_id].first;
//                size_t count = mapper_values_[mapper_id].second;
//                for (size_t i = 0; i < count; i++) {
//                    size_t reducer_id = dmr_->shuffle_indices_[mapper_id][i].first;
//                    size_t reducer_off = dmr_->shuffle_indices_[mapper_id][i].second;
//                    reducers_[reducer_id].Values()[reducer_off] = values[i];
//                }
//            }
        }

        std::vector<Reducer> &GetReducer() {
            return reducers_;
        }

    };

    template<class TValue>
    Shuffler<TValue> GetShuffler() {
        return Shuffler<TValue>(this);
    }

};


#endif //DMR_PARTITIONEDDMR_H
