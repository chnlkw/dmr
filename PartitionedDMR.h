//
// Created by chnlkw on 1/5/18.
//

#ifndef DMR_PARTITIONEDDMR_H
#define DMR_PARTITIONEDDMR_H

#include "range/v3/all.hpp"

//auto max = [](auto a, auto b) { return std::max(a, b); };
auto f_key_neq = [](auto a, auto b) { return a.first != b.first; };

using namespace ranges;

template<class TKey>
class PartitionedDMR {
    using TOff = size_t;
private:
    size_t num_mappers_;
    size_t num_reducers_;

    TKey max_key_;
    std::vector<std::pair<const TKey *, size_t>> mapper_keys_;
    std::vector<std::vector<size_t>> shuffle_pos_1;
    std::vector<std::vector<size_t>> shuffle_pos_2;

    struct Reducer {
        std::vector<TKey> keys;
        std::vector<TOff> offs;

        const std::vector<TKey> &Keys() const { return keys; }

        const std::vector<TOff> &Offs() const { return offs; }
    };

    std::vector<Reducer> reducers_;

public:
    PartitionedDMR(size_t num_mappers, size_t num_reducers) :
            num_mappers_(num_mappers),
            num_reducers_(num_reducers),
            mapper_keys_(num_mappers),
            reducers_(num_reducers),
            shuffle_pos_1(num_mappers),
            shuffle_pos_2(num_mappers),
            max_key_(0) {

    }

    void SetMapperKeys(size_t mapper_id, const TKey *keys, size_t count) {
        mapper_keys_[mapper_id] = {keys, count};
        max_key_ = accumulate(view::counted(keys, count), max_key_, max);
    }

    void Prepare() {
        TKey key_partition_size = (max_key_ + 1 + num_mappers_ - 1) / num_mappers_;
        auto partitioner = [key_partition_size](TKey k) { return k / key_partition_size; };

        for (size_t mapper_id = 0; mapper_id < num_mappers_; mapper_id++) {
            auto &m = mapper_keys_[mapper_id];
            std::vector<TKey> k(m.first, m.first + m.second);

            auto &pos1 = shuffle_pos_1[mapper_id];
            pos1 = view::ints(0LU, k.size());

            std::vector<size_t> par_id = view::transform(k, partitioner);

            sort(view::zip(par_id, view::zip(k, pos1)));

            auto &pos2 = shuffle_pos_2[mapper_id];
            pos2 = view::zip(par_id, view::ints)
                   | view::adjacent_filter(f_key_neq)
                   | view::values;
            pos2.push_back(k.size());
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

            Reducer(const PartitionedDMR::Reducer &reducer) :
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
            for (const PartitionedDMR::Reducer &r : dmr_->reducers_) {
                reducers_.emplace_back(r);
            }
        }

        void SetMapperValues(size_t mapper_id, const TValue *values, size_t count) {
            if (count != dmr_->mapper_keys_[mapper_id].second) {
                throw std::runtime_error("SetMapperValues error, count mismatch");
            }
            mapper_values_[mapper_id] = {values, count};
        }

        void Run() {
//            for (size_t mapper_id = 0; mapper_id < dmr_->num_mappers_; mapper_id++) {
//                const TValue *values = mapper_values_[mapper_id].first;
//                size_t count = mapper_values_[mapper_id].second;
//                for (size_t i = 0; i < count; i++) {
//                    size_t reducer_id = dmr_->shuffle_indices_[mapper_id][i].first;
//                    size_t reducer_off = dmr_->shuffle_indices_[mapper_id][i].second;
//                    reducers_[reducer_id].Values()[reducer_off] = values[i];
//                }
//            }
        }

        std::vector<Reducer> &Reducers() {
            return reducers_;
        }

    };

    template<class TValue>
    Shuffler<TValue> GetShuffler() {
        return Shuffler<TValue>(this);
    }

};


#endif //DMR_PARTITIONEDDMR_H
