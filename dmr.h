//
// Created by chnlkw on 12/29/17.
//

#ifndef DMR_DMR_H
#define DMR_DMR_H

#include <iostream>
#include <vector>
#include <algorithm>

//#include "RMA.h"

template<class TIdx>
class DMR {
    using TOff = size_t;
private:
    size_t num_mappers_;
    size_t num_reducers_;

    std::vector<std::pair<const TIdx *, size_t>> mapper_keys_;
    std::vector<std::vector<std::pair<size_t, size_t>>> shuffle_indices_;

    struct Reducer {
        std::vector<TIdx> keys;
        std::vector<TOff> offs;

        const std::vector<TIdx> &Keys() const { return keys; }

        const std::vector<TOff> &Offs() const { return offs; }
    };

    std::vector<Reducer> reducers_;

public:
    DMR(size_t num_mappers, size_t num_reducers) :
            num_mappers_(num_mappers),
            num_reducers_(num_reducers),
            mapper_keys_(num_mappers),
            reducers_(num_reducers),
            shuffle_indices_(num_mappers) {

    }

    void SetMapperKeys(size_t mapper_id, const TIdx *keys, size_t count) {
        mapper_keys_[mapper_id] = {keys, count};
        shuffle_indices_[mapper_id].resize(count);
    }

    void Prepare() {
        auto num_reducers = num_reducers_;
        auto idx_to_reducer = [num_reducers](TIdx idx) { return idx % num_reducers; };
        std::vector<std::vector<std::pair<TIdx, std::pair<size_t, size_t>>>> metas(num_reducers_);

        for (size_t mapper_id = 0; mapper_id < num_mappers_; mapper_id++) {
            auto &keys = mapper_keys_[mapper_id];
            for (size_t i = 0; i < keys.second; i++) {
                TIdx idx = keys.first[i];
                size_t reducer_id = idx_to_reducer(idx);
                metas[reducer_id].push_back({idx, {mapper_id, i}});
//                fprintf(stderr, "k=%d mapped to reducer %d\n", idx, reducer_id);
            }
        }
        for (size_t reducer_id = 0; reducer_id < num_reducers; reducer_id++) {
            std::sort(metas[reducer_id].begin(), metas[reducer_id].end());
        }

        reducers_.resize(num_reducers);
        for (size_t reducer_id = 0; reducer_id < num_reducers; reducer_id++) {
            auto &reducer = reducers_[reducer_id];
            TIdx last_key = -1;
            for (size_t i = 0; i < metas[reducer_id].size(); i++) {
                TIdx k = metas[reducer_id][i].first;
                auto mapper_pos = metas[reducer_id][i].second;
                shuffle_indices_[mapper_pos.first][mapper_pos.second] = {reducer_id, i};

                if (i == 0 || k != last_key) {
                    reducer.keys.push_back(k);
                    reducer.offs.push_back(i);
                    last_key = k;
                }
            }
            reducer.offs.push_back(metas[reducer_id].size());
//            fprintf(stderr, "reducer %d : keys = %d offs = %d size = %d\n", reducer_id, reducer.keys.size(),
//                    reducer.offs.size(), metas[reducer_id].size());
        }
    }

    template<class TValue>
    struct Shuffler {

    private:
        const DMR *dmr_;
        std::vector<std::pair<const TIdx *, size_t>> mapper_values_;

        struct Reducer {
            const std::vector<TIdx> &keys;
            const std::vector<TOff> &offs;
            std::vector<TValue> values;

            Reducer(const DMR::Reducer &reducer) :
                    keys(reducer.Keys()),
                    offs(reducer.Offs()),
                    values(reducer.Offs().back()) {
            }

            const std::vector<TIdx> &Keys() const { return keys; }

            const std::vector<TOff> &Offs() const { return offs; }

            std::vector<TValue> &Values() { return values; }

        };

        std::vector<Reducer> reducers_;

    public:

        Shuffler(const DMR *dmr) :
                dmr_(dmr),
                mapper_values_(dmr->num_mappers_) {
            for (const DMR::Reducer &r : dmr_->reducers_) {
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
            for (size_t mapper_id = 0; mapper_id < dmr_->num_mappers_; mapper_id++) {
                const TValue *values = mapper_values_[mapper_id].first;
                size_t count = mapper_values_[mapper_id].second;
                for (size_t i = 0; i < count; i++) {
                    size_t reducer_id = dmr_->shuffle_indices_[mapper_id][i].first;
                    size_t reducer_off = dmr_->shuffle_indices_[mapper_id][i].second;
                    reducers_[reducer_id].Values()[reducer_off] = values[i];
                }
            }
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


#endif //DMR_DMR_H
