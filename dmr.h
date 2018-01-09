//
// Created by chnlkw on 12/29/17.
//

#ifndef DMR_DMR_H
#define DMR_DMR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

//#include "RMA.h"

template<class TKey>
class DMR {
public:
    using TOff = size_t;

    struct ReducerIdx_s {
        std::vector<TKey> keys;
        std::vector<TOff> offs;

        const std::vector<TKey> &Keys() const { return keys; }

        const std::vector<TOff> &Offs() const { return offs; }
    };

    using ReducerIdx = ReducerIdx_s;

private:

    const TKey *keys_;
    size_t count_;
    std::vector<TOff> gather_indices_;

    ReducerIdx reducer_idx_;

public:
    DMR() {
    }

    void SetMapperKeys(const TKey *keys, size_t count) {
//        assert(keys != nullptr);
        keys_ = keys;
        count_ = count;

        std::vector<std::pair<TKey, TOff>> metas(count_);
        for (size_t i = 0; i < count_; i++) {
            metas[i] = {keys[i], i};
        }
        std::sort(metas.begin(), metas.end());

        gather_indices_.resize(count_);

        for (size_t i = 0; i < metas.size(); i++) {
            TKey k = metas[i].first;
            gather_indices_[i] = metas[i].second;

            if (i == 0 || metas[i].first != metas[i - 1].first) {
                reducer_idx_.keys.push_back(k);
                reducer_idx_.offs.push_back(i);
            }
        }
        reducer_idx_.offs.push_back(count_);
        reducer_idx_.keys.shrink_to_fit();
        reducer_idx_.offs.shrink_to_fit();
    }

    template<class TValue>
    struct Shuffler {

    private:
        const DMR *dmr_;
        const TKey *mapper_values_ = nullptr;

        struct Reducer {
            const std::vector<TKey> &keys;
            const std::vector<TOff> &offs;
            std::vector<TValue> values;

            Reducer(const DMR::ReducerIdx &reducer) :
                    keys(reducer.Keys()),
                    offs(reducer.Offs()),
                    values(reducer.Offs().back()) {
            }

            const std::vector<TKey> &Keys() const { return keys; }

            const std::vector<TOff> &Offs() const { return offs; }

            std::vector<TValue> &Values() { return values; }

        };

        Reducer reducer_;

    public:

        Shuffler(const DMR *dmr) :
                dmr_(dmr),
                reducer_(dmr->GetReducerIdx()) {
        }

        void SetMapperValues(const TValue *values, size_t count) {
//            assert(count > 0 && values != nullptr);
            if (count != reducer_.values.size()) {
                throw std::runtime_error("SetMapperValues error, count mismatch");
            }
            mapper_values_ = values;
        }

        void Run() {
            const TOff *idx = dmr_->gather_indices_.data();
            const TValue *src = mapper_values_;
            TValue *dst = reducer_.values.data();
            for (size_t i = 0; i < dmr_->count_; i++) {
                dst[i] = src[idx[i]];
            }
        }

        Reducer &GetReducer() {
            return reducer_;
        }

    };

    const ReducerIdx &GetReducerIdx() const {
        return reducer_idx_;
    }

    template<class TValue>
    Shuffler<TValue> GetShuffler() const {
        return Shuffler<TValue>(this);
    }

};


#endif //DMR_DMR_H
