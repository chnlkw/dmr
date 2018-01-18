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
#include "array_constructor.h"
#include "Algorithm.h"

using TOff = size_t;

template<class TKey, class ArrayConstructor = vector_constructor_t>
struct ReducerIdx_s {
    template<class T>
    using Vector = decltype(ArrayConstructor::template Construct<T>());

    size_t value_size_;
    Vector<TKey> keys;
    Vector<TOff> offs;

    const Vector<TKey> &Keys() const { return keys; }

    const Vector<TOff> &Offs() const { return offs; }

    ReducerIdx_s() {
    }

    template<class Cons>
    ReducerIdx_s(const ReducerIdx_s<TKey, Cons> &that) :
            keys(that.Keys()),
            offs(that.Offs()),
            value_size_(that.value_size_) {
    }

};

template<class TKey, class ArrayConstructor = vector_constructor_t>
class DMR {
    template<class T>
    using Vector = decltype(ArrayConstructor::template Construct<T>());

private:
    size_t size_;
    Vector<TKey> reducer_keys_;
    Vector<TOff> reducer_offs_;
    Vector<TOff> gather_indices_;

    void Prepare(const Vector<TKey> &keys) {
        Vector<std::pair<TKey, TOff>> metas(Size());
        for (size_t i = 0; i < Size(); i++) {
            metas[i] = {keys[i], i};
        }
        std::sort(metas.begin(), metas.end());

        gather_indices_.resize(Size());

        for (size_t i = 0; i < metas.size(); i++) {
            TKey k = metas[i].first;
            gather_indices_[i] = metas[i].second;

            if (i == 0 || metas[i].first != metas[i - 1].first) {
                reducer_keys_.push_back(k);
                reducer_offs_.push_back(i);
            }
        }
        reducer_offs_.push_back(Size());
        reducer_keys_.shrink_to_fit();
        reducer_offs_.shrink_to_fit();
    }

public:
    DMR(const Vector<TKey> &keys) :
            size_(keys.size()) {
        Prepare(keys);
    }

    template<class Cons>
    DMR(const DMR<TKey, Cons> &that) :
            size_(that.Size()),
            reducer_keys_(that.Keys()),
            reducer_offs_(that.Offs()),
            gather_indices_(that.GatherIndices()) {
    }

    size_t Size() const {
        return size_;
    }

    const Vector<TOff> &GatherIndices() const {
        return gather_indices_;
    }

    const Vector<TKey> &Keys() const {
        return reducer_keys_;
    }

    const Vector<TOff> &Offs() const {
        return reducer_offs_;
    }

    template<class TValue>
    struct Shuffler {
    private:
        const DMR *dmr_;
        Vector<TValue> reducer_values_;

    public:
        Shuffler(const DMR *dmr, const Vector<TValue> &mapper_values) :
                dmr_(dmr),
                reducer_values_(dmr_->Size()) {
            assert(mapper_values.size() == reducer_values_.size());
            Run(mapper_values);
        }

        const Vector<TKey> &Keys() const { return dmr_->ReducerKeys(); }

        const Vector<TOff> &Offs() const { return dmr_->ReducerOffs(); }

        Vector<TValue> &Values() { return reducer_values_; }

    private:
        void Run(const Vector<TValue> &mapper_values) {
            Algorithm::ShuffleByIdx(
                    reducer_values_,
                    mapper_values,
                    dmr_->GatherIndices()
            );

//            const TOff *idx = dmr_->GatherIndices().data();
//            const TValue *src = mapper_values.data();
//            TValue *dst = reducer_values_.data();
//            for (size_t i = 0; i < dmr_->Size(); i++) {
//                dst[i] = src[idx[i]];
//            }
        }

    };

    template<class TValue>
    Shuffler<TValue> GetShuffler(const Vector<TValue> &values) const {
        return Shuffler<TValue>(this, values);
    }

};


#endif //DMR_DMR_H
