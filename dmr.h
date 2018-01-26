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
class DMR {
    template<class T>
    using Vector = decltype(ArrayConstructor::template Construct<T>());

private:
    size_t size_;
    Vector<TKey> reducer_keys_;
    Vector<TOff> reducer_offs_;
    Vector<TOff> gather_indices_;

public:
    DMR() {}

    DMR(const Vector<TKey> &keys) {
        Prepare(keys);
    }

    void Prepare(const Vector<TKey> &keys) {
        size_ = keys.size();

        std::vector<std::pair<TKey, TOff>> metas(Size());
        for (size_t i = 0; i < Size(); i++) {
            metas[i] = {keys[i], i};
        }

        std::sort(metas.begin(), metas.end());

        std::vector<TOff> gather_indices(Size());
        std::vector<TKey> reducer_keys;
        std::vector<TOff> reducer_offs;

        gather_indices.resize(Size());

        for (size_t i = 0; i < metas.size(); i++) {
            TKey k = metas[i].first;
            gather_indices[i] = metas[i].second;

            if (i == 0 || metas[i].first != metas[i - 1].first) {
                reducer_keys.push_back(k);
                reducer_offs.push_back(i);
            }
        }
        reducer_offs.push_back(Size());
        reducer_keys.shrink_to_fit();
        reducer_offs.shrink_to_fit();

        reducer_keys_ = std::move(reducer_keys);
        reducer_offs_ = std::move(reducer_offs);
        gather_indices_ = std::move(gather_indices);
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
    Vector<TValue> ShuffleValues(const Vector<TValue> &value_in) const {
        Vector<TValue> value_out = Algorithm::Renew(value_in, value_in.size());
        Algorithm::ShuffleByIdx(
                value_out,
                value_in,
                gather_indices_
        );
        return std::move(value_out);
    }

};


#endif //DMR_DMR_H
