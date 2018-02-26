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

//template<class V1, class V2, class V3>
//void ShuffleByIdx(V1 dst, const V2 src, const V3 idx);

template<class T, class TOff>
void ShuffleByIdx(std::vector<T>& p_dst, const std::vector<T> &p_src, const std::vector<TOff> &p_idx) {
    size_t size = p_dst.size();
    assert(size == p_src.size());
    assert(size == p_idx.size());
    auto dst = p_dst.data();
    auto src = p_src.data();
    auto idx = p_idx.data();

    for (size_t i = 0; i < p_src.size(); i++) {
        dst[i] = src[idx[i]];
    }
}

template<class T, class TOff>
void ShuffleByIdx(Data<T> dst, Data<T> src, Data<TOff> idx) {
    struct TaskShuffle : TaskBase {
        Data<T> dst_, src_;
        Data<TOff> idx_;

        TaskShuffle(Engine &engine, Data<T> dst, Data<T> src, Data<TOff> idx) :
                TaskBase(engine, "Shuffle"),
                dst_(dst), src_(src), idx_(idx) {
            assert(dst.size() == src.size());
            assert(dst.size() == idx.size());
            AddInput(src);
            AddInput(idx);
            AddOutput(dst);
        }

        virtual void Run(CPUWorker *cpu) override {
            auto &dst = dst_.WriteAsync(shared_from_this(), cpu->Device(), 0);
            auto &src = src_.ReadAsync(shared_from_this(), cpu->Device(), 0);
            auto &idx = idx_.ReadAsync(shared_from_this(), cpu->Device(), 0);
            for (int i = 0; i < dst_.size(); i++) {
                dst[i] = src[idx[i]];
            }
        }

        virtual void Run(GPUWorker *gpu) override {
            auto &src = src_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream());
            auto &idx = idx_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream());
            auto &dst = dst_.WriteAsync(shared_from_this(), gpu->Device(), gpu->Stream());
            shuffle_by_idx_gpu(dst.data(), src.data(), idx.data(), src_.size(), gpu->Stream());
        }
    };
    Engine::Get().AddTask<TaskShuffle>(dst, src, idx);
}

template<class TKey, class ArrayConstructor = vector_constructor_t>
class DMR {
    template<class T>
    using Vector = decltype(ArrayConstructor::template Construct<T>());

    using TOff = size_t;

private:
    size_t size_;
    Vector<TKey> reducer_keys_;
    Vector<TOff> reducer_offs_;
    Vector<TOff> gather_indices_;

public:
    DMR() {}

    DMR(const std::vector<TKey> &keys) {
        Prepare(keys);
    }

    void Prepare(const std::vector<TKey> &keys) {
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

        reducer_keys_ = Vector<TKey>(std::move(reducer_keys));
        reducer_offs_ = Vector<TOff>(std::move(reducer_offs));
        gather_indices_ = Vector<TOff>(std::move(gather_indices));
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
    Vector<TValue> ShuffleValues(const Vector<TValue>& value_in) const {
        Vector<TValue> value_out(value_in.size());
        ShuffleByIdx(value_out, value_in, gather_indices_);
        return std::move(value_out);
    }

};


#endif //DMR_DMR_H
