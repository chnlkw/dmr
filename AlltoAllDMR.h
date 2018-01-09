//
// Created by chnlkw on 1/8/18.
//

#ifndef DMR_ALLTOALL_H
#define DMR_ALLTOALL_H

class AlltoAllDMR {
    using TOff = size_t;

    int size_;
    std::vector<std::vector<size_t>> send_counts_;
    std::vector<std::vector<size_t>> recv_counts_;
    std::vector<std::vector<size_t>> send_offs_;
    std::vector<std::vector<size_t>> recv_offs_;
    std::vector<size_t> send_sum_;
    std::vector<size_t> recv_sum_;
public:
    AlltoAllDMR(int size) :
            size_(size),
            send_counts_(size),
            recv_counts_(size),
            send_sum_(size),
            recv_sum_(size) {
    }

    int Size() const { return size_; }

    void SetCounts(size_t rank, const size_t *count) {
        send_counts_[rank].assign(count, count + size_);
        send_sum_[rank] = std::accumulate(count, count + size_, 0LU);
    }

    void Prepare() {
        for (size_t i = 0; i < size_; i++) {
            recv_counts_[i].resize(size_);
            for (size_t j = 0; j < size_; j++) {
                recv_counts_[i][j] = send_counts_[j][i];
                recv_sum_[i] += recv_counts_[i][j];
            }
        }
        recv_offs_.resize(size_);
        send_offs_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            recv_offs_[i].resize(size_);
            std::partial_sum(recv_counts_[i].begin(), recv_counts_[i].end() - 1, recv_offs_[i].begin() + 1);
            send_offs_[i].resize(size_);
            std::partial_sum(send_counts_[i].begin(), send_counts_[i].end() - 1, send_offs_[i].begin() + 1);
        }
    }

    template<class TValue>
    struct Shuffler {

    private:
        const AlltoAllDMR *dmr_;
        std::vector<std::pair<const TValue *, size_t>> mapper_values_;

        std::vector<std::vector<TValue>> reducer_values_;

    public:

        Shuffler(const AlltoAllDMR *dmr) :
                dmr_(dmr),
                mapper_values_(dmr->size_),
                reducer_values_(dmr->size_) {
        }

        void SetMapperValues(size_t mapper_id, const TValue *values, size_t count) {
            if (count != dmr_->send_sum_[mapper_id]) {
                throw std::runtime_error("SetMapperValues error, count mismatch");
            }
            mapper_values_[mapper_id] = {values, count};
        }

        void Run() {
            for (size_t i = 0; i < dmr_->size_; i++) {
                reducer_values_[i].resize(dmr_->recv_sum_[i]);
                for (size_t j = 0; j < dmr_->size_; j++) {
                    auto src_beg = mapper_values_[j].first + dmr_->send_offs_[j][i];
                    auto src_end = src_beg + dmr_->send_counts_[j][i];
                    auto dst_beg = reducer_values_[i].data() + dmr_->recv_offs_[i][j];
                    std::copy(src_beg, src_end, dst_beg);
                }
            }
        }

        std::vector<TValue> &Value(size_t reducer_id) {
            return reducer_values_[reducer_id];
        }

    };

    template<class TValue>
    Shuffler<TValue> GetShuffler() const {
        return Shuffler<TValue>(this);
    }
};

#endif //DMR_ALLTOALL_H
