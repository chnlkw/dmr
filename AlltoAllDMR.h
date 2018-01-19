//
// Created by chnlkw on 1/8/18.
//

#ifndef DMR_ALLTOALL_H
#define DMR_ALLTOALL_H

class AlltoAllDMR {
    int size_;
    std::vector<std::vector<size_t>> send_counts_;
    std::vector<std::vector<size_t>> recv_counts_;
    std::vector<std::vector<size_t>> send_offs_;
    std::vector<std::vector<size_t>> recv_offs_;
    std::vector<size_t> send_sum_;
    std::vector<size_t> recv_sum_;

public:
    AlltoAllDMR() : size_(0) {}

    AlltoAllDMR(const std::vector<std::vector<size_t>> &counts) {
        Prepare(counts);
    }

    int Size() const { return size_; }

    void Prepare(const std::vector<std::vector<size_t>> &counts) {
        size_ = counts.size();
        send_sum_.resize(size_);
        recv_sum_.resize(size_);
        send_counts_ = counts;

        for (size_t i = 0; i < size_; i++) {
            send_sum_[i] = std::accumulate(send_counts_[i].begin(), send_counts_[i].end(), 0LU);
        }

        recv_counts_.resize(size_);
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

    template<class Vec>
    std::vector<Vec> ShuffleValues(const std::vector<Vec> &value_in) const {
        std::vector<Vec> value_out;
        for (size_t i = 0; i < size_; i++) {
            assert(value_in[i].size() == send_sum_[i]);
            value_out.push_back(Algorithm::Renew(value_in[i], recv_sum_[i]));
        }
        for (size_t i = 0; i < size_; i++) {
            for (size_t j = 0; j < size_; j++) {
                auto src_beg = value_in[j].data() + send_offs_[j][i];
                auto src_end = src_beg + send_counts_[j][i];
                auto dst_beg = value_out[i].data() + recv_offs_[i][j];
                std::copy(src_beg, src_end, dst_beg);
            }
        }
        return std::move(value_out);
    }
};

#endif //DMR_ALLTOALL_H
