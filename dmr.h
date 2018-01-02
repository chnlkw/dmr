//
// Created by chnlkw on 12/29/17.
//

#ifndef DMR_DMR_H
#define DMR_DMR_H

#include <utility>

class DMR {
    using TIdx = size_t;

private:
    int num_workers_;
    int my_id_;

public:
    DMR(int num_workers, int my_id) :
            num_workers_(num_workers),
            my_id_(my_id) {
    }

    void SetInputKeys(const TIdx *keys, size_t count);

    void Prepare();

    std::pair<const TIdx *, size_t> GetOutputKeys();

    void ShuffleValue(void *values_out, const void *values_in, size_t value_size);

    template<class T>
    void ShuffleValue(T *values_out, const T *values_in) {
        ShuffleValue(values_out, values_in, sizeof(T));
    }

    virtual void Send(void *send_addr, size_t bytes, int dest);

    virtual void Recv(void *recv_addr, size_t bytes, int source);

    virtual void *Alloc(size_t bytes);

    virtual void Free(void *ptr);
};


#endif //DMR_DMR_H
