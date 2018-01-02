//
// Created by chnlkw on 2018/1/2.
//

#ifndef DMR_RMA_H
#define DMR_RMA_H


class RemoteMemory {
    int rank;
    int tag;
    size_t size;

public:
    void Put(void* local_addr, size_t size, off_t remote_off);

    void Get(void* local_addr, size_t size, off_t remote_off);
};

class RMAEngine {
    int size_;
    int rank_;
public:
    RMAEngine(int worker_size, int worker_rank);

    int Alloc(size_t size);

    void Free(int tag);

};
#endif //DMR_RMA_H
