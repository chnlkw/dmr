//
// Created by chnlkw on 11/28/17.
//

#include "DataCopy.h"

std::map<int, std::map<int, bool>> data_copy_p2p;

void DataCopy(void *dst_ptr, int dst_device, const void *src_ptr, int src_device, size_t bytes) {
    if (bytes == 0)
        return;
    assert(bytes > 0);
//    printf("data copy %d to %d\n", src_device, dst_device);
    if (src_device < 0) {
        if (dst_device < 0) { //src CPU dst CPU
            memcpy(dst_ptr, src_ptr, bytes);
        } else { // src CPU dst GPU
            CUDA_CALL(cudaMemcpy, dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice);
        }
    } else { // src GPU dst CPU
        if (dst_device < 0) {
            CUDA_CALL(cudaMemcpy, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost);
        } else { // src GPU dst GPU
            CUDA_CALL(cudaMemcpy, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice);
        }
    }
}

void
DataCopyAsync(void *dst_ptr, int dst_device, const void *src_ptr, int src_device, size_t bytes, cudaStream_t stream) {
//                std::cout << dst_device << " <- " << src_device << std::endl;
    if (src_device < 0) {
        if (dst_device < 0) { //src CPU dst CPU
            cudaStreamSynchronize(stream);
            memcpy(dst_ptr, src_ptr, bytes);
        } else { // src CPU dst GPU
            CUDA_CALL(cudaMemcpyAsync, dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream);
        }
    } else { // src GPU dst CPU
        if (dst_device < 0) {
            CUDA_CALL(cudaMemcpyAsync, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream);
        } else { // src GPU dst GPU
            if (data_copy_p2p[src_device][dst_device]) {
                CUDA_CALL(cudaMemcpyPeerAsync, dst_ptr, dst_device, src_ptr, src_device, bytes, stream);
//                std::cout << dst_device << " <- " << src_device << std::endl;
            } else CUDA_CALL(cudaMemcpyAsync, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
        }
    }
}

void DataCopyInitP2P() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            int access;
            cudaDeviceCanAccessPeer(&access, i, j);
            if (access) {
                cudaDeviceEnablePeerAccess(j, 0);
                data_copy_p2p[i][j] = true;
                std::cout << "can p2p " << i << ' ' << j << std::endl;
                CUDA_CHECK();
            }
        }
    }
}

