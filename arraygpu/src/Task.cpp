//
// Created by chnlkw on 2/28/18.
//

#include <functional>
#include "Task.h"
#include "Worker.h"
#include "Data.h"

#define LG(x) CLOG(x, "Task")

TaskBase::~TaskBase() {
    LG(INFO) << "Destory " << Name();
}

void TaskBase::PrepareData(DevicePtr dev, cudaStream_t stream) {
    for (auto &m : GetMetas()) {
        if (m.is_read_only) {
            m.data->ReadAsync(shared_from_this(), dev, stream);
        } else {
            m.data->WriteAsync(shared_from_this(), dev, stream);
        }
    }
}

