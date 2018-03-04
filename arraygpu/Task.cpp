//
// Created by chnlkw on 2/28/18.
//

#include "Task.h"

#define LG(x) CLOG(x, "Task")

TaskBase::~TaskBase() {
    LG(INFO) << "Destory " << Name();

}

