//
// Created by chnlkw on 3/13/18.
//

#ifndef DMR_CAR_H
#define DMR_CAR_H

#include "defs.h"

class Car {
private:
    static std::shared_ptr<Engine> engine;

public:
    static void Set(std::shared_ptr<Engine> e) { engine = e; }

    static Engine &Get();

    static DevicePtr GetCPUDevice();

    static void Finish() { engine.reset(); }
};


#endif //DMR_CAR_H
