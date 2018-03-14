//
// Created by chnlkw on 3/13/18.
//

#include "Car.h"
#include "Engine.h"

DevicePtr Car::GetCPUDevice() { return engine->CpuDevice(); }

Engine &Car::Get() { return *engine; }

TaskBase &Car::AddTask(TaskPtr task) {
    return engine->AddTask(task);
}

std::shared_ptr<Engine> Car::engine;
