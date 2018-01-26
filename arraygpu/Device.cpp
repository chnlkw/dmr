//
// Created by chnlkw on 1/25/18.
//

#include "Device.h"

DevicePtr Device::cpu(new CPUDevice);
DevicePtr Device::current = Device::cpu;
