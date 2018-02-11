//
// Created by chnlkw on 11/21/17.
//

#ifndef LDA_DEFS_H
#define LDA_DEFS_H

#include <memory>

class ArrayBase;

template<class T>
class Array;

class AllocatorBase;

class DeviceBase;

class Node;

class DataBase;

template<class T>
class Data;

class TaskBase;

class WorkerBase;

class CPUWorker;

class GPUWorker;

class Engine;

using ArrayBasePtr = std::shared_ptr<ArrayBase>;
template<class T>
using ArrayPtr = std::shared_ptr<Array<T>>;
using AllocatorPtr = std::shared_ptr<AllocatorBase>;
using DevicePtr = std::shared_ptr<DeviceBase>;
using NodePtr = std::shared_ptr<Node>;
using DataBasePtr = std::shared_ptr<DataBase>;
//template<class T> using DataPtr = std::shared_ptr<Data<T>>;
using TaskPtr = std::shared_ptr<TaskBase>;
using WorkerPtr = std::shared_ptr<WorkerBase>;

#endif //LDA_DEFS_H
