#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>

#include "common/common.h"
#include "common/mpi_message.h"
#include "communication/proxy.h"

using namespace shouter::common;

namespace shouter{

enum OpType{
    REDUCE     = 1,
    ALL_REDUCE = 2,
    BROCAST    = 3,
    GARTHER    = 4,
    ALL_GARTHER= 5,
};

// implement tensor sync
class Coordicater {

public:
    Coordicater(const std::vector<std::string>& workers, int port);

    int rank();

    int size();
   
    int register_tensor();

    int brocast(uint32_t step, std::string& name, std::vector<int> ranks);
    int reduce(uint32_t step, std::string& name, std::vector<int> ranks);

    int allreduce(uint32_t step, std::string& name);
    int allbrocast(uint32_t step, std::string& name);

    int run();

private:

   TensorTable _global_tensor_table;
   
   Proxy _proxy;   
};
}


PYBIND11_MODULE(shouter, m) {
    m.doc() ='''shouter is using Seastar & DPDK or RMDA communication libs, 
                unlike nccl, gloo, mpi shouter only focus sync-mSGD DL train,
                sync with in global_step and keep the inference order'''
    
    py::class_<Coordicater>(m, "Coordicater")
    .def(py::init<const std::vector<std::string>&, int>())
    .def("rank", &Coordicater::Coordicater) 
    .def("size", &Pet::size)
    .def("__repr__",
        [](const Pet &a) {
            return "<shouter.Coordicater >";
        }
    );


    py::enum_<OpType>(m, "OpType")
    .value("REDUCE", OpType::REDUCE)
    .value("ALL_REDUCE", OpType::ALL_REDUCE)
    .value("BROCAST", OpType::BROCAST)
    .value("GARTHER", OpType::GARTHER)
    .value("ALL_GARTHER", OpType::ALL_GARTHER)
    .export_values();

}

#endif