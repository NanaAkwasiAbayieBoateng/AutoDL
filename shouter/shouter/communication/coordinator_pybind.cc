#include <pybind11/pybind11.h>
#include <memory>
#include <vector>
#include <sstream>

namespace py = pybind11;


#include "communication/coordinator.h"

using namespace shouter;


PYBIND11_MODULE(Coordicater, m) {
    m.doc() ='''Coordicater is using Seastar & DPDK or RMDA communication libs, 
                unlike nccl, gloo, mpi shouter only focus sync-mSGD DL train,
                sync with in global_step and keep the inference order'''
    
    py::class_<Coordicater>(m, "Coordicater")
    .def(py::init<const std::vector<std::string>&, int>())
    .def("rank", &Coordicater::rank) 
    .def("size", &Coordicater::size)
    .def("register_reduce", &Coordicater::register_reduce)
    .def("register_allreduce", &Coordicater::register_reduce)
    .def("__repr__",
        [](const Coordicater &a) {
            std::stringstream str;
            str << "<shouter.Coordicater rank:"<<a.rank()<<", addr:"<<a.addr() <<">";
            return str.str();
        }
    );


}