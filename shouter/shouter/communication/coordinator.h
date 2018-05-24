#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>
#include <vector>


#include "common/common.h"
#include "common/mpi_message.h"
#include "communication/proxy.h"

using namespace shouter::common;

// class Proxy


// implement a interface, for python invoke
class Coordicater {

public:
    int generate_ring(const std::vector<std::string>& addrs, int local_port);

    
    int rank();

    const std::string addr();

    int size();
   


    int alloc_brocast(int rank, std::string& name, int size,  int dtype);
    int alloc_reduce(int rank, std::string& name, int size,  int dtype);
    int alloc_allreduce(std::string& name, int size,  int dtype);

    // TODO
    int alloc_gather(int rank, std::string& name, std::vector<int> shape, int dtype);
    int alloc_scatter(int rank, std::string& name, std::vector<int> shape, int dtype);

    // in tensor operation
    int start_brocast(uint32_t step, std::string& name, char* buffer, std::function<(int)> done);
    int start_reduce(uint32_t step, std::string& name, char* buffer, std::function<(int)> done);
    int start_reduce(uint32_t step, std::string& name, char* buffer, std::function<(int)> done);

    int set_log_printer(std::function<std::basic_ostringstream<char>&()> printer); 
};

}//  namespace shouter




#endif