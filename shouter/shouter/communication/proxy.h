#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>

#include "common/mpi_message.h"

namespace shouter{


// TODO: SEASTAR_UDP, SEASTAR_DPDK, NCCL, RDMA, MPI
enum ChannelLinkType {
    SEASTAR_TCP,
};


// A worker only has one Proxy use multi-thread to control every channel
// only take the communication
class Proxy {

public:
   Proxy(int port, int threadnum):

   std::shared_ptr<Proxy> setup_proxy(int port, int threadnum, )
   
   std::shared_ptr<Channel> get_channel(uint8_t rank);
   int cache_channel(uint8_t rank);

   int run();

};

}
#endif