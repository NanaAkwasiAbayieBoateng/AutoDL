#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>

#include "mpi_message.h"

namespace shouter{


// TODO: SEASTAR_UDP, SEASTAR_DPDK, NCCL, RDMA, MPI
enum ChannelLinkType {
    SEASTAR_TCP,
};


// A worker only has one Proxy use multi-thread to control every channel
// only take the communication
class Proxy {

public:
   
   // listen the port and start  parallel threads
   static std::shared_ptr<Proxy> create_proxy(ChannelLinkType type, int port, int parallel);

   
   // manager channel 
   std::shared_ptr<Channel> get_channel(uint8_t rank);
   int cache_channel(uint8_t rank);

   // process TensorId
   int  onTensorMessage(){

   }

   int run();   

};



}
#endif