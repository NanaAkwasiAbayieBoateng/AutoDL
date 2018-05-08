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



// A channcel is interface for data stream transmission
// As i prefer libco, not future or promise but for tcp stack, seastar is hard to refuse.
class Channel {
public:
   std::future<int> write(const TensorMessage& msg);
   std::future<int> read(const TensorMessage& msg);
};


// A worker only has one Proxy use multi-thread to control every channel
// only take the communication
class Proxy {

public:
   Proxy(int port, int threadnum):
   
   std::shared_ptr<Channel> get_channel(uint8_t rank);
   int cache_channel(uint8_t rank);

   int run_loop();

};

class Coordicater {

public:
    int setup_communicate_ring(std::vector<std::string> workers, int port, ChannelLinkType type=ChannelLinkType::SEASTAR_TCP);

    int setup_tensortable(std::vector<Tensor> tensors, int cache_step=1);

    int brocast(step, Tensor, uint8_t rank0, uint8_t rank1);
    int reduce(step, Tensor, uint8_t rank0, uint8_t rank1);
    int allreduce(step, Tensor);
    int allbrocast(step, Tensor);
};

}
#endif