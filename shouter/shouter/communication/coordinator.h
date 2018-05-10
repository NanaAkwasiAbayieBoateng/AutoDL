#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>

#include "common/mpi_message.h"
#include "communication/proxy.h"

namespace shouter{

// implement tensor sync
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