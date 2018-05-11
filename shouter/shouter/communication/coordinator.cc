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
    int setup_communicate_ring(std::vector<std::string> workers, int port, ChannelLinkType type=ChannelLinkType::SEASTAR_TCP){

      for i,j i  < j ring[i][j]= channel * n

    }

    int setup_tensortable(std::vector<Tensor> tensors, int cache_step=1){
      //brocast or check

    }

    int brocast(step, Tensor, uint8_t rank0, uint8_t rank1)
    {
         ring[rank0][j&mask] = writeMesage
    }

    int reduce(step, Tensor, uint8_t rank0, uint8_t rank1){

        ring[rank0][j&mask] = readMesage
    }
    int allreduce(step, Tensor){

        ring[i][j] = readMesage
    }

    int allbrocast(step, Tensor){
        ring[i][j] = writeMesage
        
    }

private:

   TensorTable _local_tensor_table;
   TensorTable _global_tensor_table;
   Proxy _proxy;   
};


typedef std::function<void(const Status&)> StatusCallback;


 ComputeAsync(OpKernelContext* context, DoneCallback done) -> get_step,  Coordicater.allreduce(step, Tensor).then([]{done()};


})


}
#endif