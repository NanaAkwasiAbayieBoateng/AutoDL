#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>

namespace shouter{

// Now implelement TCP posix api
// 
enum ChannelProtocol{
    TCP,
    UDP,    
    DPDK_TCP, // TODO
    RMDA_Roce2   // TODO
};

struct Message {
   uint32_t step;
   uint32_t tensor_id;
   const char* buffer;
   int size;
};

// A channcel is interface for data stream transmission
// As i prefer libco, not future or promise but for tcp stack, seastar is hard to refuse.
class Channel {
public:
   std::future<int> write(const Message& msg);
   std::future<int> read(const Message& msg);
   int close();
};

class TensorRegister{


};
// A worker only has one Proxy use multi-thread to control every channel
// only take the communication
class Proxy {

public:
   Proxy(int port):



};

class SeastarProxy{

};

}
#endif