#ifndef SHOUTER_CHANNEL_H
#define SHOUTER_CHANNEL_H

#include <stdint.h>
#include <iostream>


#include "common/mpi_message.h"


namespace shouter{

enum ChannelStatus { CHANNEL_INIT, CHANNEL_CONNECTING, CHANNEL_ESTABLISH, CHANNEL_FAIL, CHANNEL_ERROR, CHANNEL_CLOSED};

// A channcel is interface for data stream transmission
// As i prefer libco, not future or promise but for tcp stack, seastar is hard to refuse.

class Channel {
public:
   // only move 
   Channel() = delete;
   Channel(Channel&&) = default;
 
public:
   ChannelStatus status(){return conn.status;}

   std::future<int> write(const TensorMessage& msg){
        return conn.send(TensorMessage)
        
   }
   std::future<int> read(const TensorMessage& msg);

};



}
#endif
