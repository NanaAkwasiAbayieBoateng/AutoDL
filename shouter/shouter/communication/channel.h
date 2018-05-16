#ifndef SHOUTER_CHANNEL_H
#define SHOUTER_CHANNEL_H

#include <stdint.h>
#include <iostream>


#include "common/mpi_message.h"


namespace shouter{


// A channcel is interface for data stream transmission
// As i prefer libco, not future or promise but for tcp stack, seastar is hard to refuse.
class ChannelHandler {
public:
    ennum Status{HEAD,BODY,ERROR}
    
    // return 
    virtual int on_msg_header(char* header);

    virtual int on_msg_body(char* body);

};

class Channel {
public:
   // only move 
   Channel() = delete;
   Channel(Channel&&) = default;
 
public:
   ChannelStatus status(){return _status;}

   
   virtual future<> write(Tensor T); 

   virtual future<TensorMsgBody> read(Tensor T);

};



}
#endif
