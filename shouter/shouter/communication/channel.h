#ifndef SHOUTER_CHANNEL_H
#define SHOUTER_CHANNEL_H

#include <stdint.h>
#include <iostream>


#include "common/mpi_message.h"


namespace shouter{



class Channel {
public:
   // only move 
   Channel() = delete;
   Channel(Channel&&) = default;
 
public:
   ChannelStatus status(){return _status;}

   
   virtual future<int> send(Tensor T); 

   virtual future<int> read(Tensor T);

};



}
#endif
