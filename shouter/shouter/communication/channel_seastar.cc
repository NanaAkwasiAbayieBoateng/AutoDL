#include "channel.h"

namespace shouter{


class TensorChannelHandler : public ChannelHandler{
public:
       // return 
    virtual int on_msg_header(char* header){
        TensorMsgHeader* h = reinterpret_cast<const char*>(header);
        
        if(vaid_header(h) != 0)
           return Status::ERROR;
        
        if (h->cmd == MessageCmd::READ)
        {
           // append this queue
           tid = h->tid;
           return OK
        }

        if (h->cmd == MessageCmd::WRITE)
        {
           // append this queue
           tid = h->tid;
           set buffer //direction
           return OK;
        }
        
        return Status::ERROR;
    }

};


};
 