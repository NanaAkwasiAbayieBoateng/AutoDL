
#include <iostream>

#include "common/mpi_message.h"
#include "common/singleton.h"

namespace shouter {
namespace common {


const char* message_cmd_name(int cmd){
    switch(cmd){
    case READ: 
        return "READ";
    case WRITE: 
        return "WRITE";
    }
    return "unkown";
}

static_assert(TensorMessageLen == 16, "TensorMessage must be 16 byte");
static_assert(TensorMessageNodeLen == 16, "TensorMessageNode must be 16 byte");



class TensorMessageAllocator {
public:
    TensorMessageAllocator():
              _msg_num(0),
              idle_list(NULL){
    }

    void allocate_message(int max_msg_num){        
       
        // init the idle list
        TensorMessageNode* buffer = new TensorMessageNode[max_msg_num];
        TensorMessageNode* pre = idle_list;
        TensorMessageNode* n   = buffer + max_msg_num - 1;

        for(; n != buffer; --n){
            n->next = pre;
            pre = n;
        }
        // append the current list
        idle_list = buffer; 
    }

    TensorMessage* make_message(int cmd, int step, int tensorid, int slice){

        TensorMessage* m = NULL;
        if(idle_list){
            m = reinterpret_cast<TensorMessage*>(idle_list);
            idle_list = idle_list->next;
        }else{
           m = new TensorMessage();
           SetTensorMessageMagic(m);
        }

        m->cmd = cmd;
        m->step = step;
        m->tensorid = tensorid;
        m->slice = slice;
      
        return m;
    }

    void free_message(TensorMessage* m){        
        TensorMessageNode* n = reinterpret_cast<TensorMessageNode*>(m);
        n->next = idle_list;
        idle_list = n;
    }

private:
   int _msg_num;
   TensorMessageNode* idle_list;
};

void allocate_thread_message(int num){
    ThreadLocalSingleton<TensorMessageAllocator>::instance()
                .allocate_message(num);
}

TensorMessage* make_message(int cmd, int step, int tensorid, int slice){
    return ThreadLocalSingleton<TensorMessageAllocator>::instance()
                .make_message(cmd, step, tensorid, slice);
}

void free_message(TensorMessage* m){
    ThreadLocalSingleton<TensorMessageAllocator>::instance()
                .free_message(m);
}

} // namespace common
} // namespace SHOUTER
