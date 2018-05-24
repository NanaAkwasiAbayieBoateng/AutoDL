#ifndef SHOUTER_OPERATIONS_H
#define SHOUTER_OPERATIONS_H

#include <stdint.h>
#include <iostream>
#include <future>
#include <thread>
#include <sstream>
#include <unordered_map>

#include "common/singleton.h"
#include "common/mpi_message.h"
#include "common/common.h"

#include "common/device_helper.h"

#include "communication/proxy.h"


namespace shouter{

enum OpType{
    REDUCE     = 1,
    ALL_REDUCE = 2,
    BROCAST    = 3,
    GARTHER    = 4,
    ALL_GARTHER= 5,
};

class 

class RankTensor {
public:
   int id;
   std::string name;
   int size;

   int rank;
   std::vector<char*> rankbuffer;
   char* buffer;
};

// MetaData is global date
class MetaData {
public:
    const int ALLRANK = 0xFF; 

    // from 0 to  ALLRANK  
    int rank;
    int size;
    std::string localaddr;
    int port;

    std::vector<std::string> workers;
    std::vector<int> others;

    std::fuction< std::basic_ostringstream<char>&()> printer;  


    std::vector<RankTensor> tensors;
    std::unorder_map<std::string, RankTensor*> namemap;
};


// global data
MetaData meta;
meta.printer = [](){return std::cout;}

int Coordicater::generate_ring(const std::vector<std::string>& addrs,int local_port)
{
    meta.workers = addrs;   
    meta.localaddr = "127.0.0.1";
    meta.rank = 0;
    meta.size = addrs.size();
    meta.port = local_port;

    // keep the rank unique
    std::sort(meta.workers.begin(), meta.workers.end()); 


    // start proxy
    std::vector<std::string> local_ips = getHostIp();
    for(auto& ip : local_ips){
       for(int i = 0; i < meta.workers.size(); ++i ){
          
          char buffer[64] = {0};
          int len = snprintf(buffer, 64, "%s:%d", ip.c_str(), local_port);
          if( len > 0 && 0 == strncmp(buffer, worker.c_str(), std::min(len, worker.size())))
          {
              meta.rank = i;
              meta.localaddr = buffer;
              break;
          } 
       }
    }
    
    for(int i = 0; i < meta.size; ++i) {
        if (i != meta.rank)
           meta.others.push_back(i);
    }

    meta.printer()<<"generate_ring addr:"<<meta.localaddr<<", rank:"<<meta.rank<<std::endl; 

    // start proxy

    // construct channel
    return StatusType::OK;   
}

int Coordicater::rank() {
    return meta.rank;
}

int Coordicater::size() {
    return meta.workers.size();
}

const std::string Coordicater::addr() {
    return meta.localaddr;
}


int Coordicater::alloc_brocast(int rank, std::string& name, int size, int dtype){
    
    int tid = meta.tensors.size();
    RankTensor tensor = {tid, name, rank, {}};
    meta.tensors.push_back(tensor);

    meta.namemap.insert(std::make_pair<std::string, tid>(name, tid));    

    return tid;
};

int Coordicater::alloc_reduce(int rank, std::string& name, int size, int dtype){
   
    int tid = meta.tensors.size();
    std::vector<char*> vec;
    
    // alloc the buffer for every rank
    for(int i=0; i < meta.workers.size(); ++i) {
        char* buffer = (i == rank) ? NULL: new char[size * sizeof(float)];
        vec.push_back(buffer);
    }
     
    RankTensor tensor = {tid, name, MetaData::ALLRANK, vec};
    meta.tensors.push_back(tensor);

    meta.namemap.insert(std::make_pair<std::string, tid>(name, tid));    

    return 0;
}

int Coordicater::alloc_allreduce(std::string& name, int size,  int dtype){
    
    int tid = meta.tensors.size();
    std::vector<char*> vec;
    
    // alloc the buffer for every rank
    for(int i=0; i < meta.workers.size(); ++i) {
        char* buffer = (i == rank) ? NULL: new char[size * sizeof(float)];
        vec.push_back(buffer);
    }
     
    RankTensor tensor = {tid, name, rank, vec};
    meta.tensors.push_back(tensor);

    meta.namemap.insert(std::make_pair<std::string, tid>(name, tid));    

    return 0;
}


int Coordicater::start_brocast(uint32_t step, std::string& name, char* buffer){

    RankTensor* tensor = meta.namemap[name];
    tensor->buffer = buffer;
    int size = meta.size;
    
    // send all
    if(meta.rank == tensor->rank) {

        return seastar::parallel_for_each(meta.others, [buffer](int i){
            const std::string & addr = meta.worker[i];
            std::shared_ptr<Channel> spch = proxy.get_channel(addr);
            return spch.send(tensor->buffer, tensor->size);
        }).then([done=std::move(done)](){
            return make_ready_future<int>(0)    
        });

    }else{ // for other rank recv
        
        const std::string & addr = meta.worker[meta.rank];
        std::shared_ptr<Channel> spch = proxy.get_channel(addr);
        return spch.read(tensor->buffer, tensor->size);
    }

    return 0;     
}

int Coordicater::start_reduce(uint32_t step, std::string& name, char* buffer, std::function<(int)> done){
    
    RankTensor* tensor = meta.namemap[name];
    tensor->buffer = buffer;
    int size = meta.size;
    // send all
    if(meta.rank == tensor->rank){

       return seastar::parallel_for_each(meta.others, [buffer](int i)

            const std::string & addr = meta.worker[i];
            std::shared_ptr<Channel> spch = proxy.get_channel(addr);
            return spch.read(tensor->rankbuffer[i], tensor->size);
        }).then([done=std::move(done)](){
            return make_ready_future<int>(0)    
        });

    }else{ // for other rank
        
        const std::string & addr = meta.worker[tensor->rank];
        std::shared_ptr<Channel> spch = proxy.get_channel(addr);
        return spch.send(tensor->buffer, tensor->size);
    }

    return 0;
}

int Coordicater::start_allreduce(uint32_t step, std::string& name, char* buffer, std::function<(int)> done){
    
    RankTensor* tensor = meta.namemap[name];
    tensor->buffer = buffer;
    int size = meta.size;


    return when_all(seastar::parallel_for_each(meta.others, [buffer](int i){
                        const std::string & addr = meta.worker[i];
                        std::shared_ptr<Channel> spch = proxy.get_channel(addr);
                        return spch.send(tensor->buffer, tensor->size))
                    ,seastar::parallel_for_each(meta.others, [buffer](int i){
                        const std::string & addr = meta.worker[i];
                        std::shared_ptr<Channel> spch = proxy.get_channel(addr);
                        return spch.read(tensor->rankbuffer[i], tensor->size);
                    }).then([done=std::move(done)]{
                        return make_ready_future<int>(0)    
                    });    

}

int Coordicater::set_log_printer(std::function<std::basic_ostringstream<char>&()> printer){
    meta.printer = printer;
} 



}
#endif