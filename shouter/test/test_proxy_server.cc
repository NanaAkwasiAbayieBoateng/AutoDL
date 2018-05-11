#include "communication/proxy.h"

using namespace shouter;

TensorMsgStatus print_tensor(TensorMsgHeader & reqheader){

   
   return {0};
}



int main(int argc, char* argv[]){

    std::string ip = 127.0.0.1;
    int port = 1000;

   std::shared_ptr<Proxy>  sp = Proxy::create_proxy(ChannelLinkType::SEASTAR_TCP, 10000, 3);

   // start a loop thread
   sp.register_handle(print_tensor);

    
   std::shared_ptr<Channel>  sp_chanel = sp->get_chanel(ip, port);

   TensorMsgHeader h;
   sp_chanel->write(h).then([](TensorMsgHeader& header){

      std::cout<<"send ed";
   });

   sp.wait();

}