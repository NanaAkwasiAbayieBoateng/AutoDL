
#include <functional>

#include "proxy.h"

#include "core/future.hh"
#include "core/reactor.hh"
#include "core/sleep.hh"
#include "core/app-template.hh"

using namespace seastar;
using namespace net;

namespace shouter {


// send buffers
class seastar_send_connection {
public:
    std::vector<connected_socket> _sockets;
    semaphore _conn_connected {0};
    int _cpu;
public:

    // Establish all the TCP connections, sync call so easy to check conection is ok
    future<> connect(const std::string &addr, uint16_t port) {
        ipv4_addr server_addr = {addr， port}；
      
        engine().net().connect(make_ipv4_address(server_addr)).then([this] (connected_socket fd) {
                _sockets.push_back(std::move(fd));
                //print("Established connection %6d on cpu %3d\n", _conn_connected.current(), engine().cpu_id());
                _conn_connected.signal();
            }).or_terminate();
        }
        return _conn_connected.wait(1);
    }
 
   // callback is free buffer
   virtual future<> send_buffer(TensorMsgHeader h){
        return smp::submit_to(_cpu, [this, h, callback](){ 

            const char* buffer = reinterpret_cast<const char*>(&h);
            return  _sockets[0].write(sstring{buffer, TensorMsgHeader_LEN}).then([]{
                return _out.flush();
            });
        });
    };

       // callback is free buffer
   virtual future<> send_buffer(TensorMsgBody b){
        return smp::submit_to(_cpu, [this, b, callback](){ 
            const char* buffer = reinterpret_cast<const char*>(b);

            return  _sockets[0].write(sstring{buffer, TensorMsgBody_LEN})
                .then([]{
                    return  _sockets[0].write(sstring{b.buffer, b.size})
                .then([]{
                    return _out.flush();
                });
        });
    };

// recv buffers
class seastar_recv_connection {
public:
    seastar_srv_connection(connected_socket&& socket, socket_address addr)
       : _socket(std::move(socket))
       , _addr(addr)
       , _in(_socket.input())
       , _out(_socket.output())
       , _status(ConnectionStatus::HEAD)
       , _hanler(ChannelHandler&& h){
        // logging a new connection
    }

    ~seastar_srv_connection(){

    }

    // recv just process recv message not send, 
    // so the logic will be easy. 
    future<> process(){
         
      if(_status == ChannelHandler::Status::HEAD){

        return _in.read_exactly(TensorMsgHeader_LEN).then([this](temporary_buffer<char>&& data) mutable {

           if(data.empty()){
                _status = ChannelHandler::Status::ERROR;
                return make_ready_future<>();
           }

           _status = _hanler->on_msg_header(data.get());

           return process();
        });
      }

      if(_status == ChannelHandler::Status::BODY){
        
        return _in.read_exactly(TensorMsgBody_LEN).then([this](temporary_buffer<char>&& data) mutable {

           if(data.empty()){
                _status = ChannelHandler::Status::ERROR;
                return make_ready_future<>();
           }

           _status = _hanler->on_msg_body(data.get());

           return process();
        });        
      }

      // log error
      return _in.close();
    }

public:
    connected_socket _socket;
    socket_address _addr;
    input_stream<char> _in;
    output_stream<char> _out;
public:
    ChannelHandler::Status status;
    std::unique_ptr<ChannelHandler> hanlder;
};

class SeaStarServer{
public:
   //
   SeaStarServer(int port):_port(port){

   }
     
   // run on one smp server
   int start(){

        listen_options listen_opt = {transport::TCP, true};

        _listener = engine().listen(make_ipv4_address({_port}), lo);

        // server loop
        keep_doing([this] {
            // this will block?
            return this->_listener->accept().then([this] (connected_socket fd, socket_address addr) mutable {
                
                auto conn = make_lw_shared<seastar_recv_connection>(std::move(fd), addr);
                
                // save conection for conn.end() 
                do_until([conn] { return conn->_in.eof(); }, [conn] {
                    return conn->process();
                }).finally([conn] {
                    return conn->_out.close().finally([conn]{});
                });
            });
        }).or_terminate();
    }
   
    future<> stop() { return make_ready_future<>(); }

public:
    lw_shared_ptr<server_socket> _listener;
}

// Proxy keep the diver logic 
class SeastarProxy : public Proxy {
public:
    // parallel is smp cpu num, which we start this threads
    SeastarProxy(std::vector<std::string> ips, int port)
        :_port(port),
         _parallel(parallel){                                   
    }
   
    // call in a single thread.
    int run(){
    
        
        log_cli::print_available_loggers(std::cout);

        // configure 
        bpo::variables_map configuration;
        smp::configure(configuration);

        engine().at_exit([&] { return this->stop(); });

        engine().when_started().then([this, &configuration] {

            seastar::metrics::configure(configuration).then([this, configuration] {
                 // set scollectd use the metrics configuration, so the later
                 // need to be set first
                 scollectd::configure(configuration);
            });
        }).then([this]{
            // start the listen 
            start(_port).then([this] {
                return this->invoke_on_all(&tcp_echo_server::start);
            }).then([&, port] {
                std::cout << "TCP Echo-Server listen on: " << _port << "\n";
            });
        }).then_wrapped([] (auto&& f) {
            try {
               f.get();
            } catch (std::exception& ex) {
               std::cout << "program failed with uncaught exception: " << ex.what() << "\n";
               engine().exit(1);
            }
        });
        
        // start loop   
        auto exit_code = engine().run();
        smp::cleanup();
        return exit_code;
    }
    
public:
   
   // sort by ip
   std::vecotor<std::string> all_host; 


   // now just as all peer has direct p2p 
   uint8_t max_rank;
   uint8_t local_rank;

   // we should has many recv connections
   // each rank has 
   std::vector< std::shared_ptr<seastar_send_connection>> send_connections;
 
};


class SeastarProxy;

std::shared_ptr<Proxy> Proxy::create_proxy(ChannelLinkType type, int port, int parallel){    
    return std::make_shared<SeastarProxy>(int port, int parallel);   
}

}