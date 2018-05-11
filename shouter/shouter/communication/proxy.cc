#include "proxy.h"

#include "core/future.hh"
#include "core/reactor.hh"
#include "core/sleep.hh"
#include "core/app-template.hh"

using namespace seastar;
using namespace net;

namespace shouter {

class SeastarProxy;
std::shared_ptr<Proxy> Proxy::create_proxy(type, int port, int parallel){
    
    return std::make_shared<SeastarProxy>(int port, int parallel);
   
}



class seastar_srv_connection {
public:
    seastar_srv_connection(connected_socket&& socket, socket_address addr)
       : _socket(std::move(socket))
       , _addr(addr)
       , _in(_socket.input())
       , _out(_socket.output()){
        // logging a new connection
    }

    ~seastar_srv_connection(){

    }

    future<> onReciveMessage() {
      
    }

public:
    connected_socket _socket;
    socket_address _addr;
    input_stream<char> _in;
    output_stream<char> _out;


};


class SeastarProxy : public Proxy {
public:
    SeastarProxy(int port, int parallel):_port(port),
                                         _parallel(parallel){

                                              
    }
    
  
    int start(){

        listen_options listen_opt = {transport::TCP, true};
        _listener = engine().listen(make_ipv4_address({_port}), lo);
        keep_doing([this] {
            return this->_listener->accept().then([this] (connected_socket fd, socket_address addr) mutable {
                auto conn = make_lw_shared<seastar_srv_connection>(std::move(fd), addr);
                do_until([conn] { return conn->_in.eof(); }, [conn] {
                    return conn->process().then([conn] {
                        return conn->_out.flush();
                    });
                }).finally([conn] {
                    return conn->_out.close().finally([conn]{});
                });
            });
        }).or_terminate();
    }

    future<> stop() { return make_ready_future<>(); }

   
    int run(){

        
        log_cli::print_available_loggers(std::cout);

        // start engine
        bpo::variables_map configuration;
        smp::configure(configuration);

        engine().at_exit([&] { return this->stop(); });
 


 engine().when_started().then([this] {
        seastar::metrics::configure(this->configuration()).then([this] {
            // set scollectd use the metrics configuration, so the later
            // need to be set first
            scollectd::configure( this->configuration());
        });
    }).then([this]{
 start(_port).then([this] {

            return this->invoke_on_all(&tcp_echo_server::start);
        }).then([&, port] {
            std::cout << "TCP Echo-Server listen on: " << _port << "\n";
        });

    }
        
    ).then_wrapped([] (auto&& f) {
        try {
            f.get();
        } catch (std::exception& ex) {
            std::cout << "program failed with uncaught exception: " << ex.what() << "\n";
            engine().exit(1);
        }
    });
    auto exit_code = engine().run();
    smp::cleanup();
    return exit_code;
    }
    
public:
    int _port;
    int _parallel;

    lw_shared_ptr<server_socket> _listener;
};

}