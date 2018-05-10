#include "core/print.hh"
#include "core/reactor.hh"
#include "core/app-template.hh"
#include "core/future-util.hh"
#include "core/distributed.hh"
#include "core/semaphore.hh"
#include "core/future-util.hh"
#include <chrono>
#include <array>
using namespace seastar;

class tcp_echo_client {
private:
    unsigned _duration {0};
    unsigned _conn_per_core {0};
    unsigned _reqs_per_conn {0};
    std::vector<connected_socket> _sockets;
    semaphore _conn_connected {0};
    semaphore _conn_finished {0};
    timer<> _run_timer;
    bool _timer_based { false };
    bool _timer_done {false};
    uint64_t _total_reqs {0};
public:
    tcp_echo_client(unsigned duration, unsigned total_conn, unsigned reqs_per_conn)
        : _duration(duration)
        , _conn_per_core(total_conn / smp::count)
        , _reqs_per_conn(reqs_per_conn)
        , _run_timer([this] { _timer_done = true; })
        , _timer_based(reqs_per_conn == 0)
    {
    }

    class connection {
    private:
        connected_socket _fd;
        input_stream<char> _in;
        output_stream<char> _out;
        tcp_echo_client& _echo_client;
        uint64_t _nr_done{0};
    public:
        connection(connected_socket&& fd, tcp_echo_client& echo_client)
            : _fd(std::move(fd))
            , _in(_fd.input())
            , _out(_fd.output())
            , _echo_client(echo_client)
        {
        }

        ~connection() {
        }

        uint64_t nr_done() {
            return _nr_done;
        }

        sstring generate_payload() {
            return {"HIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHIHHIHIHIHIHIIHIHH"};
        }

        std::vector<char> encode_request_body(sstring payload) {
            std::vector<char> body;
            auto u = htonl(static_cast<uint32_t>(payload.size()));
            auto *s = reinterpret_cast<const char*>(&u);
            body.insert(body.end(), s, s + sizeof(u));
            body.insert(body.end(), payload.begin(), payload.end());
            return std::move(body);
        }

        uint32_t decode_reply_size(temporary_buffer<char>& data) {
            assert(data.size() == 4);
            auto p = reinterpret_cast<const uint8_t*>(data.get());
            uint32_t n = (static_cast<uint32_t>(p[0]) << 24)
                | (static_cast<uint32_t>(p[1]) << 16)
                | (static_cast<uint32_t>(p[2]) << 8)
                | (static_cast<uint32_t>(p[3]));
            return n;
        }
        future<> do_launch_request() {
            auto payload = generate_payload();
            auto body = encode_request_body(std::move(payload));
            return _out.write(sstring{ body.data(), body.size() }).then([this] {
                return _out.flush();
            }).then([this] {
                return _in.read_exactly(4).then([this] (auto&& data) {
                    auto payload_size = this->decode_reply_size(data);
                    return _in.read_exactly(payload_size).then([this] (auto&& payload) {
                        _nr_done++;
                        if (_echo_client.done(_nr_done)) {
                            return make_ready_future<>();
                        }
                        return this->do_launch_request();
                    });
                });
            });
        }
    };

    future<uint64_t> total_reqs() {
        print("Requests on cpu %2d: %ld\n", engine().cpu_id(), _total_reqs);
        return make_ready_future<uint64_t>(_total_reqs);
    }

    bool done(uint64_t nr_done) {
        if (_timer_based) {
            return _timer_done;
        } else {
            return nr_done >= _reqs_per_conn;
        }
    }

    future<> connect(ipv4_addr server_addr) {
        // Establish all the TCP connections
        for (unsigned i = 0; i < _conn_per_core; i++) {
            engine().net().connect(make_ipv4_address(server_addr)).then([this] (connected_socket fd) {
                _sockets.push_back(std::move(fd));
                print("Established connection %6d on cpu %3d\n", _conn_connected.current(), engine().cpu_id());
                _conn_connected.signal();
            }).or_terminate();
        }
        return _conn_connected.wait(_conn_per_core);
    }

    future<> run() {
        print("Established all %6d tcp connections on cpu %3d\n", _conn_per_core, engine().cpu_id());
        if (_timer_based) {
            _run_timer.arm(std::chrono::seconds(_duration));
        }
        for (auto&& fd : _sockets) {
            auto conn = make_lw_shared<connection>(std::move(fd), *this);
            conn->do_launch_request().then_wrapped([this, conn] (auto&& f) {
                _total_reqs += conn->nr_done();
                _conn_finished.signal();
                try {
                    f.get();
                } catch (std::exception& ex) {
                    print("Echo request exception: %s\n", ex.what());
                }
            }).finally([conn] {});
        }
        return _conn_finished.wait(_conn_per_core);
    }
    future<> stop() {
        return make_ready_future();
    }
};

namespace bpo = boost::program_options;

int main(int ac, char** av) {
    distributed<tcp_echo_client> shard_echo_client;
    app_template app;
    app.add_options()
        ("server,s", bpo::value<std::string>()->default_value("10.24.193.146:10000"), "Server address")
        ("conn,c", bpo::value<unsigned>()->default_value(100), "total connections")
        ("reqs,r", bpo::value<unsigned>()->default_value(10000), "reqs per connection")
        ("duration,d", bpo::value<unsigned>()->default_value(10), "duration of the test in seconds)");

    return app.run(ac, av, [&] () -> future<int> {
        auto& config = app.configuration();
        auto server = config["server"].as<std::string>();
        auto reqs_per_conn = config["reqs"].as<unsigned>();
        auto total_conn= config["conn"].as<unsigned>();
        auto duration = config["duration"].as<unsigned>();

        if (total_conn % smp::count != 0) {
            print("Error: conn needs to be n * cpu_nr\n");
            return make_ready_future<int>(-1);
        }

        auto started = steady_clock_type::now();
        print("========== tcp_echo_client ============\n");
        print("Server: %s\n", server);
        print("Connections: %u\n", total_conn);
        print("Requests/connection: %s\n", reqs_per_conn == 0 ? "dynamic (timer based)" : std::to_string(reqs_per_conn));
        return shard_echo_client.start(std::move(duration), std::move(total_conn), std::move(reqs_per_conn)).then([&shard_echo_client, server] {
            return shard_echo_client.invoke_on_all(&tcp_echo_client::connect, ipv4_addr{server});
        }).then([&shard_echo_client] {
            return shard_echo_client.invoke_on_all(&tcp_echo_client::run);
        }).then([&shard_echo_client] {
            return shard_echo_client.map_reduce(adder<uint64_t>(), &tcp_echo_client::total_reqs);
        }).then([&shard_echo_client, started] (uint64_t total_reqs) {
           auto finished = steady_clock_type::now();
           auto elapsed = finished - started;
           auto secs = static_cast<double>(elapsed.count() / 1000000000.0);
           print("Total cpus: %u\n", smp::count);
           print("Total requests: %u\n", total_reqs);
           print("Total time: %f\n", secs);
           print("Requests/sec: %f\n", static_cast<double>(total_reqs) / secs);
           print("==========     done     ============\n");
           return shard_echo_client.stop().then([] {
               return make_ready_future<int>(0);
           });
        });
    });
}