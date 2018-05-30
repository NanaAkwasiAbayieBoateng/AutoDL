// Test funture and promis performancex
// complie g++ -std=c++14 -I/home/tanguofu/workspace/seastar test_future.cpp
//

#include <iostream>
#include <thread>

#include <typeinfo>

#include <stdexcept>

#include "core/app-template.hh"
#include "core/future.hh"
#include "core/reactor.hh"
#include "core/sleep.hh"

#include "core/thread.hh"

// for debug types
#include "check_type.hpp"

template <typename T>
int do_something(std::unique_ptr<T> obj) {
    // do some computation based on the contents of obj, let's say the
    // result is
    // 17

    using namespace std::chrono_literals;
    seastar::sleep(200ms).then([] { std::cout << "200ms " << std::flush; });
    seastar::sleep(100ms).then([] { std::cout << "100ms " << std::flush; });

    // auto result = seastar::sleep(1s).then([] { std::cout << "Done.\n";
    // return
    // seastar::make_ready_future<int>(17); });
    auto result = seastar::make_ready_future<int>(17);

    std::cout << check_type<decltype(result)>() << std::endl;
    // std::tuple
    // return 17;
    // result.wait();
    return std::get<0>(result.get());
}

template <typename T>
seastar::future<int> slow_do_something(std::unique_ptr<T> obj) {
    using namespace std::chrono_literals;
    // The following line won't compile...
    return seastar::sleep(10ms).then([obj = std::move(obj)]() mutable {
        return do_something(std::move(obj));
    });
}

seastar::future<> do_all() {
    using namespace std::chrono_literals;
    seastar::future<int> slow_two = seastar::sleep(2s).then([] { return 2; });
    return when_all(seastar::sleep(1s), std::move(slow_two),
                    seastar::make_ready_future<double>(3.5))
        .then([](auto tup) {
            std::cout << std::get<0>(tup).available() << "\n";
            std::cout << std::get<1>(tup).get0() << "\n";
            std::cout << std::get<2>(tup).get0() << "\n";
        });
}

seastar::future<> f() {
    std::cout << seastar::smp::count << "\n";
    using namespace std::chrono_literals;
    const auto &&a = seastar::sleep(1s);
    std::cout << check_type<decltype(a)>() << std::endl;
    auto b = 100ms;
    using typea = decltype(b);
    std::cout << check_type<typea>() << std::endl;

    std::vector<typea> span;
    span.push_back(typea(100ms));
    span.push_back(typea(200ms));
    span.push_back(typea(300ms));
    span.push_back(typea(400ms));

    std::vector<seastar::future<int>> actions;

    for (int i = 0; i < 4; i++) {
        auto f = seastar::sleep(span[i]).then([i] {
            std::cout << "sleep:" << i << std::endl;
            return i;
        });

        actions.push_back(std::move(f));
    }

    seastar::thread th([&actions] {

        // seastar::sleep(std::chrono::seconds(1)).get();

        auto &&f = seastar::when_all_succeed(actions.begin(), actions.end())
                       .then([](std::vector<int> &&a) {
                           std::cout << check_type<decltype(a)>() << std::endl;
                       });


        f.wait();

    });

    th.join().wait();

    std::unique_ptr<int> p = std::make_unique<int>(100);
    return slow_do_something(std::move(p))
        .then(
            [](int i) { std::cout << "slow_do_something:" << i << std::endl; })
        .then([] { return do_all(); });
}

seastar::future<> f1() {
    seastar::sleep(std::chrono::seconds(1)).get();
    seastar::thread th([] {
        std::cout << "Hi.\n";
        for (int i = 1; i < 4; i++) {
            seastar::sleep(std::chrono::seconds(1)).get();
            std::cout << i << "\n";
        }
    });
    return do_with(std::move(th), [](auto &th) { return th.join(); });
}

int main(int argc, char **argv) {
    seastar::app_template app;
    app.run(argc, argv, f1);

    return 0;
}