// Test funture and promis performance
// complie g++ -std=c++14 -I/home/tanguofu/workspace/seastar test_future.cpp
// 

#include <iostream>
#include <thread>

#include "core/future.hh"
#include "core/reactor.hh"
#include "core/sleep.hh"
#include "core/app-template.hh"




template<typename T>
int do_something(std::unique_ptr<T> obj) {
     // do some computation based on the contents of obj, let's say the result is 17
     return 17;
}

template<typename T>
seastar::future<int> slow_do_something(std::unique_ptr<T> obj) {
    using namespace std::chrono_literals;
    // The following line won't compile...
    return seastar::sleep(10ms).then([obj=std::move(obj)] () mutable { return do_something(std::move(obj)); });
}

seastar::future<> do_all() {
    using namespace std::chrono_literals;
    seastar::future<int> slow_two = seastar::sleep(2s).then([] { return 2; });
    return when_all(seastar::sleep(1s), std::move(slow_two), 
                    seastar::make_ready_future<double>(3.5)
           ).then([] (auto tup) {
            std::cout << std::get<0>(tup).available() << "\n";
            std::cout << std::get<1>(tup).get0() << "\n";
            std::cout << std::get<2>(tup).get0() << "\n";
    });
}


seastar::future<> f(){
  std::unique_ptr<int> p = std::make_unique<int>(100);
  return slow_do_something(std::move(p)).then([](int i){
      std::cout<<"slow_do_something:"<<i<<std::endl;
  }).then([]{
      return do_all();
  });
}





#include <stdexcept>

int main(int argc, char** argv) {
    seastar::app_template app;
    
    /*try {
        app.run(argc, argv, [] {
            std::cout << seastar::smp::count << "\n";
            return seastar::make_ready_future<>();
        });
    } catch(...) {
        std::cerr << "Failed to start application: "
                  << std::current_exception() << "\n";
        return 1;
    }*/

    app.run(argc, argv, f);

    return 0;
}