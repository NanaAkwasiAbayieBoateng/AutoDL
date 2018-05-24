
#ifndef SHOUTER_DEVICE_HELPER_H
#define SHOUTER_DEVICE_HELPER_H

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>


#include <sys/types.h>
#include <ifaddrs.h>

namespace shouter {
namespace common {

// get all ip addr in local devices

std::vector<std::string> getHostIp()
{
    // use ipv4    
    std::string ip = "127.0.0.1";
    std::vector<std::string> result;
    result.push_back(ip);

    struct ifaddrs * if_addr = NULL;
    if(0 != getifaddrs(&if_addr))
    {
        return result;
    }
    
    struct ifaddrs * if_addr_back = if_addr;
    
    
    for(; if_addr != NULL; if_addr = if_addr->ifa_next)
    {
        // not ipv4
        if (if_addr->ifa_addr->sa_family != AF_INET)
            continue;

        void * tmp = &((struct sockaddr_in *)if_addr->ifa_addr)->sin_addr;

        char buffer[256]={0};

        inet_ntop(AF_INET, tmp, buffer, INET_ADDRSTRLEN);

        ip = buffer;

        if(ip.size() == 0 || ip.find("127.0") == 0 || ip.find("0.0") == 0)
        {
            continue;
        }
        result.push_back(ip);
    }

    if(if_addr_back)
        freeifaddrs(if_addr_back);

    return result;
}


}
}
#endif