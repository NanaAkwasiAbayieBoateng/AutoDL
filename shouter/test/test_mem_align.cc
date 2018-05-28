#include <stdio.h>
#include <stdint.h>




struct __attribute__((packed)) TensorMessage {
  uint8_t  cmd:4;       // MessageCmd
  uint32_t magic:28;    // fixed: 12f0c8d, to verify the message is right
  uint32_t step;        // global_step

  uint32_t req_sec;     // for trace (now() - START_SEC)*1000 + ms * 1000  
  uint8_t  slice:8;     // Max 256 if each connect between send a peer
  uint32_t tensorid:24; // 16777216 variable nums

  // uint32_t size;     // the size can be 
  char* buffer[0];      //  only for read, take no Space body
};


// for cache the message 
struct __attribute__((packed)) TensorMessageNode{
  uint64_t pad1;  // keep magic
  TensorMessageNode *next;
};

static constexpr size_t TensorMessageLen = sizeof(TensorMessage);
static constexpr size_t TensorMessageNodeLen = sizeof(TensorMessageNode);





int main(int argc, char* argv[]){
    
    printf("%ld\n", TensorMessageLen);
    printf("%ld\n", TensorMessageNodeLen);


    return 0;
}