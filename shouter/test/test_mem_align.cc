
#include <stdio.h>
#include <stdint.h>


struct TensorMsgHeader {
  uint8_t  cmd;      // MessageCmd
  uint8_t  srank;    // 256 is max worker num
  uint16_t nums;     // the num of TensorMsgBody
  uint32_t step;     // step
  uint32_t req_ms;   // for trace
  uint32_t session;  // for read  write seq is same the read seq
};
static constexpr size_t TensorMsgHeader_LEN = sizeof(TensorMsgHeader);

struct  TensorMsgBody {
  uint32_t tensorid;
  uint8_t  splice;   // Max 256(rank ) splice 
  uint32_t size:24;
  uint8_t* buffer[0];    
};
static constexpr size_t TensorMsgBody_LEN = sizeof(TensorMsgBody);



int main(int argc, char* argv[]){
    
    printf("%ld\n", TensorMsgHeader_LEN);
    printf("%ld\n", TensorMsgBody_LEN);


    return 0;
}