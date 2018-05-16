// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef SHOUTER_MPI_MESSAGE_H
#define SHOUTER_MPI_MESSAGE_H

#include <string>
#include <vector>

#include "common.h"


namespace shouter {
namespace common {



#define offset_of(type, field) ( (unsigned int)&(((type *)(0))->field) )  
#define container_of(ptr, type, field) (type *)((char *)ptr - offset_of(type, field)) 

enum MessageCmd {READ, WRITE};

// all message process async,  for A read message from B:
//   1. A send a ReadMesage  to B
//   2. B set write message A when tensor is ready.

// the stream is TensorMsgHeader + TensorMsgBody*N + TensorMsgBody(tensorid 0)

// START_SEC is 2018-01-01 00:00:00 timestamp
static constexpr uint32_t START_SEC = 1514736000; 
struct __attribute__((packed)) TensorMsgHeader {
  uint8_t  cmd;      // MessageCmd
  uint8_t  srank;    // 256 is max worker num
  uint8_t  drank;
  uint8_t  dmask;
  uint32_t step;     // step 
  uint32_t req_sec;  // for trace (now() - START_SEC)*1000 + ms * 1000
  uint32_t operid;  // for read  write seq is same the read seq
};
static constexpr size_t TensorMsgHeader_LEN = sizeof(TensorMsgHeader);

// as the channel is stream, so need keep the stream packet is right
int vaid_header(TensorMsgHeader& h);



static_assert(TensorMsgHeader_LEN == 16, "TensorMsgHeader must be 16 byte");

// end msage is tensorid 0
struct __attribute__((packed)) TensorMsgBody {
  uint32_t tensorid;
  uint8_t  splice;   // Max 256(rank ) splice 
  uint32_t size:24;  // Max 16G buffer;
  char* buffer[0];    
};
static constexpr size_t TensorMsgBody_LEN = sizeof(TensorMsgBody);
static_assert(TensorMsgBody_LEN == 8, "TensorMsgBody must be 16 byte");
// as the channel is stream, so need keep the stream packet is right
int vaid_body(TensorMsgHeader& h);
} // namespace common
} // namespace SHOUTER

#endif // SHOUTER_MPI_MESSAGE_H
