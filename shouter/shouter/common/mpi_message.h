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



namespace shouter {
namespace common {



#define offset_of(type, field) ( (unsigned int)&(((type *)(0))->field) )  
#define container_of(ptr, type, field) (type *)((char *)ptr - offset_of(type, field)) 

enum MessageCmd {READ, WRITE};

const char* message_cmd_name(int cmd);

// all message process async,  for A read message from B:
//   1. A send a ReadMesage  to B
//   2. B set write message A when tensor is ready.

// the stream is TensorMessage buffer TensorMessage TensorMessage for easy decode and encode

// START_SEC is 2018-01-01 00:00:00 timestamp
static constexpr uint32_t START_SEC = 1514736000;


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

#define SetTensorMessageMagic(m)            (m->magic = 0x12f0c8d)
#define CheckTensorMessageMagic(m)  (m->magic == 0x12f0c8d)

// just a helper function, call every thread
void allocate_thread_message(int num);

TensorMessage* make_message(int cmd, int step, int tensorid, int slice);

void free_message(TensorMessage* m);


} // namespace common
} // namespace SHOUTER

#endif // SHOUTER_MPI_MESSAGE_H
