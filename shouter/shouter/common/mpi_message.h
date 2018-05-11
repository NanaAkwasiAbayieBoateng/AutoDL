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


struct __attribute__((packed)) TensorMsgHeader {
  uint8_t cmd;      // MessageCmd
  uint8_t srank;    // 256 is max worker num
  uint8_t drank;    // srank -> [drank_0,...,drank_1]
  uint8_t dmask;  
  uint32_t step;    // step
  
  uint8_t  splice;   
  uint32_t size:24;
  uint32_t tensorid;
  uint8_t* buffer;  
  
};
static constexpr size_t TensorMsgHeader_LEN = sizeof(TensorMsgHeader);

struct __attribute__((packed)) TensorMsgStatus {
  uint8_t status;  
};
static constexpr size_t TensorMsgStatus_LEN = sizeof(TensorMsgStatus);

} // namespace common
} // namespace SHOUTER

#endif // SHOUTER_MPI_MESSAGE_H
