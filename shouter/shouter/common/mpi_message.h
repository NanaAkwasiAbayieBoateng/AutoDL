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



enum MessageCmd {READ, WRITE};

class TensorMessage {
public:
  uint8_t cmd;      // MessageCmd
  uint8_t srank;    // 256 is max worker num
  uint8_t drank_0;  // srank -> [drank_0...drank_1]
  uint8_t drank_1;  
  uint32_t step;    // step 
  uint32_t id;
  uint32_t size;
  uint8_t* buffer;  
};



} // namespace SHOUTER

#endif // SHOUTER_MPI_MESSAGE_H
