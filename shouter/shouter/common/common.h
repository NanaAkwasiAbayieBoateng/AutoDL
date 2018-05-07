// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef SHOUTER_COMMON_H
#define SHOUTER_COMMON_H

#include<stdint.h>

#include <memory>
#include <string>
#include <vector>



namespace shouter {
namespace common {

// List of supported frameworks.
enum Framework { TENSORFLOW };

enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED };

class Status {
public:
  Status();
  static Status OK();
  static Status UnknownError(std::string message);
  static Status PreconditionError(std::string message);
  static Status Aborted(std::string message);
  bool ok() const;
  StatusType type() const;
  const std::string& reason() const;

private:
  StatusType type_ = StatusType::OK;
  std::string reason_ = "";
  Status(StatusType type, std::string reason);
};

class TensorShape {
public:
  void AddDim(int64_t dim);
  void AppendShape(TensorShape& other);

  const std::string DebugString() const;
  int dims() const;
  int64_t dim_size(int idx) const;
  int64_t num_elements() const;

  inline bool operator==(const TensorShape& rhs) const {
    return shape_ == rhs.shape_;
  }

  inline bool operator!=(const TensorShape& rhs) const {
    return shape_ != rhs.shape_;
  }

private:
  std::vector<int64_t> shape_;
};

class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent(){};
};

class OpContext;

class PersistentBuffer {
public:
  virtual const void* AccessData(std::shared_ptr<OpContext> context) const = 0;
  virtual ~PersistentBuffer(){};
};

enum MPIDataType {
  SHOUTER_UINT8 = 0,
  SHOUTER_INT8 = 1,
  SHOUTER_UINT16 = 2,
  SHOUTER_INT16 = 3,
  SHOUTER_INT32 = 4,
  SHOUTER_INT64 = 5,
  SHOUTER_FLOAT32 = 6,
  SHOUTER_FLOAT64 = 7,
  SHOUTER_BOOL = 8
};

const std::string& MPIDataType_Name(MPIDataType value);

class Tensor {
public:
  virtual const MPIDataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  // the id in graph , must keep all node id is same
  virtual int64_t id() const = 0;
  virtual ~Tensor(){};
};

// every model variable is finalize
class TensorTable {
public:
  
  virtual int register_tensor(const Tensor* tensor);
  virtual int size();

  virtual Tensor* operator[](int id);
  virtual Tensor* operator[](const std::string& name);
};

class OpContext {
public:
  // These allocators are fully synchronous, unlike TensorFlow counterparts.
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) = 0;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Framework framework() const = 0;
  virtual ~OpContext(){};
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMMON_H
