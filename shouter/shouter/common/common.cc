#include <sstream>
#include <assert.h>

#include "common/common.h"

namespace shouter {
namespace common {

const std::string& MPIDataType_Name(MPIDataType value) {
  switch (value) {
  case SHOUTER_UINT8:
    static const std::string uint8("uint8");
    return uint8;
  case SHOUTER_INT8:
    static const std::string int8("int8");
    return int8;
  case SHOUTER_UINT16:
    static const std::string uint16("uint16");
    return uint16;
  case SHOUTER_INT16:
    static const std::string int16("int16");
    return int16;
  case SHOUTER_INT32:
    static const std::string int32("int32");
    return int32;
  case SHOUTER_INT64:
    static const std::string int64("int64");
    return int64;
  case SHOUTER_FLOAT32:
    static const std::string float32("float32");
    return float32;
  case SHOUTER_FLOAT64:
    static const std::string float64("float64");
    return float64;
  case SHOUTER_BOOL:
    static const std::string bool_("bool");
    return bool_;
  default:
    static const std::string unknown("<unknown>");
    return unknown;
  }
}


Status::Status() {
}

Status::Status(StatusType type, std::string reason) {
  type_ = type;
  reason_ = reason;
}

Status Status::OK() {
  return Status();
}

Status Status::UnknownError(std::string message) {
  return Status(StatusType::UNKNOWN_ERROR, message);
}

Status Status::PreconditionError(std::string message) {
  return Status(StatusType::PRECONDITION_ERROR, message);
}

Status Status::Aborted(std::string message) {
  return Status(StatusType::ABORTED, message);
}

bool Status::ok() const {
  return type_ == StatusType::OK;
}

StatusType Status::type() const {
  return type_;
}

const std::string& Status::reason() const {
  return reason_;
}

void TensorShape::AddDim(int64_t dim) {
  shape_.push_back(dim);
}

void TensorShape::AppendShape(TensorShape& other) {
  for (auto dim : other.shape_) {
    shape_.push_back(dim);
  }
}

const std::string TensorShape::DebugString() const {
  std::stringstream args;
  args << "[";
  for (auto it = shape_.begin(); it != shape_.end(); it++) {
    if (it != shape_.begin()) {
      args << ", ";
    }
    args << *it;
  }
  args << "]";
  return args.str();
}

int TensorShape::dims() const {
  return (int)shape_.size();
}

int64_t TensorShape::dim_size(int idx) const {
  assert(idx >= 0);
  assert(idx < shape_.size());
  return shape_[idx];
}

int64_t TensorShape::num_elements() const {
  int64_t result = 1;
  for (auto dim : shape_) {
    result *= dim;
  }
  return result;
}

} // namespace common
} // namespace horovod
