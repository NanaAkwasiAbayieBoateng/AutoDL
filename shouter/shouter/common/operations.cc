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

#include <assert.h>
#include <atomic>
#include <cstring>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>

// for vscode IDE
#define HAVE_CUDA 1
#define HAVE_NCCL 1
#define HOROVOD_GPU_ALLREDUCE  'N'

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif

#define OMPI_SKIP_MPICXX
#include "hashes.h"
#include "mpi_message.h"
#include "operations.h"
#include "timeline.h"

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements MPI ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * hardware-optimized communication libraries through the MPI implementation.
 *
 * The primary logic of the allreduce, allgather and broadcast are in MPI and
 * NCCL implementations. The background thread which facilitates MPI operations
 * is run in BackgroundThreadLoop(). The provided ops are:
 *      – HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all MPI processes in the global communicator.
 *      – HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all MPI processes in the
 *          global communicator.
 *      - HorovodBroadcast:
 *          Perform a broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * Additionally, this library provides C APIs to initialize Horovod and query
 * rank, local rank and world size.  These are used in Python directly through
 * ctypes.
 */



namespace shouter {
namespace common {

namespace {

// For clarify in argument lists.
#define RANK_ZERO 0

// A tag used for all coordinator messaging.
#define TAG_NOTIFY 1

// Stall-check warning time
#define STALL_WARNING_TIME std::chrono::seconds(60)

static const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use shouter.init().");

static const Status SHUT_DOWN_ERROR = Status::Aborted(
    "Horovod has been shut down. This has been caused by an exception on one "
    "of the rank or an attempt to allreduce, allgather or broadcast a tensor "
    "after one of the ranks has finished execution.");

#if HAVE_NCCL
ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case SHOUTER_INT32:
    return ncclInt32;
  case SHOUTER_INT64:
    return ncclInt64;
  case SHOUTER_FLOAT32:
    return ncclFloat32;
  case SHOUTER_FLOAT64:
    return ncclFloat64;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in NCCL mode.");
  }
}
#endif

#define MPI_CHECK(entries, op_name, op)                                        \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      for (auto it = entries.begin(); it != entries.end(); it++) {             \
        timeline.End(it->tensor_name, nullptr);                                \
        it->callback(Status::UnknownError(                                     \
            std::string(op_name) + " failed, see MPI output for details."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define CUDA_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      for (auto it = entries.begin(); it != entries.end(); it++) {             \
        timeline.End(it->tensor_name, nullptr);                                \
        it->callback(Status::UnknownError(std::string(op_name) + " failed: " + \
                                          cudaGetErrorString(cuda_result)));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define NCCL_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      for (auto it = entries.begin(); it != entries.end(); it++) {             \
        timeline.End(it->tensor_name, nullptr);                                \
        it->callback(Status::UnknownError(std::string(op_name) + " failed: " + \
                                          ncclGetErrorString(nccl_result)));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

// This event management code is only used in NCCL.
#ifdef HAVE_NCCL
cudaError_t GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

/*
  auto& mutex = horovod_global.cuda_events_mutex;
  auto& queue = horovod_global.cuda_events[device];
  {
    std::lock_guard<std::mutex> guard(mutex);
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }
  */

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                             cudaEventDisableTiming);
}

cudaError_t ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  /*auto& mutex = horovod_global.cuda_events_mutex;
  auto& queue = horovod_global.cuda_events[device];
  {
    std::lock_guard<std::mutex> guard(mutex);
    queue.push(event);
  }
  */

  return cudaSuccess;
}

#define RECORD_EVENT(entries, event, stream)                                   \
  CUDA_CHECK(entries, "GetCudaEvent", GetCudaEvent(&event))                    \
  CUDA_CHECK(entries, "cudaEventRecord", cudaEventRecord(event, stream))

#define RELEASE_EVENT(entries, event)                                          \
  CUDA_CHECK(entries, "ReleaseCudaEvent", ReleaseCudaEvent(event))
#endif

#define ACTIVITY_START_ALL(entries, timeline, activity)                        \
  {                                                                            \
    for (auto it = entries.begin(); it != entries.end(); it++) {               \
      timeline.ActivityStart(it->tensor_name, activity);                       \
    }                                                                          \
  }

#define ACTIVITY_END_ALL(entries, timeline)                                    \
  {                                                                            \
    for (auto it = entries.begin(); it != entries.end(); it++) {               \
      timeline.ActivityEnd(it->tensor_name);                                   \
    }                                                                          \
  }



// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce() {

}

} // for namespace {

Status CheckInitialized() {
  if (false) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void shouter_init() { InitializeHorovodOnce(); }

int shouter_rank() {
    return 1;
}

int shouter_local_rank() {
    return 1;
}

int shouter_size() {
    return 1;
}

int shouter_local_size() {
    return 1;
}

int shouter_mpi_threads_supported() {
    return 1;
}
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  /*
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }*/
  return Status::OK();
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  /*MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
  */
  return Status::OK();
}


// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  /*
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(MPIRequest::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); i++) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
   */
  return Status::OK();
}

} // namespace common
} // namespace horovod
