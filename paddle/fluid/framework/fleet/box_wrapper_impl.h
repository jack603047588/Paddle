/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#ifdef PADDLE_WITH_BOX_PS
#include <glog/logging.h>
#include <vector>

// The producer side.
#include <scalopus_tracing/tracing.h>
#include <scalopus_transport/transport_loopback.h>
// The catapult recorder side.
#include <scalopus_catapult/catapult_recorder.h>
#include <scalopus_general/endpoint_manager_poll.h>
#include <scalopus_general/general_provider.h>
#include <scalopus_tracing/native_trace_provider.h>

DECLARE_bool(enable_pullpush_dedup_keys);

namespace paddle {
namespace framework {

void BoxWrapper::PullSparseCaseCPU(const paddle::platform::Place& place,
                                    const std::vector<const uint64_t*>& keys,
                                    const std::vector<float*>& values,
                                    const std::vector<int64_t>& slot_lengths,
                                    const int hidden_size,
                                    const int expand_embed_dim,
                                    const int skip_offset, bool expand_only) {
  //  VLOG(3) << "Begin PullSparse";
  int device_id = GetPlaceDeviceId(place);
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_pull_timer;
  platform::Timer& pull_boxps_timer = dev.boxps_pull_timer;
  platform::Timer& pull_dedup_timer = dev.pull_dedup_timer;
  all_timer.Resume();

  int slot_num = static_cast<int>(slot_lengths.size());
  int64_t* slot_lens = reinterpret_cast<int64_t*>(
      dev.slot_lens.mutable_data<int64_t>({(slot_num + 1), 1}, place));
  int64_t total_length = 0;
  slot_lens[0] = 0;
  for (int i = 0; i < slot_num; i++) {
    total_length += slot_lengths[i];
    slot_lens[i + 1] = total_length;
  }
  dev.total_key_length = total_length;

    uint64_t* total_keys =
      reinterpret_cast<uint64_t*>(dev.keys_tensor.mutable_data<int64_t>(
          {static_cast<int64_t>(total_length * 2), 1}, place));
  int* key2slot = reinterpret_cast<int*>(dev.keys2slot.mutable_data<int>(
      {static_cast<int64_t>(total_length * 5), 1}, place));
  int* total_dims = reinterpret_cast<int*>(
      dev.dims_tensor.mutable_data<int>({total_length, 1}, place));

  dev.copy_keys_timer.Resume();
  this->CopyCPUKeys(place, keys, total_keys, slot_lens, slot_num,
                    static_cast<int>(total_length), key2slot);
  dev.copy_keys_timer.Pause();

  // dedup keys pull
  uint32_t* d_restore_idx =
      reinterpret_cast<uint32_t*>(&key2slot[total_length]);
  uint32_t* d_sorted_idx =
      reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
  uint32_t* d_offset = reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
  uint32_t* d_merged_cnts =
      reinterpret_cast<uint32_t*>(&d_offset[total_length]);
  uint64_t* d_merged_keys =
      reinterpret_cast<uint64_t*>(&total_keys[total_length]);

  pull_dedup_timer.Resume();
  int dedup_size =
      boxps_ptr_->DedupKeysAndFillIdx(device_id, total_length,
                                      total_keys,     // input
                                      d_merged_keys,  // output
                                      d_restore_idx,  // pull fill idx
                                      d_sorted_idx,   // sort old idx
                                      d_offset,       // offset
                                      d_merged_cnts);
  pull_dedup_timer.Pause();
  PADDLE_ENFORCE_GT(dedup_size, 0,
                    platform::errors::PreconditionNotMet(
                        "dedup keys need more than zero failed in BoxPS."));
  dev.dedup_key_length = dedup_size;

  int64_t total_bytes = dedup_size * feature_pull_size_;
  void* total_values_gpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

  pull_boxps_timer.Resume();

  int ret = boxps_ptr_->PullSparseGPU(d_merged_keys,
                                      reinterpret_cast<void*>(total_values_gpu),
                                      static_cast<int>(dedup_size), device_id);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "PullSparseGPU failed in BoxPS."));
  pull_boxps_timer.Pause();

  dev.copy_values_timer.Resume();
  this->CopyForPullCPU(place, keys, values, total_values_gpu, slot_lens,
                       slot_num, key2slot, hidden_size, expand_embed_dim,
                       total_length, total_dims, skip_offset, expand_only,
                       d_restore_idx);
  dev.copy_values_timer.Pause();

  all_timer.Pause();
}

void BoxWrapper::PullSparseCaseGPU(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<float*>& values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int skip_offset, bool expand_only) {
#if defined(PADDLE_WITH_CUDA)
  //  VLOG(3) << "Begin PullSparse";
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_pull_timer;
  platform::Timer& pull_boxps_timer = dev.boxps_pull_timer;
  platform::Timer& pull_dedup_timer = dev.pull_dedup_timer;
  all_timer.Resume();

    // construct slot_level lod info
  std::vector<int64_t> slot_lengths_lod;
  slot_lengths_lod.push_back(0);

  int64_t total_length = 0;
  int slot_num = static_cast<int>(slot_lengths.size());
  for (int i = 0; i < slot_num; i++) {
    total_length += slot_lengths[i];
    slot_lengths_lod.push_back(total_length);
  }
  dev.total_key_length = total_length;

  auto stream = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();

  boxps::FeaturePullOffset* pull_offset = nullptr;
  if (dev.pull_offset.memory_size() == 0) {
    pull_offset = dev.pull_offset.mutable_data<boxps::FeaturePullOffset>(
        sizeof(boxps::FeaturePullOffset), place);
    cudaMemcpyAsync(pull_offset, &pull_info_, sizeof(boxps::FeaturePullOffset),
                    cudaMemcpyHostToDevice, stream);
  } else {
    pull_offset = dev.pull_offset.data<boxps::FeaturePullOffset>();
  }

  uint64_t* total_keys = nullptr;
  int* key2slot = nullptr;
  if (FLAGS_enable_pullpush_dedup_keys) {
    total_keys =
        reinterpret_cast<uint64_t*>(dev.keys_tensor.mutable_data<int64_t>(
            {static_cast<int64_t>(total_length * 2), 1}, place));
    key2slot = reinterpret_cast<int*>(dev.keys2slot.mutable_data<int>(
        {static_cast<int64_t>(total_length * 5), 1}, place));
  } else {
    total_keys = reinterpret_cast<uint64_t*>(
        dev.keys_tensor.mutable_data<int64_t>({total_length, 1}, place));
    key2slot = reinterpret_cast<int*>(
        dev.keys2slot.mutable_data<int>({total_length, 1}, place));
  }

  int* total_dims = reinterpret_cast<int*>(
      dev.dims_tensor.mutable_data<int>({total_length, 1}, place));

  uint64_t** gpu_keys = dev.keys_ptr_tensor.mutable_data<uint64_t*>(
      static_cast<int>(keys.size() * sizeof(uint64_t*)), place);

  int64_t* slot_lens = reinterpret_cast<int64_t*>(
      dev.slot_lens.mutable_data<int64_t>({(slot_num + 1), 1}, place));

  dev.copy_keys_timer.Resume();
  cudaMemcpyAsync(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(slot_lens, slot_lengths_lod.data(),
                  slot_lengths_lod.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice, stream);
  this->CopyKeys(place, gpu_keys, total_keys, slot_lens, slot_num,
                 static_cast<int>(total_length), key2slot);
  dev.copy_keys_timer.Pause();

  // dedup keys pull
  if (FLAGS_enable_pullpush_dedup_keys) {
    uint32_t* d_restore_idx =
        reinterpret_cast<uint32_t*>(&key2slot[total_length]);
    uint32_t* d_sorted_idx =
        reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
    uint32_t* d_offset =
        reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
    uint32_t* d_merged_cnts =
        reinterpret_cast<uint32_t*>(&d_offset[total_length]);
    uint64_t* d_merged_keys =
        reinterpret_cast<uint64_t*>(&total_keys[total_length]);

    pull_dedup_timer.Resume();
    int dedup_size =
        boxps_ptr_->DedupKeysAndFillIdx(device_id, total_length,
                                        total_keys,     // input
                                        d_merged_keys,  // output
                                        d_restore_idx,  // pull fill idx
                                        d_sorted_idx,   // sort old idx
                                        d_offset,       // offset
                                        d_merged_cnts);
    pull_dedup_timer.Pause();

    PADDLE_ENFORCE_GT(dedup_size, 0,
                      platform::errors::PreconditionNotMet(
                          "dedup keys need more than zero failed in BoxPS."));
    dev.dedup_key_length = dedup_size;

    int64_t total_bytes = dedup_size * feature_pull_size_;
    void* total_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

    pull_boxps_timer.Resume();

    int ret = boxps_ptr_->PullSparseGPU(
        d_merged_keys, reinterpret_cast<void*>(total_values_gpu),
        static_cast<int>(dedup_size), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();
    // values.size() not sure equal slot_num
    float** gpu_values = dev.values_ptr_tensor.mutable_data<float*>(
        static_cast<int>(values.size() * sizeof(float*)), place);

    dev.copy_values_timer.Resume();
    cudaMemcpyAsync(gpu_values, values.data(), values.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);

    this->CopyForPull(place, gpu_keys, gpu_values, total_values_gpu,
                      pull_offset, slot_lens, slot_num, key2slot, hidden_size,
                      expand_embed_dim, total_length, total_dims, skip_offset,
                      expand_only, d_restore_idx);
    dev.copy_values_timer.Pause();
  } else {
    int64_t total_bytes = total_length * feature_pull_size_;
    void* total_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

    pull_boxps_timer.Resume();

    int ret = boxps_ptr_->PullSparseGPU(
        total_keys, reinterpret_cast<void*>(total_values_gpu),
        static_cast<int>(total_length), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();
    // values.size() not sure equal slot_num
    float** gpu_values = dev.values_ptr_tensor.mutable_data<float*>(
        static_cast<int>(values.size() * sizeof(float*)), place);

    dev.copy_values_timer.Resume();
    cudaMemcpyAsync(gpu_values, values.data(), values.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);

    this->CopyForPull(place, gpu_keys, gpu_values, total_values_gpu,
                      pull_offset, slot_lens, slot_num, key2slot, hidden_size,
                      expand_embed_dim, total_length, total_dims, skip_offset,
                      expand_only);
    dev.copy_values_timer.Pause();
  }
  all_timer.Pause();
#endif
}

void BoxWrapper::PullSparseCaseXPU(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<float*>& values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int skip_offset, bool expand_only) {
#ifdef PADDLE_WITH_XPU_KP
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto ctx_xpu = static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  static bool use_l3_tensor = std::getenv("XPU_PADDLE_L3_TENSOR")!=NULL ?
                    (std::strcmp(std::getenv("XPU_PADDLE_L3_TENSOR"), "1") == 0 ? true:false) :
                    false;
  phi::Place l3_place =
   static_cast<platform::XPUDeviceContext*>(dev_ctx)->GetL3Place();
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];

  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  int slot_num = static_cast<int>(slot_lengths.size());
  dev.total_key_length = total_length;

  VLOG(3) << "Begine BoxPs PullSparse";
  xpu::ctx_guard RAII_GUARD(ctx_xpu);

  int64_t total_bytes = total_length * feature_pull_size_;
  void* total_values_xpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

  TRACE_SCOPE_START("copy keys", xpu_wait(ctx_xpu->xpu_stream));
  VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
  LoDTensor& total_keys_tensor = dev.keys_tensor;
  uint32_t* total_keys;
  if(use_l3_tensor) {
    total_keys = reinterpret_cast<uint32_t*>(
        total_keys_tensor.mutable_data<int32_t>({total_length, 1}, l3_place));
  } else {
    total_keys = reinterpret_cast<uint32_t*>(
        total_keys_tensor.mutable_data<int32_t>({total_length, 1}, place));
  }
  int* key2slot = nullptr;
  key2slot = reinterpret_cast<int*>(
      dev.keys2slot.mutable_data<int>({total_length, 1}, place));

  // construct slot_level lod info
  std::vector<int64_t> slot_lengths_lod(slot_num + 1, 0);
  for (int i = 1; i <= slot_num ; i++) {
    slot_lengths_lod[i] = slot_lengths_lod[i - 1] + slot_lengths[i - 1];
  }

  int* total_dims = reinterpret_cast<int*>(
      dev.dims_tensor.mutable_data<int>({total_length, 1}, place));

  uint64_t** xpu_keys = dev.keys_ptr_tensor.mutable_data<uint64_t*>(
      static_cast<int>(keys.size() * sizeof(uint64_t*)), place);
  int64_t* slot_lens = reinterpret_cast<int64_t*>(
      dev.slot_lens.mutable_data<int64_t>({(slot_num + 1), 1}, place));
  xpu_memcpy(xpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
                  XPU_HOST_TO_DEVICE);
  xpu_memcpy(slot_lens, slot_lengths_lod.data(),
                  slot_lengths_lod.size() * sizeof(int64_t),
                  XPU_HOST_TO_DEVICE);

  TRACE_SCOPE_START("CopyKeys", xpu_wait(ctx_xpu->xpu_stream));
  box_wrapper_kernel_->CopyKeys(place, xpu_keys, total_keys, slot_lens,
                  static_cast<int>(slot_lengths.size()),
                  static_cast<int>(total_length), key2slot);
  VLOG(3) << "Begin call PullSparseXPU in BoxPS, dev: " << device_id
            << " len: " << total_length;
  TRACE_SCOPE_END("CopyKeys", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("copy keys", xpu_wait(ctx_xpu->xpu_stream));

  TRACE_SCOPE_START("PullSparseXPU", xpu_wait(ctx_xpu->xpu_stream));
  pull_boxps_timer.Start();
  boxps_ptr_->PullSparseXPU(total_keys, total_values_xpu,
      static_cast<int>(total_length), device_id);
  pull_boxps_timer.Pause();
  TRACE_SCOPE_END("PullSparseXPU", xpu_wait(ctx_xpu->xpu_stream));

  VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
          << "]";

  TRACE_SCOPE_START("pull copy", xpu_wait(ctx_xpu->xpu_stream));
  boxps::FeaturePullOffset* pull_offset = nullptr;
  if (dev.pull_offset.memory_size() == 0) {
    pull_offset = dev.pull_offset.mutable_data<boxps::FeaturePullOffset>(
        sizeof(boxps::FeaturePullOffset), place);
    xpu_memcpy(pull_offset, &pull_info_, sizeof(boxps::FeaturePullOffset),
                    XPU_HOST_TO_DEVICE);
  } else {
    pull_offset = dev.pull_offset.data<boxps::FeaturePullOffset>();
  }

  float** xpu_values = dev.values_ptr_tensor.mutable_data<float*>(
        static_cast<int>(values.size() * sizeof(float*)), place);
  xpu_memcpy(xpu_values, values.data(), values.size() * sizeof(float*),
                  XPU_HOST_TO_DEVICE);

  TRACE_SCOPE_START("CopyForPull", xpu_wait(ctx_xpu->xpu_stream));
  box_wrapper_kernel_->CopyForPull(place, xpu_keys, (float**)values.data(), total_values_xpu,
                      pull_offset, slot_lengths_lod.data(), slot_num, key2slot, hidden_size,
                      expand_embed_dim, total_length, total_dims, skip_offset,
                      expand_only);
  TRACE_SCOPE_END("CopyForPull", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("pull copy", xpu_wait(ctx_xpu->xpu_stream));
  all_timer.Pause();
#endif
}

void BoxWrapper::PullSparseCase(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<float*>& values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int skip_offset, bool expand_only) {
  if (platform::is_cpu_place(place)) {
    PullSparseCaseCPU(place, keys, values, slot_lengths, hidden_size,
                      expand_embed_dim, skip_offset, expand_only);
  } else if (platform::is_gpu_place(place)) {
    PullSparseCaseGPU(place, keys, values, slot_lengths, hidden_size,
                      expand_embed_dim, skip_offset, expand_only);
  } else {
    PullSparseCaseXPU(place, keys, values, slot_lengths, hidden_size,
                      expand_embed_dim, skip_offset, expand_only);
  }
}

void BoxWrapper::PushSparseGradCaseGPU(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths, const int hidden_size,
    const int expand_embed_dim, const int batch_size, const int skip_offset,
    bool expand_only) {
#if defined(PADDLE_WITH_CUDA)
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;

  all_timer.Resume();

  auto stream = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
  uint64_t* total_keys =
      reinterpret_cast<uint64_t*>(dev.keys_tensor.data<int64_t>());
  int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
  int slot_num = static_cast<int>(slot_lengths.size());
  if (!dev.d_slot_vector.IsInitialized()) {
    int* buf_slot_vector = reinterpret_cast<int*>(
        dev.d_slot_vector.mutable_data<int>({slot_num, 1}, place));
    cudaMemcpyAsync(buf_slot_vector, slot_vector_.data(),
                    slot_num * sizeof(int), cudaMemcpyHostToDevice, stream);
  }

  boxps::FeaturePushOffset* push_offset = nullptr;
  if (dev.push_offset.memory_size() == 0) {
    push_offset = dev.push_offset.mutable_data<boxps::FeaturePushOffset>(
        sizeof(boxps::FeaturePushOffset), place);
    cudaMemcpyAsync(push_offset, &push_info_, sizeof(boxps::FeaturePushOffset),
                    cudaMemcpyHostToDevice, stream);
  } else {
    push_offset = dev.push_offset.data<boxps::FeaturePushOffset>();
  }

  const int64_t* slot_lens =
      reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
  const int* d_slot_vector = dev.d_slot_vector.data<int>();
  const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());
  float** gpu_values = dev.values_ptr_tensor.data<float*>();
  cudaMemcpyAsync(gpu_values, grad_values.data(),
                  grad_values.size() * sizeof(float*), cudaMemcpyHostToDevice,
                  stream);

  int64_t total_length = dev.total_key_length;
  // dedup keys pull
  if (FLAGS_enable_pullpush_dedup_keys) {
    const uint32_t* d_restore_idx =
        reinterpret_cast<const uint32_t*>(&key2slot[total_length]);
    const uint32_t* d_sorted_idx =
        reinterpret_cast<const uint32_t*>(&key2slot[total_length * 2]);
    const uint32_t* d_offset =
        reinterpret_cast<const uint32_t*>(&d_sorted_idx[total_length]);
    const uint32_t* d_merged_cnts =
        reinterpret_cast<const uint32_t*>(&d_offset[total_length]);
    uint64_t* d_merged_keys = &total_keys[total_length];

    int64_t dedup_size = dev.dedup_key_length;
    int64_t total_bytes = dedup_size * feature_push_size_;
    void* total_grad_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
    this->CopyForPush(place, gpu_values, total_grad_values_gpu, push_offset,
                      total_length, dedup_size, d_slot_vector, slot_lens,
                      slot_num, hidden_size, expand_embed_dim, batch_size,
                      total_dims, key2slot, skip_offset, expand_only,
                      d_sorted_idx, d_offset, d_merged_cnts, d_restore_idx);

    push_boxps_timer.Resume();
    int ret = boxps_ptr_->PushSparseGPU(
        d_merged_keys, reinterpret_cast<void*>(total_grad_values_gpu),
        static_cast<int>(dedup_size), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PushSparseGPU failed in BoxPS."));
    push_boxps_timer.Pause();
  } else {
    int64_t total_bytes = total_length * feature_push_size_;
    void* total_grad_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
    this->CopyForPush(place, gpu_values, total_grad_values_gpu, push_offset,
                      total_length, 0, d_slot_vector, slot_lens, slot_num,
                      hidden_size, expand_embed_dim, batch_size, total_dims,
                      key2slot, skip_offset, expand_only);

    push_boxps_timer.Resume();
    int ret = boxps_ptr_->PushSparseGPU(
        total_keys, reinterpret_cast<void*>(total_grad_values_gpu),
        static_cast<int>(total_length), device_id);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "PushSparseGPU failed in BoxPS."));
    push_boxps_timer.Pause();
  }
  all_timer.Pause();
#endif
}

void BoxWrapper::PushSparseGradCaseXPU(const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths, const int hidden_size,
    const int expand_embed_dim, const int batch_size, const int skip_offset,
    bool expand_only) {
#ifdef PADDLE_WITH_XPU_KP
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto ctx_xpu = static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  ctx_xpu = ctx_xpu;//avoid unused error without TRACE_PROFILE

  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;

  all_timer.Resume();

  TRACE_SCOPE_START("push copy", xpu_wait(ctx_xpu->xpu_stream));
  int64_t total_length = dev.total_key_length;
  int64_t total_bytes = total_length * feature_push_size_;
  void* total_grad_values_xpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
  uint32_t* total_keys =
      reinterpret_cast<uint32_t*>(dev.keys_tensor.data<int32_t>());
  int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
  int slot_num = static_cast<int>(slot_lengths.size());

  if (!dev.d_slot_vector.IsInitialized()) {
    int* buf_slot_vector = reinterpret_cast<int*>(
        dev.d_slot_vector.mutable_data<int>({slot_num, 1}, place));
    xpu_memcpy(buf_slot_vector, slot_vector_.data(),
                    slot_num * sizeof(int), XPU_HOST_TO_DEVICE);
  }

  boxps::FeaturePushOffset* push_offset = nullptr;
  if (dev.push_offset.memory_size() == 0) {
    push_offset = dev.push_offset.mutable_data<boxps::FeaturePushOffset>(
        sizeof(boxps::FeaturePushOffset), place);
    xpu_memcpy(push_offset, &push_info_, sizeof(boxps::FeaturePushOffset),
        XPU_HOST_TO_DEVICE);
  } else {
    push_offset = dev.push_offset.data<boxps::FeaturePushOffset>();
  }

  auto slot_lengths_lod = slot_lengths;
  for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  const int64_t* slot_lens =
      reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
  const int* slot_vector = dev.d_slot_vector.data<int>();
  const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());
  float** xpu_values = dev.values_ptr_tensor.data<float*>();
  xpu_memcpy(xpu_values, grad_values.data(),
                  grad_values.size() * sizeof(float*), XPU_HOST_TO_DEVICE);
  TRACE_SCOPE_START("CopyForPush's xpu::copy", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("CopyForPush's xpu::copy", xpu_wait(ctx_xpu->xpu_stream));

  TRACE_SCOPE_START("CopyForPush", xpu_wait(ctx_xpu->xpu_stream));

  float* real_grad_values;
  for (int i = 0; i < slot_num; i++) {
    if(grad_values[i] != nullptr) {
      real_grad_values = const_cast<float*>(grad_values[i]);
      break;
    }
  }
  box_wrapper_kernel_->CopyForPush(place, real_grad_values, total_grad_values_xpu,
      push_offset, total_length, slot_vector, slot_lens, slot_num,
      hidden_size, batch_size, total_dims, skip_offset, key2slot);

  push_boxps_timer.Resume();
  TRACE_SCOPE_END("CopyForPush", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("push copy", xpu_wait(ctx_xpu->xpu_stream));

  TRACE_SCOPE_START("PushSparseXPU", xpu_wait(ctx_xpu->xpu_stream));
  int ret = boxps_ptr_->PushSparseXPU(total_keys,
      reinterpret_cast<void*>(total_grad_values_xpu),
      static_cast<int>(total_length), device_id);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                              "PushSparseXPU failed in BoxPS."));
  push_boxps_timer.Pause();
  TRACE_SCOPE_END("PushSparseXPU", xpu_wait(ctx_xpu->xpu_stream));

  all_timer.Pause();

#endif
}

void BoxWrapper::PushSparseGradCaseCPU(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths, const int hidden_size,
    const int expand_embed_dim, const int batch_size, const int skip_offset,
    bool expand_only) {
  int device_id = GetPlaceDeviceId(place);
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;

  all_timer.Resume();

  uint64_t* total_keys =
      reinterpret_cast<uint64_t*>(dev.keys_tensor.data<int64_t>());
  int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
  int slot_num = static_cast<int>(slot_lengths.size());

  const int64_t* slot_lens =
      reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
  const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());

  int64_t total_length = dev.total_key_length;
  // dedup keys pull
  const uint32_t* d_sorted_idx =
      reinterpret_cast<const uint32_t*>(&key2slot[total_length * 2]);
  const uint32_t* d_offset =
      reinterpret_cast<const uint32_t*>(&d_sorted_idx[total_length]);
  const uint32_t* d_merged_cnts =
      reinterpret_cast<const uint32_t*>(&d_offset[total_length]);
  uint64_t* d_merged_keys = &total_keys[total_length];

  int64_t dedup_size = dev.dedup_key_length;
  int64_t total_bytes = dedup_size * feature_push_size_;
  void* total_grad_values_gpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
  this->CopyForPushCPU(place, grad_values, total_grad_values_gpu,
                       slot_vector_.data(), slot_lens, slot_num, hidden_size,
                       expand_embed_dim, dedup_size, batch_size, total_dims,
                       key2slot, skip_offset, expand_only, d_sorted_idx,
                       d_offset, d_merged_cnts);

  push_boxps_timer.Resume();
  int ret = boxps_ptr_->PushSparseGPU(
      d_merged_keys, reinterpret_cast<void*>(total_grad_values_gpu),
      static_cast<int>(dedup_size), device_id);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "PushSparseGPU failed in BoxPS."));
  push_boxps_timer.Pause();

  all_timer.Pause();
}

void BoxWrapper::PushSparseGradCase(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths, const int hidden_size,
    const int expand_embed_dim, const int batch_size, const int skip_offset,
    bool expand_only) {
  if (platform::is_cpu_place(place)) {
    PushSparseGradCaseCPU(place, keys, grad_values, slot_lengths, hidden_size,
                          expand_embed_dim, batch_size, skip_offset,
                          expand_only);
  } else if (platform::is_gpu_place(place)) {
    PushSparseGradCaseGPU(place, keys, grad_values, slot_lengths, hidden_size,
                          expand_embed_dim, batch_size, skip_offset,
                          expand_only);
  } else {
    PushSparseGradCaseXPU(place, keys, grad_values, slot_lengths, hidden_size,
                          expand_embed_dim, batch_size, skip_offset,
                          expand_only);
  }
}

}  // namespace framework
}  // namespace paddle
#endif
