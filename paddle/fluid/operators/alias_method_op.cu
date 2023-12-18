/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/alias_method_op.h"

#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <sstream>
#include <string>

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

constexpr int CUDA_NUM_THREADS = platform::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0

__global__ void SetData(int64_t* tab, const float* idx, uint64_t n) {
  CUDA_KERNEL_LOOP(i, n) { tab[int64_t(idx[i])] = 1; }
}

__global__ void RejectSampling(float* out, const int num, const float* accept,
                               const float* alias, const size_t len,
                               const int device_id, const int device_num,
                               int64_t* noids, uint64_t seed) {
  // https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions

  curandStatePhilox4_32_10_t state;
  int seq = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(seed, seq, 0, &state);

  CUDA_KERNEL_LOOP(j, num) {
    while (true) {
      float i = curand_uniform(&state);                     // (0, 1]
      int r = std::ceil(curand_uniform(&state) * len) - 1;  // [0, n)

      int s = i <= accept[r] ? r : alias[r];
      // printf("i:%f r:%d acc:%f ali:%f s: %d\n",
      //        i, r, accept[r], alias[r], s);
      // if (s % device_num == device_id && noids[s] == 0) {
      if (noids[s] == 0) {
        out[j] = static_cast<float>(s);
        break;
      }
    }
  }
}

template <typename T>
class AliasMethodCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto place = context.GetPlace();
    int device_id = place.GetDeviceId();
    int device_num = platform::GetGPUDeviceCount();
    auto stream = dynamic_cast<phi::GPUContext*>(
                  platform::DeviceContextPool::Instance().Get(place))
                  ->stream();

    const framework::Tensor* accept =
        context.Input<framework::Tensor>("Accept");
    const framework::Tensor* alias = context.Input<framework::Tensor>("Alias");
    const framework::Tensor* noids = context.Input<framework::Tensor>("Noids");
    framework::Tensor* out = context.Output<framework::Tensor>("Out");
    int num = context.Attr<int>("Num");

    framework::Tensor table;
    int64_t* d_noids =
        table.mutable_data<int64_t>({accept->numel(), 1}, context.GetPlace());
    cudaMemsetAsync(d_noids, 0, accept->numel() * sizeof(int64_t), stream);
    SetData<<<CUDA_BLOCK(noids->numel()), stream>>>(
        d_noids, noids->data<float>(), noids->numel());

    uint64_t seed = std::random_device()();
    float* d_out = out->mutable_data<float>(context.GetPlace());
    RejectSampling<<<CUDA_BLOCK(num), stream>>>(
        d_out, num, accept->data<float>(), alias->data<float>(),
        accept->numel(), device_id, device_num, d_noids, seed);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(alias_method, ops::AliasMethodCUDAKernel<float>)
