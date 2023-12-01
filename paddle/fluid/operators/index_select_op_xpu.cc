//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/index_select_op.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class IndexSelectXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.Input<LoDTensor>("X");
    auto index = ctx.Input<LoDTensor>("Index");
    auto out = ctx.Output<LoDTensor>("Out");
    auto dim = ctx.Attr<int>("dim");

    auto place = ctx.GetPlace();
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(place);

    int index_len = index->dims()[0];

    const auto& index_type = index->dtype();
    bool index_type_match =
        index_type == phi::DataType::INT64 || index_type == phi::DataType::INT32;
    PADDLE_ENFORCE_EQ(index_type_match,
                      true,
                      phi::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          index_type,
                          phi::DataType::INT32,
                          phi::DataType::INT64));

    // static int target_id = std::getenv("HSQ_XPURT_TARGET_DEVICE")!=NULL ?
    //                         std::stoi(std::string(std::getenv("HSQ_XPURT_TARGET_DEVICE"))) :
    //                         0;
    // int dev_id = ctx.GetPlace().GetDeviceId();
    // // if(dev_id == target_id) {
    // //   printf("[hsq] input shape: %d, %d\n", (int)x->dims()[0], (int)x->dims()[1]);
    // //   printf("[hsq] index_len: %d\n", index_len);
    // //   printf("[hsq] out shape: %d, %d\n", (int)out->dims()[0], (int)out->dims()[1]);
    // // }

    // auto cpu_device_ctx = platform::DeviceContextPool::Instance().Get(phi::CPUPlace());
    // framework::ExecutionContext cpu_execution_ctx(ctx.GetOp(), ctx.scope(), *cpu_device_ctx, ctx.Context());

    // LoDTensor x_cpu_copy;
    // framework::TensorCopySync(*x, platform::CPUPlace(), &x_cpu_copy);
    // LoDTensor index_cpu_copy;
    // framework::TensorCopySync(*index, platform::CPUPlace(), &index_cpu_copy);
    // LoDTensor out_cpu_copy;
    // framework::TensorCopySync(*out, platform::CPUPlace(), &out_cpu_copy);

    // if (index_type == phi::DataType::INT32) {
    //   IndexSelectInner<DeviceContext, T, int>(cpu_execution_ctx, &x_cpu_copy, index_cpu_copy, &out_cpu_copy, dim);
    // } else if (index_type == phi::DataType::INT64) {
    //   IndexSelectInner<DeviceContext, T, int64_t>(cpu_execution_ctx, &x_cpu_copy, index_cpu_copy, &out_cpu_copy, dim);
    // }

    int r = -1;
    std::vector<int> xshape = phi::vectorize<int>(x->dims());
    if (index_type == phi::DataType::INT64) {
        const int64_t* index_data = index->data<int64_t>();
        r = xpu::gather<T, int64_t>(xpu_context, x_data, index_data, out_data, xshape, index_len, dim);
    } else {
        const int* index_data = index->data<int>();
        r = xpu::gather<T, int>(xpu_context, x_data, index_data, out_data, xshape, index_len, dim);
    }

    // LoDTensor out_ref_cpu_copy;
    // framework::TensorCopySync(*out, platform::CPUPlace(), &out_ref_cpu_copy);
    // bool correct = true;
    // float diff = 1e-5;
    // for (int i = 0; i < out_ref_cpu_copy.numel(); i++) {
    //   T* ref_data = out_ref_cpu_copy.data<T>();
    //   T* cpu_data = out_cpu_copy.data<T>();
    //   if(std::abs(*(ref_data + i) - *(cpu_data+i)) > diff) {
    //     correct = false;
    //     printf("[hsq] error in %d, out_ref_cpu_copy[%d]=%f, out_cpu_copy[%d]=%f\n", i, i, *(ref_data+i), i, *(cpu_data+i));
    //     break;
    //   }
    // }
    // if(dev_id == target_id) {
    //   if(correct) {
    //     printf("[hsq] index_select op test passed\n");
    //   }
    // }

    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The index_select XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class IndexSelectGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.Input<LoDTensor>("X");
    auto index = ctx.Input<LoDTensor>("Index");
    auto out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto dim = ctx.Attr<int>("dim");

    auto place = ctx.GetPlace();
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

    const auto& index_type = index->dtype();
    bool index_type_match =
        index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match,
                      true,
                      phi::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          index_type,
                          phi::DataType::INT32,
                          phi::DataType::INT64));

    const T* x_data = x->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    T* x_grad_data = x_grad->mutable_data<T>(place);

    // auto cpu_device_ctx = platform::DeviceContextPool::Instance().Get(phi::CPUPlace());
    // framework::ExecutionContext cpu_execution_ctx(ctx.GetOp(), ctx.scope(), *cpu_device_ctx, ctx.Context());

    // LoDTensor out_grad_cpu_copy;
    // framework::TensorCopySync(*out_grad, platform::CPUPlace(), &out_grad_cpu_copy);
    // LoDTensor index_cpu_copy;
    // framework::TensorCopySync(*index, platform::CPUPlace(), &index_cpu_copy);
    // LoDTensor x_grad_cpu_copy;
    // framework::TensorCopySync(*x_grad, platform::CPUPlace(), &x_grad_cpu_copy);
    // if (index_type == phi::DataType::INT32) {
    //   IndexSelectGradInner<phi::CPUContext, T, int>(cpu_execution_ctx, out_grad_cpu_copy, index_cpu_copy, &x_grad_cpu_copy, dim);
    // } else if (index_type == phi::DataType::INT64) {
    //   IndexSelectGradInner<phi::CPUContext, T, int64_t>(
    //       cpu_execution_ctx, out_grad_cpu_copy, index_cpu_copy, &x_grad_cpu_copy, dim);
    // }

    int r = -1;
    std::vector<int64_t> out_grad_shape = phi::vectorize<int64_t>(out_grad->dims());
    std::vector<int64_t> x_grad_shape = phi::vectorize<int64_t>(x_grad->dims());

    // static int target_id = std::getenv("HSQ_XPURT_TARGET_DEVICE")!=NULL ?
    //                         std::stoi(std::string(std::getenv("HSQ_XPURT_TARGET_DEVICE"))) :
    //                         0;
    // int dev_id = ctx.GetPlace().GetDeviceId();
    // // if(dev_id == target_id) {
    // //   printf("[hsq] out_grad_shape:[");
    // //   for(int i = 0; i < (int)out_grad_shape.size(); i++) {
    // //     printf("%d, ", (int)out_grad_shape[i]);
    // //   }
    // //   printf("]\n");

    // //   printf("[hsq] x_grad_shape:[");
    // //   for(int i = 0; i < (int)x_grad_shape.size(); i++) {
    // //     printf("%d, ", (int)x_grad_shape[i]);
    // //   }
    // //   printf("]\n");
    // // }
    if (index_type == phi::DataType::INT64) {
        const int64_t* index_data = index->data<int64_t>();
        r = xpu::index_select_grad<T, int64_t>(xpu_context,
                                               x_data,
                                               index_data,
                                               out_grad_data,
                                               dim,
                                               x_grad_data,
                                               out_grad_shape,
                                               x_grad_shape);
    } else {
        const int* index_data = index->data<int>();
        r = xpu::index_select_grad<T, int>(xpu_context,
                                           x_data,
                                           index_data,
                                           out_grad_data,
                                           dim,
                                           x_grad_data,
                                           out_grad_shape,
                                           x_grad_shape);
    }

    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The index_select_grad XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));

    // LoDTensor x_grad_ref_cpu_copy;
    // framework::TensorCopySync(*x_grad, platform::CPUPlace(), &x_grad_ref_cpu_copy);
    // bool correct = true;
    // float diff = 1e-5;
    // for (int i = 0; i < x_grad_ref_cpu_copy.numel(); i++) {
    //   T* ref_data = x_grad_ref_cpu_copy.data<T>();
    //   T* cpu_data = x_grad_cpu_copy.data<T>();
    //   if(std::abs(*(ref_data + i) - *(cpu_data+i)) > diff) {
    //     correct = false;
    //     printf("[hsq] error in %d, out_ref_cpu_copy[%d]=%f, out_cpu_copy[%d]=%f\n", i, i, *(ref_data+i), i, *(cpu_data+i));
    //     break;
    //   }
    // }

    // if(dev_id == target_id) {
    //   if(correct) {
    //     printf("[hsq] index_select_grad op test passed\n");
    //   }
    // }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(index_select, ops::IndexSelectXPUKernel<paddle::platform::XPUDeviceContext, float>)

REGISTER_OP_XPU_KERNEL(index_select_grad, ops::IndexSelectGradXPUKernel<paddle::platform::XPUDeviceContext, float>)
