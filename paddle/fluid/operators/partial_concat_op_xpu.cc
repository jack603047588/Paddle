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

#include "paddle/fluid/operators/partial_concat_op.h"

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class PartialConcatXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_vars = ctx.MultiInput<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE_EQ(in_vars[0] != nullptr,
                      true,
                      platform::errors::InvalidArgument(
                          "The input of partial concat should not be null."));

    auto input_dim = in_vars[0]->dims();
    PADDLE_ENFORCE_EQ(input_dim.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Only supports 2-D array with batch size in the 1st "
                          "dimension and data in the 2nd."));
    auto in_size = input_dim[1];
    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);

    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) {
      partial_len = in_size - start_index;
    }
    //TODO: what if partial_len > in_size
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

    int in_num = in_vars.size();
    int batch_size = input_dim[0];
    // int out_batch_len = partial_len * in_num;

    std::vector<framework::LoDTensor> tmp_tensors(in_num);
    std::vector<const T*> tmp_tensors_data(in_num);
    std::vector<std::vector<int>> tmp_outs_shape(in_num);
    for (size_t i = 0; i < in_vars.size(); i++) {
        tmp_tensors[i].Resize(phi::make_ddim({batch_size, partial_len}));
        tmp_tensors_data[i] = tmp_tensors[i].mutable_data<T>(ctx.GetPlace());

        tmp_outs_shape[i] = std::vector<int>({batch_size, partial_len});

        const T* input_data = in_vars[i]->data<T>();

        std::vector<int> xshape = phi::vectorize<int>(in_vars[i]->dims());
        std::vector<int> starts = {0, start_index};
        std::vector<int> ends = {batch_size, start_index + partial_len + 1};//要截取的x的每个维度的终止坐标(不包含)

        int r = xpu::slice<T>(xpu_context,
                              input_data,
                              const_cast<T*>(tmp_tensors_data[i]),
                              xshape,
                              starts,
                              ends);
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The partial_concat XPU OP's slice return wrong value[%d %s]",
                            r, XPUAPIErrorMsg[r]));
    }

    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    // static int target_id = std::getenv("HSQ_XPURT_TARGET_DEVICE")!=NULL ?
    //                         std::stoi(std::string(std::getenv("HSQ_XPURT_TARGET_DEVICE"))) :
    //                         0;
    // int dev_id = ctx.GetPlace().GetDeviceId();
    // // if(dev_id == target_id) {
    // //     printf("[hsq] in_vars.size(): %d, start_index: %d, partial_len: %d\n", in_num, start_index, partial_len);
    // //     printf("[hsq] input shape: ");
    // //     for (size_t i = 0; i < in_vars.size(); ++i) {
    // //         printf("[%d, %d], ", (int)in_vars[i]->dims()[0], (int)in_vars[i]->dims()[1]);
    // //     }
    // //     printf("]\n");
    // // }
    // // auto cpu_device_ctx = platform::DeviceContextPool::Instance().Get(phi::CPUPlace());
    // std::vector<Tensor> x_cpu_copys(in_num);
    // for (size_t i = 0; i < in_vars.size(); i++) {
    //     framework::TensorCopySync(*(in_vars[i]), platform::CPUPlace(), &(x_cpu_copys[i]));
    // }
    // Tensor out_cpu_copy;
    // framework::TensorCopySync(*out, platform::CPUPlace(), &out_cpu_copy);
    // T* out_cpu_data = out_cpu_copy.data<T>();
    // for (size_t i = 0; i < in_vars.size(); ++i) {
    //   for (int j = 0; j < batch_size; ++j) {
    //     const T* in_data = x_cpu_copys[i].data<T>();
    //     memcpy(out_cpu_data + out_batch_len * j + partial_len * i,
    //            in_data + in_size * j + start_index,
    //            partial_len * sizeof(T));
    //   }
    // }

    int axis = 1;
    int r = xpu::concat<T>(xpu_context,
                           tmp_tensors_data,
                           out_data,
                           tmp_outs_shape,
                           axis);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External(
                        "The partial_concat XPU OP's concat return wrong value[%d %s]",
                        r, XPUAPIErrorMsg[r]));

    // Tensor out_ref_cpu_copy;
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
    //     printf("[hsq] partial_concat op test passed\n");
    //   }
    // }
  }
};

template <typename DeviceContext, typename T>
class PartialConcatGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto xs_grad = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(ins[0] != nullptr,
                      true,
                      platform::errors::InvalidArgument(
                          "The input of partial concat should not be null."));
    // all parameters
    int batch_size = ins[0]->dims()[0];
    int in_size = ins[0]->dims()[1];
    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);
    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) {
        partial_len = in_size - start_index;
    }

    auto in_num = ins.size();

    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

    std::vector<framework::LoDTensor> tmp_tensors(in_num);
    std::vector<const T*> tmp_tensors_data(in_num);

    // std::vector<Tensor> xs_grad_cpu_copys(in_num);
    // std::vector<Tensor> xs_grad_ref_cpu_copys(in_num);

    const T* out_grad_data = out_grad->data<T>();
    for (size_t i = 0; i < in_num; i++) {
        tmp_tensors[i].Resize(phi::make_ddim({batch_size, partial_len}));
        tmp_tensors_data[i] = tmp_tensors[i].mutable_data<T>(ctx.GetPlace());

        std::vector<int> xshape = phi::vectorize<int>(out_grad->dims());
        std::vector<int> starts = {0, int(partial_len * i)};
        std::vector<int> ends = {batch_size, int(partial_len * i + partial_len + 1)};//要截取的x的每个维度的终止坐标(不包含)

        int r = xpu::slice<T>(xpu_context,
                              out_grad_data,
                              const_cast<T*>(tmp_tensors_data[i]),
                              xshape,
                              starts,
                              ends);
        PADDLE_ENFORCE_EQ(
            r,
            xpu::Error_t::SUCCESS,
            platform::errors::External("The partial_concat_grad XPU OP's slice "
                                       "return wrong value[%d %s]",
                                       r,
                                       XPUAPIErrorMsg[r]));

        std::vector<int> tmp_shape = {batch_size, partial_len};
        std::vector<int> pad_left = {0, start_index};
        std::vector<int> pad_right = {0, in_size - start_index - partial_len};
        T* xs_grad_data = xs_grad[i]->mutable_data<T>(ctx.GetPlace());

        // framework::TensorCopySync(*(xs_grad[i]), platform::CPUPlace(), &(xs_grad_cpu_copys[i]));

        r = xpu::pad<T>(xpu_context,
                        tmp_tensors_data[i],
                        xs_grad_data,
                        tmp_shape,
                        pad_left,
                        pad_right,
                        T(0));
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The partial_concat_grad XPU OP's pad return wrong value[%d %s]",
                            r, XPUAPIErrorMsg[r]));

        // framework::TensorCopySync(*(xs_grad[i]), platform::CPUPlace(), &(xs_grad_ref_cpu_copys[i]));
    }


    // auto grad_batch_len = partial_len * in_num;
    // auto all_length = grad_batch_len * batch_size;
    // Tensor out_grad_cpu_copy;
    // framework::TensorCopySync(*out_grad, platform::CPUPlace(), &out_grad_cpu_copy);

    // // initialize
    // auto& place =
    //     *ctx.template device_context<phi::CPUContext>().eigen_device();
    // for (size_t i = 0; i < xs_grad_cpu_copys.size(); ++i) {
    // //   xs_grad_cpu_copys[i]->mutable_data<T>(ctx.GetPlace());
    //   auto dxt = framework::EigenVector<T>::Flatten(xs_grad_cpu_copys[i]);
    //   dxt.device(place) = dxt.constant(static_cast<T>(0));
    // }

    // auto* out_grad_t = out_grad_cpu_copy.data<T>();
    // for (size_t id = 0; id < all_length; id += partial_len) {
    //   int bs_id = id / grad_batch_len;
    //   int bs_index = id % grad_batch_len;
    //   int var_id = bs_index / partial_len;
    //   auto* out_t = xs_grad_ref_cpu_copys[var_id].data<T>();
    //   memcpy(out_t + bs_id * in_size + start_index,
    //          out_grad_t + id,
    //          partial_len * sizeof(T));
    // }

    // bool correct = true;
    // float diff = 1e-5;
    // for (size_t i = 0; i < in_num; i++) {
    //     T* ref_data = xs_grad_ref_cpu_copys[i].data<T>();
    //     T* cpu_data = xs_grad_cpu_copys[i].data<T>();
    //     for (int j = 0; j < xs_grad_cpu_copys[i].numel(); j++) {

    //         if(std::abs(*(ref_data + j) - *(cpu_data+j)) > diff) {
    //             correct = false;
    //             printf("[hsq] error in %d, out_ref_cpu_copy[%d]=%f, out_cpu_copy[%d]=%f\n", j, j, *(ref_data+j), j, *(cpu_data+j));
    //             break;
    //         }
    //     }
    // }
    // static int target_id = std::getenv("HSQ_XPURT_TARGET_DEVICE")!=NULL ?
    //                         std::stoi(std::string(std::getenv("HSQ_XPURT_TARGET_DEVICE"))) :
    //                         0;
    // int dev_id = ctx.GetPlace().GetDeviceId();
    // if(dev_id == target_id) {
    //   if(correct) {
    //     printf("[hsq] partial_concat_grad op test passed\n");
    //   }
    // }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(partial_concat, ops::PartialConcatXPUKernel<paddle::platform::XPUDeviceContext, float>)
REGISTER_OP_XPU_KERNEL(partial_concat_grad, ops::PartialConcatGradXPUKernel<paddle::platform::XPUDeviceContext, float>)
