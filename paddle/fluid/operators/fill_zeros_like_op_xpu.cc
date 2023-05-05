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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/core/lod_utils.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillZerosLikeXPUKernel : public framework::OpKernel<T> {
    // using XPUType = typename XPUTypeTrait<T>::Type;
    using XPUInTDType = typename XPUTypeTrait<T>::Type;

   public:

    void Compute(const framework::ExecutionContext& context) const override {
        // auto* x = context.Input<framework::Tensor>("X");
        auto* out = context.Output<framework::Tensor>("Out");
        out->mutable_data<T>(context.GetPlace());

        VLOG(0) << "##yeeland_fill_zeros_like_XPU##";
        auto& dev_ctx = context.template device_context<DeviceContext>();

        auto out_data = reinterpret_cast<XPUInTDType*>(out->data<T>());
        int ret =
            xpu::constant(dev_ctx.x_context(), out_data, out->numel(), static_cast<XPUInTDType>(0));

        PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                          phi::errors::External("XPU constant API return wrong value[%d %s].", ret,
                                                XPUAPIErrorMsg[ret]));
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    fill_zeros_like,
    ops::FillZerosLikeXPUKernel<paddle::platform::XPUDeviceContext, float>
);
#endif
