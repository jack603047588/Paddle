// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_XPU
#include <sstream>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_memory_aligment.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "xpu/refactor/math.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SliceMultiTensorOpXPUKernel : public framework::OpKernel<T> {
    using XPUType = typename XPUTypeTrait<T>::Type;
   public:
    void Compute(const framework::ExecutionContext& context) const override {
        auto fuse_tensor = context.Input<framework::LoDTensor>("Input");
        auto in_tensors = context.MultiInput<framework::LoDTensor>("X");
        // Init the continuous space
        auto out_tensors = context.MultiOutput<framework::LoDTensor>("Output");

        int id = context.Attr<int>("id");
        int num = context.Attr<int>("num");

        size_t in_size = in_tensors.size();
        size_t out_size = out_tensors.size();
        // num data
        CHECK(in_size == out_size || out_size / num == in_size);

        // Make the outputs point to the continuous space.
        int64_t numel = fuse_tensor->numel();
        int64_t offset = (id * numel) / num;
        VLOG(1) << "##yeeland_slice_multi_tensor_XPU";

        // fprintf(stdout,
        //         "fuse length: %d(dim: %s), in length: %d(dim: %s), offset: %d, id: %d, num: %d, "
        //         "out length: %d(dim: %s), in size:%d, out size: %d\n",
        //         int(fuse_tensor->numel()), fuse_tensor->dims().to_str().c_str(),
        //         int(in_tensors[0]->numel()), in_tensors[0]->dims().to_str().c_str(), 0, id, num,
        //         int(out_tensors[0]->numel()), out_tensors[0]->dims().to_str().c_str(), int(in_size),
        //         int(out_size));

        auto& fuse_dim = fuse_tensor->dims();
        // adjust fuse
        if (fuse_dim.size() > 1 && fuse_dim[0] != numel) {
            paddle::framework::DDim dim(fuse_dim);
            dim[0] = numel;
            dim[1] = 1;
            const_cast<framework::LoDTensor*>(fuse_tensor)->Resize(dim);
        }

        auto& dev_ctx = context.template device_context<DeviceContext>();
        const XPUType* in_data = reinterpret_cast<const XPUType*>(fuse_tensor->data<T>());
        size_t shape_size = fuse_dim.size();
        std::vector<int> shape(shape_size, 0);
        for (size_t i = 0; i < shape_size; ++i) {
          shape[i] = fuse_dim[i];
        }
        std::vector<int> starts_extension(shape_size, 0);
        std::vector<int> ends_extension(shape_size, 0);
        starts_extension[1] = 0;
        ends_extension[1] = 1;
        for (size_t i = 0; i < out_tensors.size(); ++i) {
            size_t idx = i % in_size;
            auto dim = in_tensors[idx]->dims();
            size_t len = static_cast<size_t>(in_tensors[idx]->numel());
            CHECK(static_cast<int64_t>(offset + len) <= numel)
                << "fuse dim: " << fuse_dim.to_str() << ", dim:" << dim.to_str()
                << ", offset:" << offset << ", len:" << len;
            // slice tensor
            starts_extension[0] = int(offset);
            ends_extension[0] = int(offset + len);
            XPUType* out_data = reinterpret_cast<XPUType*>(out_tensors[i]->mutable_data<T>(dev_ctx.GetPlace()));
            int r = xpu::slice<XPUType>(dev_ctx.x_context(),
                                        in_data, out_data, shape,
                                        starts_extension, ends_extension);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
            // out_tensors[i]->out_data.Resize(dim);
            offset += len;
        }
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(slice_multi_tensor,
                       ops::SliceMultiTensorOpXPUKernel<paddle::platform::XPUDeviceContext, float>,
                      //  ops::SliceMultiTensorOpXPUKernel<paddle::platform::XPUDeviceContext, float16>,
                       ops::SliceMultiTensorOpXPUKernel<paddle::platform::XPUDeviceContext, int>);
#endif
