/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/masked_data_norm_op.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/data_layout.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

class MaskedDataNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "MaskedDataNorm");
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "MaskedDataNorm");
    OP_INOUT_CHECK(
        ctx->HasInput("BatchSize"), "Input", "BatchSize", "MaskedDataNorm");
    OP_INOUT_CHECK(
        ctx->HasInput("BatchSum"), "Input", "BatchSum", "MaskedDataNorm");
    OP_INOUT_CHECK(ctx->HasInput("BatchSquareSum"),
                   "Input",
                   "BatchSquareSum",
                   "MaskedDataNorm");
    OP_INOUT_CHECK(
        ctx->HasOutput("Means"), "Output", "Means", "MaskedDataNorm");
    OP_INOUT_CHECK(
        ctx->HasOutput("Scales"), "Output", "Scales", "MaskedDataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "MaskedDataNorm");
    bool enable_scale_and_shift =
        ctx->Attrs().Get<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("scale_w"),
          true,
          platform::errors::InvalidArgument(
              "Input(scale_w) of MaskedDataNormOp should not be null."));
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("bias"),
          true,
          platform::errors::InvalidArgument(
              "Input(bias) of MaskedDataNormOp should not be null."));
    }

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5,
                      true,
                      platform::errors::InvalidArgument(
                          "Input X must have 2 to 5 dimensions."));

    const auto m_dims = ctx->GetInputDim("Mask");
    if (m_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          m_dims[1],
          1,
          platform::errors::InvalidArgument(
              "The last dim of mask should be 1 when it is 2D, but we get %d",
              m_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          m_dims.size(),
          1,
          platform::errors::InvalidArgument(
              "The mask should be 1D, when it is not 2D, but we get %d",
              m_dims.size()));
    }

    const int64_t C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize").size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The input dim of BatchSize shouold be 1"));
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum").size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The input dim of BatchSum shouold be 1"));
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum").size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The input dim of BatchSquareSum shouold be 1"));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize")[0],
                        C,
                        platform::errors::InvalidArgument(
                            "The input dim[0] of BatchSize shouold be C"));
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum")[0],
                        C,
                        platform::errors::InvalidArgument(
                            "The input dim[0] of BatchSum shouold be C"));
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum")[0],
                        C,
                        platform::errors::InvalidArgument(
                            "The input dim[0] of BatchSqureSum shouold be C"));
    }

    if (enable_scale_and_shift) {
      auto scale_dim = ctx->GetInputDim("scale_w");
      auto bias_dim = ctx->GetInputDim("bias");

      PADDLE_ENFORCE_EQ(
          scale_dim.size(),
          1UL,
          platform::errors::InvalidArgument("the dimensionof scale"
                                            "must equal to 1. But received: "
                                            "the shape of scale is [%s], "
                                            "the dimensionof scale is [%d]",
                                            scale_dim,
                                            scale_dim.size()));
      PADDLE_ENFORCE_EQ(
          bias_dim.size(),
          1UL,
          platform::errors::InvalidArgument("the dimension of bias"
                                            "must equal to 1. But received: "
                                            "the shape of bias is [%s],"
                                            "the dimension of bias is [%d]",
                                            bias_dim,
                                            bias_dim.size()));

      bool check = true;
      if ((!ctx->IsRuntime()) &&
          (phi::product(scale_dim) <= 0 || phi::product(bias_dim) <= 0)) {
        check = false;
      }

      if (check) {
        PADDLE_ENFORCE_EQ(scale_dim[0],
                          C,
                          platform::errors::InvalidArgument(
                              "the shape of scale must equal to [%d]"
                              "But received: the shape of scale is [%d]",
                              C,
                              scale_dim[0]));
        PADDLE_ENFORCE_EQ(bias_dim[0],
                          C,
                          platform::errors::InvalidArgument(
                              "the shape of bias must equal to [%d]"
                              "But received: the shape of bias is [%d]",
                              C,
                              bias_dim[0]));
      }
    }

    ctx->SetOutputDim("Y", x_dims);
    ctx->SetOutputDim("Means", {C});
    ctx->SetOutputDim("Scales", {C});
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto dn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      dn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSize"),
                      platform::errors::InvalidArgument(
                          "BatchSize input should be of float type"));
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSum"),
                      platform::errors::InvalidArgument(
                          "BatchSum input should be of float type"));
    PADDLE_ENFORCE_EQ(
        dn_param_type,
        OperatorWithKernel::IndicateVarDataType(ctx, "BatchSquareSum"),
        platform::errors::InvalidArgument(
            "BatchSquareSum input should be of float type"));

    bool enable_scale_and_shift = ctx.Attr<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
      PADDLE_ENFORCE_EQ(dn_param_type,
                        OperatorWithKernel::IndicateVarDataType(ctx, "scale_w"),
                        platform::errors::InvalidArgument(
                            "scale_w input should be of float type"));
      PADDLE_ENFORCE_EQ(dn_param_type,
                        OperatorWithKernel::IndicateVarDataType(ctx, "bias"),
                        platform::errors::InvalidArgument(
                            "bias input should be of float type"));
    }
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(
        input_data_type, ctx.GetPlace(), layout, library);
  }
};

class MaskedDataNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-4)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                            true,
                            platform::errors::InvalidArgument(
                                "'epsilon' should be between 0.0 and 0.001."));
        });
    AddAttr<int>("slot_dim",
                 "(int, default -1) Dimension of one slot if set, "
                 "when the input is concated by slot-wise embeddings")
        .SetDefault(-1);
    AddAttr<float>(
        "summary_decay_rate",
        "(float, default 0.9999999) The decay rate when update the summary")
        .SetDefault(0.9999999);
    AddAttr<bool>(
        "enable_scale_and_shift",
        "(bool, default false) Set to true to enable scale and shift such as "
        "batch_norm op")
        .SetDefault(false);
    AddInput("scale_w",
             "scale_w is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddInput("bias",
             "bias is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
    AddAttr<bool>("sync_stats", "(bool, default false) only used in multi-GPU")
        .SetDefault(false);
    AddAttr<bool>("update_norm", "(bool, default true) used in update_norm")
        .SetDefault(true);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddInput("X", "The input tensor");
    AddInput("Mask", "The mask tensor");
    AddInput("BatchSize",
             "BatchSize is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSum",
             "BatchSum is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSquareSum",
             "The global BatchSquareSum (for training) or "
             "estimated BatchSquareSum (for testing)");
    AddOutput("Y", "result after normalization");
    AddOutput("Means",
              "Mean of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("Scales",
              "Scales of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddComment(R"DOC(
Data Normalization.

Can be used as a normalizer function for data
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
  }
};

template <typename T>
class MaskedDataNormKernel<phi::CPUContext, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unimplemented kernel for MaskedDataNorm, only support GPU "
        "now."));
  }
};

class MaskedDataNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "MaskedDataNormGrad");
    OP_INOUT_CHECK(
        ctx->HasInput("Mask"), "Input", "Mask", "MaskedDataNormGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   framework::GradVarName("Y"),
                   "MaskedDataNormGrad");
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSize"),
        true,
        platform::errors::NotFound(
            "Output(BatchSize) of MaskedDataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSum"),
        true,
        platform::errors::NotFound(
            "Output(BatchSum) of MaskedDataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSquareSum"),
        true,
        platform::errors::NotFound("Output(BatchSquareSum) of "
                                   "MaskedDataNormGradOp should not be null."));
    OP_INOUT_CHECK(
        ctx->HasInput("Means"), "Input", "Means", "MaskedDataNormGrad");
    OP_INOUT_CHECK(
        ctx->HasInput("Scales"), "Input", "Scales", "MaskedDataNormGrad");
    bool enable_scale_and_shift =
        ctx->Attrs().Get<bool>("enable_scale_and_shift");
    // check output
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSize")),
                   "Output",
                   framework::GradVarName("BatchSize"),
                   "MaskedDataNormGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSum")),
                   "Output",
                   framework::GradVarName("BatchSum"),
                   "MaskedDataNormGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSquareSum")),
                   "Output",
                   framework::GradVarName("BatchSquareSum"),
                   "MaskedDataNormGrad");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    ctx->SetOutputDim(framework::GradVarName("BatchSize"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSum"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSquareSum"), {C});
    if (enable_scale_and_shift) {
      const bool has_scale_grad =
          ctx->HasOutput(framework::GradVarName("scale_w"));
      const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("bias"));

      PADDLE_ENFORCE_EQ((has_scale_grad == has_bias_grad),
                        true,
                        platform::errors::InvalidArgument(
                            "Output(Scale@GRAD) and Output(Bias@GRAD)"
                            "must be null or not be null at same time. "
                            "But now, has Scale@Grad=[%d], has Bias@GRAD=[%d]",
                            has_scale_grad,
                            has_bias_grad));
      if (has_scale_grad) {
        ctx->SetOutputDim(framework::GradVarName("scale_w"), {C});
        ctx->SetOutputDim(framework::GradVarName("bias"), {C});
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Y@GRAD can not be found for computation"));
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Y@GRAD can not be found for computation"));
    }

    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

template <typename T>
class MaskedDataNormGradKernel<phi::CPUContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unimplemented kernel for MaskedDataNorm, only support GPU "
        "now."));
  }
};

template <typename T>
class MaskedDataNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("masked_data_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Mask", this->Input("Mask"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("scale_w", this->Input("scale_w"));
    op->SetInput("bias", this->Input("bias"));
    op->SetOutput("BatchSize", this->Input("BatchSize"));
    op->SetOutput("BatchSum", this->Input("BatchSum"));
    op->SetOutput("BatchSquareSum", this->Input("BatchSquareSum"));
    op->SetInput("Scales", this->Output("Scales"));
    op->SetInput("Means", this->Output("Means"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("BatchSize"),
                  this->InputGrad("BatchSize"));
    op->SetOutput(framework::GradVarName("BatchSum"),
                  this->InputGrad("BatchSum"));
    op->SetOutput(framework::GradVarName("BatchSquareSum"),
                  this->InputGrad("BatchSquareSum"));
    op->SetOutput(framework::GradVarName("scale_w"),
                  this->InputGrad("scale_w"));
    op->SetOutput(framework::GradVarName("bias"), this->InputGrad("bias"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(masked_data_norm,
                  ops::MaskedDataNormOp,
                  ops::MaskedDataNormOpMaker,
                  ops::MaskedDataNormGradMaker<paddle::framework::OpDesc>,
                  ops::MaskedDataNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(masked_data_norm_grad, ops::MaskedDataNormGradOp);

REGISTER_OP_CPU_KERNEL(masked_data_norm,
                       ops::MaskedDataNormKernel<phi::CPUContext, float>,
                       ops::MaskedDataNormKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(masked_data_norm_grad,
                       ops::MaskedDataNormGradKernel<phi::CPUContext, float>,
                       ops::MaskedDataNormGradKernel<phi::CPUContext, double>);
REGISTER_OP_VERSION(masked_data_norm)
    .AddCheckpoint(
        R"ROC(
              upgrad masked_data_norm op by adding scale_w to support scale and shift.)ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "scale_w",
            "scale_w is used to do scale duirng masked_data_norm like "
            "batchnorm "));
