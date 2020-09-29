/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/op_def/op_schema.h"

namespace ge {
DOMI_OP_SCHEMA(Data).Output("y");

DOMI_OP_SCHEMA(Const).Output("y");

DOMI_OP_SCHEMA(ConvolutionDepthwise)
    .Input("x")
    .Input("w")
    .Input("b", OpSchema::Optional)
    .Output("y")
    .Attr("group", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("num_output", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("pad_mode", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("pad", AttributeType::INTLIST, IntTuple{0, 0, 0, 0})
    .Attr("stride", AttributeType::INTLIST, IntTuple{1, 1})
    .Attr("dilation", AttributeType::INTLIST, IntTuple{1, 1})
    .Attr("kernel", AttributeType::INTLIST, IntTuple{0, 0})
    .Attr("before_pad", AttributeType::INTLIST, IntTuple{0, 0, 0, 0});

DOMI_OP_SCHEMA(Region)
    .Input("x")
    .Output("y")
    .Attr("casses", AttributeType::INT, static_cast<int64_t>(20))
    .Attr("coords", AttributeType::INT, static_cast<int64_t>(4))
    .Attr("boxes", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("background", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("softmax", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("softmax_tree", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("yolo_version", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(Gather)
    .Input("params")
    .Input("indices")
    .Input("axis", OpSchema::Optional)
    .Output("y")
    .Attr("params_type", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("indices_type", AttributeType::INT, static_cast<int64_t>(3))
    .Attr("validate_indices", AttributeType::BOOL, static_cast<bool>(true));

DOMI_OP_SCHEMA(ArgMax)
    .Input("input")
    .Output("output")
    .Attr("axis", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("keep_dims", AttributeType::BOOL, static_cast<bool>(true))
    .Attr("axis_type", AttributeType::INT, static_cast<int64_t>(3))
    .Attr("outmaxval", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("topk", AttributeType::UINT, static_cast<uint32_t>(1));

DOMI_OP_SCHEMA(Split)
    .Input("x")
    .Input("axis", OpSchema::Optional)
    .Output("y")
    .Attr("T", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("num_split", AttributeType::INT, static_cast<int64_t>(1));

DOMI_OP_SCHEMA(SplitV)
    .Input("x")
    .Input("axis", OpSchema::Optional)
    .Output("y")
    .Attr("T", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("Tlen", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("num_split", AttributeType::INT, static_cast<int64_t>(1));

DOMI_OP_SCHEMA(Fill).Input("x").Input("value").Output("y").Attr("T", AttributeType::INT, static_cast<int64_t>(1));
DOMI_OP_SCHEMA(Rsqrt).Input("x").Output("y");
DOMI_OP_SCHEMA(BiasAdd)
    .Input("x")
    .Input("bias")
    .Output("y")
    .Attr("format", AttributeType::INT, static_cast<int64_t>(1));
DOMI_OP_SCHEMA(Reverse)
    .Input("x")
    .Input("axis")
    .Output("y")
    .Attr("T", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("Tidx", AttributeType::INT, static_cast<int64_t>(1));
DOMI_OP_SCHEMA(Unpack)
    .Input("x")
    .Output("y")
    .Attr("T", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("axis", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("num", AttributeType::INT, static_cast<int64_t>(1));
DOMI_OP_SCHEMA(Yolo2Reorg)
    .Input("x")
    .Output("y")
    .Attr("reverse", AttributeType::BOOL, static_cast<bool>(1))
    .Attr("stride", AttributeType::INT, static_cast<int64_t>(1));

DOMI_OP_SCHEMA(ReduceSum)
    .Input("x")
    .Output("y")
    .Attr("Tidx", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("keep_dims", AttributeType::BOOL, static_cast<bool>(1));

DOMI_OP_SCHEMA(Concat)
    .Input("x")
    .Output("y")
    .Attr("Tidx", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("N", AttributeType::INT, static_cast<int64_t>(1));

DOMI_OP_SCHEMA(ResizeBilinear)
    .Input("x")
    .Input("sizes")
    .Output("y")
    .Attr("output_dim_mode", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("align_corners", AttributeType::BOOL, static_cast<bool>(1))
    .Attr("zoom_factor", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("shrink_factor", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("height", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("width", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("pad_begin", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("pad_end", AttributeType::INT, static_cast<int64_t>(1));

DOMI_OP_SCHEMA(LRN)
    .Input("x")
    .Output("y")
    .Attr("lrn_normregion", AttributeType::UINT, static_cast<uint32_t>(0))
    .Attr("lrn_k", AttributeType::FLOAT, static_cast<float>(1))
    .Attr("lrn_localsize", AttributeType::UINT, static_cast<uint32_t>(5))
    .Attr("lrn_alpha", AttributeType::FLOAT, static_cast<float>(1))
    .Attr("lrn_beta", AttributeType::FLOAT, static_cast<float>(0.75));

DOMI_OP_SCHEMA(Maximum).Input("x").Input("w").Output("y");

DOMI_OP_SCHEMA(Slice)
    .Input("x")
    .Output("y")
    .Attr("axis", AttributeType::INT, static_cast<int64_t>(2))
    .AttrRequired("offsets", AttributeType::INTLIST);

DOMI_OP_SCHEMA(Pad)
    .Input("x")
    .Input("paddings")
    .Input("constant_values", OpSchema::Optional)
    .Output("y")
    .Attr("T", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("t_paddings", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(PadV2)
    .Input("input")
    .Output("output")
    .Attr("constant_values", AttributeType::INT, static_cast<int64_t>(0))
    .AttrRequired("paddings", AttributeType::INTLIST);

DOMI_OP_SCHEMA(MirrorPad)
    .Input("input")
    .Output("output")
    .AttrRequired("paddings", AttributeType::INTLIST)
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(2));

DOMI_OP_SCHEMA(Upsample)
    .Input("input")
    .Input("scales")
    .Output("output")
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(Cast)
    .Input("x")
    .Output("y")
    .Attr("DstT", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("SrcT", AttributeType::INT, static_cast<int64_t>(1));
DOMI_OP_SCHEMA(LogicalNot).Input("x").Output("y");
DOMI_OP_SCHEMA(LogicalAnd).Input("x1").Input("x2").Output("y");
DOMI_OP_SCHEMA(LogicalOr).Input("x1").Input("x2").Output("y");
DOMI_OP_SCHEMA(Equal).Input("x1").Input("x2").Output("y").Attr("T", AttributeType::INT, static_cast<int64_t>(1));

DOMI_OP_SCHEMA(MatMul)
    .Input("a")
    .Input("b")
    .Output("product")
    .Attr("transposeX", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("transposeW", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(RNN)
    .Input("x")
    .Input("cont")
    .Input("xstatic", OpSchema::Optional)
    .Input("w")                            // filter
    .Input("b")                            // bias
    .Input("seqlen")                       // T
    .Input("hx")                           // Hx
    .Input("cx")                           // cx
    .Output("y")
    .Output("cyfw")
    .Output("hyfw")
    .Output("cybw")
    .Output("hybw")
    .Attr("hidden_size", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("num_layers", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("support_cont", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("support_xstatic", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("input_mode", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("direction_mode", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("input_data_layout", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("output_data_layout", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(FrameworkOp).Attr("framework_type", AttributeType::INT, static_cast<int64_t>(3));
DOMI_OP_SCHEMA(Multinomial)
    .Input("logits")
    .Output("output")
    .Attr("num_samples", AttributeType::INT, static_cast<int64_t>(0))
    .AttrRequired("seed", AttributeType::INT)
    .AttrRequired("seed2", AttributeType::INT);
DOMI_OP_SCHEMA(ReverseSequence)
    .Input("input")
    .Input("seq_lengths")
    .Output("output")
    .AttrRequired("seq_dim", AttributeType::INT)
    .AttrRequired("batch_dim", AttributeType::INT);

DOMI_OP_SCHEMA(Interp)
    .Input("x")
    .Output("y")
    .Attr("output_dim_mode", AttributeType::INT, static_cast<int64_t>(2))
    .Attr("zoom_factor", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("shrink_factor", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("height", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("width", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("pad_begin", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("pad_end", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(ShuffleChannel).Input("x").Output("y").Attr("group", AttributeType::UINT, static_cast<uint32_t>(1));

DOMI_OP_SCHEMA(Conv2DBackpropFilter)
    .Input("x")
    .Input("w")
    .Input("b", OpSchema::Optional)
    .Output("y")
    .Attr("padding", AttributeType::INT, static_cast<int64_t>(1))
    .Attr("pads", AttributeType::UINTLIST, UintTuple{0, 0, 0, 0})
    .Attr("strides", AttributeType::UINTLIST, UintTuple{1, 1})
    .Attr("dilations", AttributeType::UINTLIST, UintTuple{1, 1});

DOMI_OP_SCHEMA(Conv2DBackpropInput)
    .Input("input_sizes")
    .Input("filter")
    .Input("out_backprop")
    .Output("output")
    .Attr("data_format", AttributeType::STRING, static_cast<std::string>("NHWC"))
    .Attr("group", AttributeType::UINT, static_cast<uint32_t>(1))
    .Attr("padding", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("dilations", AttributeType::UINTLIST, UintTuple{1, 1})
    .Attr("strides", AttributeType::UINTLIST, UintTuple{1, 1})
    .Attr("pad", AttributeType::UINTLIST, UintTuple{0, 0, 0, 0});
DOMI_OP_SCHEMA(BiasAddGrad).Input("dy").Output("db").Attr("format", AttributeType::INT, static_cast<int64_t>(1));
DOMI_OP_SCHEMA(ReluGrad).Input("dy").Input("x").Output("dx");

DOMI_OP_SCHEMA(MeanGrad).Input("dy").Output("dx");

DOMI_OP_SCHEMA(NonMaxSuppression)
    .Input("boxes")
    .Input("scores")
    .Output("selected_indices")
    .Attr("max_output_size", AttributeType::INT, static_cast<int64_t>(-1))
    .Attr("iou_threshold", AttributeType::FLOAT, static_cast<float>(0.5))
    .Attr("score_threshold", AttributeType::FLOAT, static_cast<float>(-1));

DOMI_OP_SCHEMA(CropAndResize)
    .Input("image")
    .Input("boxes")
    .Input("box_ind")
    .Output("crops")
    .Attr("method", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("extrapolation_value", AttributeType::FLOAT, static_cast<float>(0))
    .Attr("crop_size_h", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("crop_size_w", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(TopKV2)
    .Input("input")
    .Input("k")
    .Output("value")
    .Output("indices")
    .AttrRequired("sorted", AttributeType::BOOL);

DOMI_OP_SCHEMA(InvertPermutation).Input("x").Output("y");

DOMI_OP_SCHEMA(GatherV2)
    .Input("params")
    .Input("indices")
    .Input("axis", OpSchema::Optional)
    .Output("y")
    .Attr("Tparams", AttributeType::INT, static_cast<int64_t>(0))   // default: DT_FLOAT
    .Attr("Tindices", AttributeType::INT, static_cast<int64_t>(3))  // default: DT_INT32
    .Attr("Taxis", AttributeType::INT, static_cast<int64_t>(3));    // default: DT_INT32

DOMI_OP_SCHEMA(HighWay)
    .Input("x")
    .Input("tw")  // filter
    .Input("tb")  // bias
    .Input("uw")  // filter
    .Input("ub")  // bias
    .Output("y");

DOMI_OP_SCHEMA(Reciprocal).Input("x").Output("y");

DOMI_OP_SCHEMA(Asinh).Input("input").Output("output");

DOMI_OP_SCHEMA(Acosh).Input("input").Output("output");

DOMI_OP_SCHEMA(Minimum).Input("x").Input("y").Output("output");

DOMI_OP_SCHEMA(Clip).Input("input").Input("min").Input("max").Output("output");

DOMI_OP_SCHEMA(FusedBatchNorm)
    .Input("x")
    .Input("scale")
    .Input("offset")
    .Input("mean")
    .Input("variance")
    .Output("y")
    .Output("batch_mean")
    .Output("batch_variance")
    .Output("reserve_space_1")
    .Output("reserve_space_2")
    .Attr("data_format", AttributeType::STRING, static_cast<std::string>("NHWC"))
    .Attr("epsilon", AttributeType::FLOAT, static_cast<float>(0.0001))
    .Attr("is_training", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(FusedBatchNormGrad)
    .Input("dy")
    .Input("x")
    .Input("bnscale")
    .Input("save_mean")
    .Input("save_variance")
    .Output("dx")
    .Output("result_bn_scale_diff")
    .Output("result_bn_bias_diff")
    .Attr("data_format", AttributeType::STRING, static_cast<std::string>("NHWC"))
    .Attr("epsilon", AttributeType::FLOAT, static_cast<float>(0.0))
    .Attr("is_training", AttributeType::BOOL, static_cast<bool>(true));

DOMI_OP_SCHEMA(MaxPoolWithArgmax)
    .Input("x")
    .Output("y")
    .Output("argmax")
    .AttrRequired("window", AttributeType::INTLIST)
    .AttrRequired("stride", AttributeType::INTLIST)
    .AttrRequired("pad_mode", AttributeType::INT)
    .AttrRequired("ceil_mode", AttributeType::BOOL)
    .AttrRequired("data_mode", AttributeType::INT);

DOMI_OP_SCHEMA(MaxPoolGradWithArgmax)
    .Input("input")
    .Input("grad")
    .Output("output")
    .AttrRequired("window", AttributeType::INTLIST)
    .AttrRequired("stride", AttributeType::INTLIST)
    .AttrRequired("pad_mode", AttributeType::INT)
    .AttrRequired("ceil_mode", AttributeType::BOOL)
    .AttrRequired("data_mode", AttributeType::INT);

DOMI_OP_SCHEMA(HcomBroadcast)
    .AttrRequired("root_rank", AttributeType::INT)
    .AttrRequired("group", AttributeType::STRING);

DOMI_OP_SCHEMA(HcomAllReduce)
    .Input("x")
    .Output("y")
    .AttrRequired("reduction", AttributeType::STRING)
    .AttrRequired("group", AttributeType::STRING);

DOMI_OP_SCHEMA(HcomAllGather)
    .Input("x")
    .Output("y")
    .AttrRequired("rank_size", AttributeType::INT)
    .AttrRequired("group", AttributeType::STRING);

DOMI_OP_SCHEMA(SparseSoftmaxCrossEntropyWithLogits)
    .Input("features")
    .Input("labels")
    .Output("loss")
    .Output("backprop")
    .AttrRequired("T", AttributeType::INT)
    .Attr("Tlabels", AttributeType::INT, static_cast<int64_t>(9));

DOMI_OP_SCHEMA(Snapshot).Input("input").Output("output").AttrRequired("T", AttributeType::INT);

DOMI_OP_SCHEMA(ReduceProd)
    .Input("bottom")
    .Output("top")
    .AttrRequired("axes", AttributeType::INTLIST)
    .Attr("keep_dims", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(ReduceAll)
    .Input("x")
    .Output("y")
    .AttrRequired("axes", AttributeType::INTLIST)
    .Attr("keep_dims", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(ReduceMax)
    .Input("x")
    .Output("y")
    .AttrRequired("axis", AttributeType::INTLIST)
    .Attr("keep_dims", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(AddN).Input("x").Output("y");

DOMI_OP_SCHEMA(ShapeN)
    .Input("x")
    .Output("y")
    .AttrRequired("N", AttributeType::INT)
    .AttrRequired("in_type", AttributeType::INT)
    .AttrRequired("dtype", AttributeType::INT);

DOMI_OP_SCHEMA(ReduceMin)
    .Input("x")
    .Output("y")
    .AttrRequired("axis", AttributeType::INTLIST)
    .Attr("keep_dims", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(Sqrt).Input("x").Output("y");

DOMI_OP_SCHEMA(L2Loss).Input("x").Output("y");

DOMI_OP_SCHEMA(Multiply).Input("x").Input("y").Output("z");

DOMI_OP_SCHEMA(Add).Input("x").Output("y");

DOMI_OP_SCHEMA(Constant).Output("y");

DOMI_OP_SCHEMA(ApplyMomentum)
    .Input("variable")
    .Input("accumulation")
    .Input("learningRate")
    .Input("gradient")
    .Input("momuntum")
    .Input("fp16variable")
    .Attr("algo", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(AvgPoolGrad)
    .Input("shape")
    .Input("grad")
    .Output("output")
    .Attr("padding", AttributeType::INT, static_cast<int64_t>(0))
    .Attr("data_format", AttributeType::STRING, static_cast<std::string>("NHWC"))
    .Attr("strides", AttributeType::UINTLIST, UintTuple{0, 0, 0, 0})
    .Attr("ksize", AttributeType::UINTLIST, UintTuple{0, 0, 0, 0});

DOMI_OP_SCHEMA(Lars)
    .Input("w")
    .Input("g")
    .Input("weight_decay")
    .Output("y")
    .Attr("hyperpara", AttributeType::FLOAT, static_cast<float>(0.001))
    .Attr("epsilon", AttributeType::FLOAT, static_cast<float>(0.00001));

DOMI_OP_SCHEMA(AssignSub)
    .Input("variable")
    .Input("input")
    .Input("output")
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(AssignAdd)
    .Input("variable")
    .Input("input")
    .Output("output")
    .Attr("mode", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(SpaceToBatchND).Input("input").Input("block_shape").Input("paddings").Output("output");

DOMI_OP_SCHEMA(Variable)
    .Output("variable")
    .Attr("container", AttributeType::STRING, static_cast<std::string>(""))
    .Attr("shared_name", AttributeType::STRING, static_cast<std::string>(""))
    .AttrRequired("dtype", AttributeType::INT);

DOMI_OP_SCHEMA(Assign).Input("variable").Input("value").Output("variable");

DOMI_OP_SCHEMA(VarIsInitializedOp).Input("variable").Output("value");

DOMI_OP_SCHEMA(NoOp).Attr("algo", AttributeType::INT, static_cast<int64_t>(0));

DOMI_OP_SCHEMA(LogTimeStamp)
    .Attr("logid", AttributeType::STRING, static_cast<std::string>(""))
    .Attr("notify", AttributeType::BOOL, static_cast<bool>(false));

DOMI_OP_SCHEMA(ResizeNearestNeighbor)
    .Input("images")
    .Output("resized_images")
    .Attr("align_corners", AttributeType::BOOL, static_cast<bool>(false))
    .AttrRequired("height", AttributeType::INT)
    .AttrRequired("width", AttributeType::INT);

DOMI_OP_SCHEMA(BatchToSpaceND).Input("input").Input("block_shape").Input("crops").Output("output");

DOMI_OP_SCHEMA(Assert).Input("x").Input("w").Output("y");

DOMI_OP_SCHEMA(Pow).Input("x").Input("y").Output("z");

DOMI_OP_SCHEMA(GreaterEqual).Input("x1").Input("x2").Output("y");

DOMI_OP_SCHEMA(SpaceToDepth)
    .Input("input")
    .Output("output")
    .Attr("block_size", AttributeType::INT, static_cast<int64_t>(0))
    .AttrRequired("T", AttributeType::INT)
    .Attr("data_format", AttributeType::STRING, static_cast<std::string>("NHWC"));

DOMI_OP_SCHEMA(DepthToSpace)
    .Input("input")
    .Output("output")
    .Attr("block_size", AttributeType::INT, static_cast<int64_t>(0))
    .AttrRequired("T", AttributeType::INT)
    .Attr("data_format", AttributeType::STRING, static_cast<std::string>("NHWC"));

DOMI_OP_SCHEMA(Rint).Input("input").Output("output").AttrRequired("T", AttributeType::INT);

DOMI_OP_SCHEMA(ExtractImagePatches)
    .Input("images")
    .Output("y")
    .AttrRequired("ksizes", AttributeType::INTLIST)
    .AttrRequired("strides", AttributeType::INTLIST)
    .AttrRequired("rates", AttributeType::INTLIST)
    .AttrRequired("padding", AttributeType::STRING);

DOMI_OP_SCHEMA(Atan).Input("x").Output("output");

DOMI_OP_SCHEMA(Atanh).Input("x").Output("output");

DOMI_OP_SCHEMA(Acos).Input("x").Output("y");

DOMI_OP_SCHEMA(Asin).Input("x").Output("y");

DOMI_OP_SCHEMA(Log)
    .Input("x")
    .Output("output")
    .AttrRequired("scale", AttributeType::INT)
    .AttrRequired("shift", AttributeType::INT)
    .AttrRequired("base", AttributeType::INT);

DOMI_OP_SCHEMA(Neg).Input("input").Output("output");

DOMI_OP_SCHEMA(Tan).Input("x").Output("output");

DOMI_OP_SCHEMA(Round).Input("x").Output("output");

DOMI_OP_SCHEMA(Exp)
    .Input("x")
    .Output("y")
    .Attr("scale", AttributeType::FLOAT, static_cast<float>(1))
    .Attr("shift", AttributeType::FLOAT, static_cast<float>(0))
    .Attr("base", AttributeType::FLOAT, static_cast<float>(-1));

DOMI_OP_SCHEMA(Less).Input("x").Input("y").Output("output");

DOMI_OP_SCHEMA(LessEqual).Input("x").Input("y").Output("output");

DOMI_OP_SCHEMA(OneHot).Input("indices").Input("depth").Input("on_value").Input("off_value").Output("output");

DOMI_OP_SCHEMA(ZerosLike).Input("x").Output("y");

DOMI_OP_SCHEMA(Where).Input("x").Output("y");

DOMI_OP_SCHEMA(RefSwitch).Input("x").Output("y");

DOMI_OP_SCHEMA(FakeQuantWithMinMaxVars)
    .Input("x")
    .Input("min")
    .Input("max")
    .Output("y")
    .Attr("narrow_range", AttributeType::BOOL, static_cast<bool>(false))
    .Attr("num_bits", AttributeType::INT, static_cast<int64_t>(8));

DOMI_OP_SCHEMA(Sinh).Input("x").Output("y");

DOMI_OP_SCHEMA(Cosh).Input("x").Output("y");

DOMI_OP_SCHEMA(Floor).Input("x").Output("output");

DOMI_OP_SCHEMA(RandomUniform).Input("input").Output("output");

DOMI_OP_SCHEMA(BatchMatMul).Input("x").Input("y").Output("output");

DOMI_OP_SCHEMA(FloorMod).Input("x").Input("y").Output("output");

DOMI_OP_SCHEMA(SquaredDifference).Input("x").Input("y").Output("output");

DOMI_OP_SCHEMA(LayerNorm).Input("x").Output("output").AttrRequired("Epsilon", AttributeType::FLOAT);

DOMI_OP_SCHEMA(SSDPostProcessor)
    .Input("trueImgShape")
    .Input("boxEncoding")
    .Input("anchors")
    .Input("clsPred")
    .Output("detectBoxes")
    .Output("detectScores")
    .Output("detectNum")
    .Output("detectClasses")
    .AttrRequired("numClasses", AttributeType::INT)
    .AttrRequired("scoreThreshold", AttributeType::FLOAT)
    .AttrRequired("iouThreshold", AttributeType::FLOAT)
    .AttrRequired("maxDetectionsPerClass", AttributeType::INT)
    .AttrRequired("maxTotalDetections", AttributeType::INT)
    .AttrRequired("boxTypeNum", AttributeType::UINT)
    .AttrRequired("scaleFactors_0", AttributeType::UINT)
    .AttrRequired("scaleFactors_1", AttributeType::UINT)
    .AttrRequired("scaleFactors_2", AttributeType::UINT)
    .AttrRequired("scaleFactors_3", AttributeType::UINT)
    .AttrRequired("imgH", AttributeType::INT)
    .AttrRequired("imgW", AttributeType::INT)
    .AttrRequired("useStaticShape", AttributeType::BOOL)
    .AttrRequired("convertScoresMode", AttributeType::INT);

DOMI_OP_SCHEMA(RetinaPostProcessor)
    .Input("anchors")
    .Input("regression")
    .Input("classification")
    .Output("detectBoxes")
    .Output("detectScores")
    .Output("detectLabels")
    .Output("detectNum")
    .AttrRequired("numClasses", AttributeType::INT)
    .AttrRequired("maxDetections", AttributeType::INT)
    .AttrRequired("nmsThreshold", AttributeType::FLOAT)
    .AttrRequired("scoreThreshold", AttributeType::FLOAT)
    .AttrRequired("imgH", AttributeType::INT)
    .AttrRequired("imgW", AttributeType::INT)
    .AttrRequired("boxTypeNum", AttributeType::UINT)
    .AttrRequired("means", AttributeType::FLOATLIST)
    .AttrRequired("stds", AttributeType::FLOATLIST);

DOMI_OP_SCHEMA(ROIInterPooling)
    .Input("input")
    .Input("input_1")
    .Output("maxPool")
    .AttrRequired("hStride", AttributeType::INT)
    .AttrRequired("wStride", AttributeType::INT)
    .AttrRequired("hKernel", AttributeType::INT)
    .AttrRequired("wKernel", AttributeType::INT)
    .AttrRequired("hResize", AttributeType::INT)
    .AttrRequired("wResize", AttributeType::INT)
    .AttrRequired("hFeatureMap", AttributeType::INT)
    .AttrRequired("wFeatureMap", AttributeType::INT);

DOMI_OP_SCHEMA(FirstStageProcessor)
    .Input("anchors")
    .Input("boxEncoding")
    .Input("clsPred")
    .Input("trueImgShape")
    .Output("detectBoxes")
    .Output("detectScores")
    .Output("detectLables")
    .Output("detectNum")
    .AttrRequired("scaleFactorsNum", AttributeType::INT)
    .AttrRequired("iouThreshold", AttributeType::FLOAT)
    .AttrRequired("scoreThreshold", AttributeType::FLOAT)
    .AttrRequired("maxSizePerClass", AttributeType::INT)
    .AttrRequired("maxTotalSize", AttributeType::INT)
    .AttrRequired("imgH", AttributeType::INT)
    .AttrRequired("imgW", AttributeType::INT)
    .AttrRequired("boxTypeNum", AttributeType::UINT)
    .AttrRequired("scaleFactors_0", AttributeType::UINT)
    .AttrRequired("scaleFactors_1", AttributeType::UINT)
    .AttrRequired("scaleFactors_2", AttributeType::UINT)
    .AttrRequired("scaleFactors_3", AttributeType::UINT);

DOMI_OP_SCHEMA(SecondStageProcessor)
    .Input("anchors")
    .Input("boxEncoding")
    .Input("clsPred")
    .Input("validBoxNum")
    .Input("trueImgShape")
    .Output("detectBoxes")
    .Output("detectScores")
    .Output("detectLables")
    .Output("detectNum")
    .AttrRequired("scaleFactorsNum", AttributeType::INT)
    .AttrRequired("iouThreshold", AttributeType::FLOAT)
    .AttrRequired("scoreThreshold", AttributeType::FLOAT)
    .AttrRequired("maxSizePerClass", AttributeType::INT)
    .AttrRequired("maxTotalSize", AttributeType::INT)
    .AttrRequired("numClasses", AttributeType::INT)
    .AttrRequired("scaleFactors_0", AttributeType::UINT)
    .AttrRequired("scaleFactors_1", AttributeType::UINT)
    .AttrRequired("scaleFactors_2", AttributeType::UINT)
    .AttrRequired("scaleFactors_3", AttributeType::UINT);

DOMI_OP_SCHEMA(StreamSwitch)
    .Input("loopIndex")
    .Input("itersPerLoop")
    .AttrRequired("switch_condition", AttributeType::UINT)
    .AttrRequired("true_branch_stream", AttributeType::INT);

DOMI_OP_SCHEMA(StreamActive).AttrRequired("active_stream_list", AttributeType::INTLIST);

DOMI_OP_SCHEMA(MemcpyAsync).Input("in").Output("out");

DOMI_OP_SCHEMA(CleanAddr)
    .AttrRequired("automic_add_addr_start", AttributeType::INT)
    .AttrRequired("automic_add_mem_size", AttributeType::INT);
}  // namespace ge
