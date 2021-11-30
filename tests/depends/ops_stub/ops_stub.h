/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MAIN_OPS_STUB_H
#define MAIN_OPS_STUB_H

#include "external/graph/operator_reg.h"
#include "register/op_registry.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
// for ir
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

REG_OP(Variable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Variable)

REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Tensor, Tensor())
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Const)

REG_OP(Assign)
    .INPUT(resource, TensorType::ALL())
    .INPUT(value, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Assign) REG_OP(Sqrt)
    .INPUT(x, TensorType{(DT_FLOAT.DT_FLOAT16)})
    .OUTPUT(y, TensorType{(DT_FLOAT, DT_FLOAT16)})
    .ATTR(T, Int, 1)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Sqrt)

REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)

REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Identity)

REG_OP(Abs)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Abs)

REG_OP(PartitionedCall)
    .DYNAMIC_INPUT(args, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(f)
    .ATTR(config, String, "")
    .ATTR(config_proto, String, "")
    .ATTR(executor_type, String, "")
    .OP_END_FACTORY_REG(PartitionedCall)

// for plugin
static Status ParseParamsStub(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

static Status ParseParamByOpFuncStub(const ge::Operator &op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

static Status ParseSubgraphPostFnIfStub(const std::string& subgraph_name, const ge::Graph& graph) {
  domi::AutoMappingSubgraphIOIndexFunc auto_mapping_subgraph_index_func =
      domi::FrameworkRegistry::Instance().GetAutoMappingSubgraphIOIndexFunc(domi::ONNX);
  if (auto_mapping_subgraph_index_func == nullptr) {
    std::cout<<"auto mapping if subgraph func is nullptr!"<<std::endl;
    return FAILED;
  }
  return auto_mapping_subgraph_index_func(graph,
                                          [&](int data_index, int &parent_index) -> Status {
                                            parent_index = data_index + 1;
                                            return SUCCESS;
                                          },
                                          [&](int output_index, int &parent_index) -> Status {
                                            parent_index = output_index;
                                            return SUCCESS;
                                          });
}

static Status ParseParamsClipV9Stub(const Message* op_src, ge::Operator& op_dest) {
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  // 1.add dynamic input and out
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("output", 1);

  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "ai.onnx::9::Clip");
  return SUCCESS;
}

static Status ParseOpToGraphClipV9Stub(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto abs0 = op::Abs("abs0").set_input_x(data0);

  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(abs0, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}


//  caffe plugin
REGISTER_CUSTOM_OP("Data")
  .FrameworkType(domi::CAFFE)
  .OriginOpType("Input")
  .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("Abs")
    .FrameworkType(domi::CAFFE)
    .OriginOpType("AbsVal")
    .ParseParamsFn(ParseParamsStub);

// onnx plugin
REGISTER_CUSTOM_OP("Conv2D")
  .FrameworkType(domi::ONNX)
  .OriginOpType("ai.onnx::11::Conv")
  .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("If")
  .FrameworkType(domi::ONNX)
  .OriginOpType({"ai.onnx::9::If",
                 "ai.onnx::10::If",
                 "ai.onnx::11::If",
                 "ai.onnx::12::If",
                 "ai.onnx::13::If"})
  .ParseParamsFn(ParseParamsStub)
  .ParseParamsByOperatorFn(ParseParamByOpFuncStub)
  .ParseSubgraphPostFn(ParseSubgraphPostFnIfStub);

REGISTER_CUSTOM_OP("Add")
  .FrameworkType(domi::ONNX)
      .OriginOpType("ai.onnx::11::Add")
      .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("Identity")
  .FrameworkType(domi::ONNX)
      .OriginOpType("ai.onnx::11::Identity")
      .ParseParamsFn(ParseParamsStub);

// tf plugin
REGISTER_CUSTOM_OP("Add")
  .FrameworkType(domi::TENSORFLOW)
      .OriginOpType("Add")
      .ParseParamsFn(ParseParamsStub);


REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(domi::ONNX)
    .OriginOpType({"ai.onnx::9::Clip"})
    .ParseParamsFn(ParseParamsClipV9Stub)
    .ParseOpToGraphFn(ParseOpToGraphClipV9Stub);
}  // namespace ge
#endif  // MAIN_OPS_STUB_H
