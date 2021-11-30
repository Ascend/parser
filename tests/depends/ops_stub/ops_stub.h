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
REG_OP(TensorArray)
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(handle, TensorType({DT_RESOURCE}))
    .OUTPUT(flow, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(element_shape, ListInt, ge::UNKNOWN_RANK)
    .ATTR(dynamic_size, Bool, false)
    .ATTR(clear_after_read, Bool, true)
    .ATTR(identical_element_shapes, Bool, false)
    .ATTR(tensor_array_name, String, "")
    .OP_END_FACTORY_REG(TensorArray)

REG_OP(TensorArrayWrite)
    .INPUT(handle, TensorType({DT_RESOURCE}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8,
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL,
        DT_STRING, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(flow_in, TensorType({DT_FLOAT}))
    .OUTPUT(flow_out, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(TensorArrayWrite)

REG_OP(AvgPool3DGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3DGrad)

REG_OP(Merge)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Merge)

REG_OP(NoOp)
    .OP_END_FACTORY_REG(NoOp)

REG_OP(VarIsInitializedOp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(VarIsInitializedOp)

REG_OP(AssignVariableOp)
    .INPUT(resource, TensorType({DT_RESOURCE}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
        DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(AssignVariableOp)

REG_OP(ReadVariableOp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ReadVariableOp)

REG_OP(Reshape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

REG_OP(VarHandleOp)
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(shape, ListInt, ge::UNKNOWN_SHAPE)
    .OUTPUT(y, TensorType({DT_RESOURCE}))
    .OP_END_FACTORY_REG(VarHandleOp)

REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(Fill)

REG_OP(ShapeN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ShapeN)

REG_OP(Switch)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(output_true, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Switch)

REG_OP(RefSwitch)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(output_true, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(RefSwitch)

REG_OP(Enter)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE,
        DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
        DT_UINT64, DT_BOOL}))
    .REQUIRED_ATTR(frame_name, String)
    .REQUIRED_ATTR(is_constant, Bool)
    .OP_END_FACTORY_REG(Enter)

REG_OP(VariableV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(VariableV2)

REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)

REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Mul)

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
REGISTER_CUSTOM_OP("TensorArray")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("TensorArrayV3")
    .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("TensorArrayWrite")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("TensorArrayWriteV3")
    .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("BlockLSTM")
    .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("Merge")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("HistogramSummary")
    .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("NoOp")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("NoOp")
    .ParseParamsFn(ParseParamsStub);

REGISTER_CUSTOM_OP("Fill")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("Fill")
    .ParseParamsFn(ParseParamsStub);
}  // namespace ge


#endif  // MAIN_OPS_STUB_H
