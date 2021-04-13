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

#include "onnx_data_parser.h"
#include <unordered_map>
#include "common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "parser/onnx/onnx_util.h"

using domi::ONNX;
using namespace ge::parser;

namespace ge {
Status OnnxDataParser::ParseParams(const Message *op_src, ge::Operator &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node_src = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  GE_CHECK_NOTNULL(node_src);
  GELOGD("Onnx op node name = %s, op type= %s, parse params", node_src->name().c_str(), node_src->op_type().c_str());
  if (ParseInputFromModel(op_src, op_def) != SUCCESS) {
    GELOGE(FAILED, "[Parse][Shape] of data op %s from model failed", op_def.GetName().c_str());
    return FAILED;
  }
  // Subgraph data operator don't need parse input shape
  // the shape mappings from parent node input
  if (IsSubgraphDataOp()) {
    return SUCCESS;
  }

  if (ParseInputFromUser(op_def) != SUCCESS) {
    GELOGE(FAILED, "[Parse][Shape] of data op %s from user failed", op_def.GetName().c_str());
    return FAILED;
  }

  ge::TensorDesc tensor_desc;
  tensor_desc.SetShape(ge::Shape(user_input_dims_v_));
  int64_t type = 1;
  (void)op_def.GetAttr(ge::DATA_ATTR_NAME_DATA_TYPE, type);
  tensor_desc.SetDataType(static_cast<DataType>(type));

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_def);
  op_def.UpdateInputDesc(op_desc->GetInputNameByIndex(0), tensor_desc);
  op_def.UpdateOutputDesc(op_desc->GetOutputNameByIndex(0), tensor_desc);

  return SUCCESS;
}

int64_t OnnxDataParser::ParseInputTensor(const ge::onnx::AttributeProto &attribute) {
  const ::ge::onnx::TensorProto it_tensor = attribute.t();
  int64_t data_type = it_tensor.data_type();
  GELOGI("Attr name: %s, data type: %ld ", attribute.name().c_str(), data_type);
  for (auto dim : it_tensor.dims()) {
    model_input_dims_v_.push_back(dim);
  }
  return data_type;
}

Status OnnxDataParser::ParseInputFromModel(const Message *op_src, ge::Operator &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  GE_CHECK_NOTNULL(node);

  // Get attr t:'input_tensor' form NodeProto
  int64_t data_type = 1;
  int64_t index = 0;
  is_subgraph_data_op_ = false;
  for (auto it : node->attribute()) {
    if (it.name() == ge::kAttrNameInput) {
      data_type = ParseInputTensor(it);
    } else if (it.name() == ge::kAttrNameIndex) {
      index = it.i();
      GELOGI("The node has attribute with index: %ld", index);
    } else if (it.name() == ge::kAttrNameIsSubgraphOp) {
      is_subgraph_data_op_ = true;
    }
  }

  op_def.SetAttr(ge::ATTR_NAME_INDEX, index);
  if (IsSubgraphDataOp()) {
    return SUCCESS;
  }

  // Trans onnx type to ge type
  DataType type = OnnxUtil::ConvertOnnxDataType(data_type);
  if (type == ge::DataType::DT_UNDEFINED) {
    REPORT_INNER_ERROR("E19999", "tensor_proto date type %ld is undefined.", data_type);
    GELOGE(domi::PARAM_INVALID, "[Check][Param]tensor_proto date type %ld is undefined.", data_type);
    return FAILED;
  }
  op_def.SetAttr(ge::DATA_ATTR_NAME_DATA_TYPE, static_cast<int64_t>(type));

  return SUCCESS;
}

Status OnnxDataParser::ParseInputFromUser(const ge::Operator &op_def) {
  std::map<std::string, std::vector<int64_t>> input_dims = GetParserContext().input_dims;
  // User not designate the input_shape
  std::string name = op_def.GetName();
  if (input_dims.count(name) == 0) {
    GELOGI("input shape of node %s is not designated ,need parse from model", name.c_str());
    for (uint32_t i = 0; i < model_input_dims_v_.size(); i++) {
      user_input_dims_v_.push_back(model_input_dims_v_[i]);
    }
    return SUCCESS;
  }

  /// User designate the input_shape by passing '--input_shape=xxx:x,x,x,x'
  /// Two cases below both OK:
  /// 1. the input_shape not defined in the model(dimension is 0).
  /// 2. the input_shape defined in the model(dimension greater than 0), and the dimension matches with user
  /// designate_dim.
  std::vector<int64_t> designated_dims = input_dims.at(name);
  size_t input_dim_size = designated_dims.size();
  if (!(model_input_dims_v_.empty() || input_dim_size == model_input_dims_v_.size())) {
    GELOGD("user designated input_dim_num %zu does match input_dim_num %zu defined by model",
           input_dim_size, model_input_dims_v_.size());
    return domi::PARAM_INVALID;
  }

  // replace with the user designated_dims
  user_input_dims_v_.swap(designated_dims);

  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(ONNX, DATA, OnnxDataParser);
}  // namespace ge
