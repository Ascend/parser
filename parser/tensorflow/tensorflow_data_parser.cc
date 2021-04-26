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

#include "parser/tensorflow/tensorflow_data_parser.h"
#include <unordered_map>
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"

using domi::tensorflow::AttrValue;
using domi::tensorflow::NodeDef;
using domi::TENSORFLOW;
using ge::parser::DATA;

namespace ge {
namespace {
const int64_t kValidShapeMinValue = -2;
}  // namespace
Status TensorFlowDataParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const NodeDef *node_src = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node_src->name().c_str(), node_src->op().c_str());
  GE_CHECK_NOTNULL(op_def);
  GE_RETURN_WITH_LOG_IF_ERROR(ParseInputFromModel(op_src, op_def), "parse shape of data op %s from model failed",
                              op_def->GetName().c_str());

  GE_RETURN_WITH_LOG_IF_ERROR(ParseInputFromUser(op_src, op_def), "parse shape of data op %s from user failed",
                              op_def->GetName().c_str());

  GE_RETURN_WITH_LOG_IF_ERROR(CheckInputShape(op_def->GetName()),
                              "input node %s :check user designated input shape not match input shape defined in model",
                              op_def->GetName().c_str());

  // Parse data dimension values and add them to op_def
  GE_RETURN_WITH_LOG_IF_ERROR(ParseShape(user_input_dims_v, op_def), "TensorFlowDataParser::ParseShape failed");

  return SUCCESS;
}

Status TensorFlowDataParser::ParseInputFromModel(const Message *op_src, ge::OpDescPtr &op_def) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op_def);

  const NodeDef *node = DOMI_DYNAMIC_CAST<const domi::tensorflow::NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);

  domi::tensorflow::AttrValue attr_value;
  if (TensorFlowUtil::FindAttrValue(node, TENSORFLOW_ATTR_DTYPE, attr_value)) {
    // Check dtype attribute must be type
    GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TYPE),
                                "check Attr %s failed", TENSORFLOW_ATTR_DTYPE.c_str());

    domi::tensorflow::DataType tf_type = attr_value.type();
    ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tf_type);
    CHECK_FALSE_EXEC(type != ge::DataType::DT_UNDEFINED,
                     REPORT_CALL_ERROR("E19999", "Data type %s of node %s is not supported",
                                       DataType_Name(tf_type).c_str(),
                                       node->name().c_str());
                     GELOGE(domi::PARAM_INVALID,
                            "Data type %s of node %s is not supported.",
                            DataType_Name(tf_type).c_str(),
                            node->name().c_str());
                     return domi::PARAM_INVALID);

    GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::SetInt(op_def, DATA_ATTR_NAME_DATA_TYPE, static_cast<int64_t>(type)), FAILED,
                           "SetAttr:%s to node:%s(%s) failed", DATA_ATTR_NAME_DATA_TYPE.c_str(),
                           op_def->GetName().c_str(), op_def->GetType().c_str());
  }

  if (!TensorFlowUtil::FindAttrValue(node, TENSORFLOW_ATTR_SHAPE, attr_value)) {
    // in some case, data could be without shape and is updated by `input_shape` option in following process
    GELOGW("input data node %s do not find shape.", node->name().c_str());
    return SUCCESS;
  }

  // Check shape attribute must be shape
  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_SHAPE),
                              "check Attr %s failed", TENSORFLOW_ATTR_SHAPE.c_str());

  const TensorShapeProto &data_shape = attr_value.shape();
  for (auto i = 0; i < data_shape.dim_size(); i++) {
    model_input_dims_v.push_back(data_shape.dim(i).size());
  }

  return SUCCESS;
}

Status TensorFlowDataParser::ParseInputFromUser(const Message *op_src, const ge::OpDescPtr &op_def) {
  GE_CHECK_NOTNULL(op_def);
  (void)op_src;
  const ge::ParserContext &ctx = GetParserContext();
  std::map<std::string, std::vector<int64_t>> input_dims = ctx.input_dims;
  // User not designate the input_shape
  std::string name = op_def->GetName();
  if (input_dims.count(name) == 0) {
    GELOGI("input shape of node %s is not designated ,need parse from model", name.c_str());
    for (uint32_t i = 0; i < model_input_dims_v.size(); i++) {
      user_input_dims_v.push_back(model_input_dims_v[i]);
    }

    return SUCCESS;
  }

  /* User designate the input_shape by passing '--input_shape=xxx:x,x,x,x' */
  // Two cases below both OK:
  // 1. the input_shape not defined in the model(dimension is 0).
  // 2. the input_shape defined in the model(dimension greater than 0), and the dimension matches with user
  // designate_dim.
  std::vector<int64_t> designated_dims = input_dims.at(name);
  size_t input_dim_size_ = designated_dims.size();

  GE_CHK_BOOL_RET_STATUS(model_input_dims_v.empty() || input_dim_size_ == model_input_dims_v.size(),
                         domi::PARAM_INVALID,
                         "user designated input_dim_num %zu does match input_dim_num %zu defined by model",
                         input_dim_size_,
                         model_input_dims_v.size());

  // replace with the user designated_dims
  user_input_dims_v.swap(designated_dims);

  return SUCCESS;
}

Status TensorFlowDataParser::CheckInputShape(const std::string &name) {
  for (uint32_t i = 0; i < user_input_dims_v.size(); i++) {
    // if input_shape has some placeholders, user should designate them.
    // dim i = 0, means empty tensor.
    // dim i = -1 or -2, means unknown shape.
    GE_CHK_BOOL_RET_STATUS(user_input_dims_v[i] >= kValidShapeMinValue, domi::PARAM_INVALID,
        "parse data node %s: shape contains placeholder, but not designated by user", name.c_str());
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, DATA, TensorFlowDataParser);
}  // namespace ge
