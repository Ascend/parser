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

#include "parser/tensorflow/tensorflow_shape_n_parser.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/op_def/shape_n_op.h"
#include "parser/common/util.h"

using domi::TENSORFLOW;
using domi::tensorflow::AttrValue;
using domi::tensorflow::DataType;
using domi::tensorflow::DT_FLOAT;
using domi::tensorflow::DT_INT32;
using namespace ge::parser;

namespace {
    const std::string kShapeAttrDtype = "out_type";
}  // namespace

namespace ge {
Status TensorFlowShapeNParser::ParseInType(const domi::tensorflow::NodeDef *node, ShapeNOperator *op) {
  // The upper caller guarantees the input params is not empty.
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_T, attr),
                   op->InType(domi::TensorAssign::ConvertTensorflowDataType(DT_FLOAT));
                   return SUCCESS);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, "type"), "check Attr T failed");

  domi::tensorflow::DataType tf_type = attr.type();
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tf_type);
  CHECK_FALSE_EXEC(type != ge::DataType::DT_UNDEFINED,
                   REPORT_CALL_ERROR("E19999", "Data type %s of node %s is not supported",
                                     DataType_Name(tf_type).c_str(), node->name().c_str());
                   GELOGE(FAILED, "Data type %s of node %s is not supported.",
                          DataType_Name(tf_type).c_str(), node->name().c_str());
                   return PARAM_INVALID);

  op->InType(type);

  return SUCCESS;
}

Status TensorFlowShapeNParser::ParseOutType(const domi::tensorflow::NodeDef *node, ShapeNOperator *op) {
  // The upper caller guarantees the input params is not empty.
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, kShapeAttrDtype, attr),
                   op->OutType(domi::TensorAssign::ConvertTensorflowDataType(DT_INT32));
                   return SUCCESS);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, "type"), "check Attr T failed");

  domi::tensorflow::DataType tf_type = attr.type();
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tf_type);
  CHECK_FALSE_EXEC(type != ge::DataType::DT_UNDEFINED,
                   REPORT_CALL_ERROR("E19999", "Data type %s of node %s is not supported",
                                     DataType_Name(tf_type).c_str(), node->name().c_str());
                   GELOGE(FAILED, "Data type %s of node %s is not supported.",
                          DataType_Name(tf_type).c_str(), node->name().c_str());
                   return PARAM_INVALID);

  op->OutType(type);

  return SUCCESS;
}

Status TensorFlowShapeNParser::ParseN(const domi::tensorflow::NodeDef *node, ShapeNOperator *op) {
  // The upper caller guarantees the input params is not empty.
  domi::tensorflow::AttrValue attr;
  const int64_t attr_n = 2;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, SHAPEN_ATTR_N, attr), op->N(attr_n); return SUCCESS);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, "int"), "check Attr N failed");

  op->N(attr.i());

  return SUCCESS;
}

Status TensorFlowShapeNParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  GE_CHECK_NOTNULL(op_dest);
  const NodeDef *node = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  ShapeNOperator op;
  op.Name(node->name());

  GE_RETURN_IF_ERROR(PreParseParams(node, &op));

  GE_RETURN_WITH_LOG_IF_ERROR(ParseInType(node, &op), "Parse in type for node %s failed.", node->name().c_str());

  GE_RETURN_WITH_LOG_IF_ERROR(ParseN(node, &op), "Parse N for node %s failed.", node->name().c_str());

  GE_RETURN_WITH_LOG_IF_ERROR(ParseOutType(node, &op), "Parse out type for node %s failed.", node->name().c_str());

  GE_RETURN_IF_ERROR(PostParseParams(node, &op));

  // add dynamic input/output
  domi::tensorflow::AttrValue attr_num;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, SHAPEN_ATTR_N, attr_num),
                   REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                                     node->name().c_str(), SHAPEN_ATTR_N.c_str());
                   GELOGE(FAILED, "Get Attr N failed in Node %s.", node->name().c_str());
                   return PARAM_INVALID);
  int32_t dynamic_tensor_num = attr_num.i();

  Status ret;
  domi::tensorflow::AttrValue output_attr_value;
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_attr_value)) {
    GE_CHK_STATUS_RET(
      TensorFlowUtil::TransTensorDescriptor(output_attr_value, &op, TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG),
      "trans output_attr_value failed, op: %s", node->name().c_str());
    ret = ConvertToOpDesc(op, op_dest);
    if (ret != SUCCESS) {
      return ret;
    }
  } else {
    ret = ConvertToOpDesc(op, op_dest);
    if (ret != SUCCESS) {
      return ret;
    }
    graphStatus status = op_dest->AddDynamicOutputDesc("y", dynamic_tensor_num);
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add Dynamic OuputDesc name:y to node:%s(%s) failed",
                        op_dest->GetName().c_str(), op_dest->GetType().c_str());
      GELOGE(FAILED, "Add dynamic output:y for node:%s failed.", op_dest->GetName().c_str());
      return FAILED;
    }
  }
  graphStatus status = op_dest->AddDynamicInputDesc("x", dynamic_tensor_num);
  if (status != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add Dynamic InputDesc name:x to node:%s(%s) failed",
                      op_dest->GetName().c_str(), op_dest->GetType().c_str());
    GELOGE(FAILED, "Add dynamic input:x for node:%s failed.", op_dest->GetName().c_str());
    return FAILED;
  }
  GELOGI("add dynamic input and output for op [%s], type[%s], name: %s, number:%d", op_dest->GetName().c_str(),
         op_dest->GetType().c_str(), SHAPEN_ATTR_N.c_str(), dynamic_tensor_num);
  return SUCCESS;
}

// AUTO GEN PLEASE DO NOT MODIFY IT
Status TensorFlowShapeNParser::PreParseParams(const domi::tensorflow::NodeDef *node, ShapeNOperator *op) {
  return SUCCESS;
}

Status TensorFlowShapeNParser::PostParseParams(const domi::tensorflow::NodeDef *node, ShapeNOperator *op) {
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, SHAPEN, TensorFlowShapeNParser);
}  // namespace ge
