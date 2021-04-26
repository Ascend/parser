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

#include "parser/tensorflow/tensorflow_ref_switch_parser.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/op_def/ref_switch_op.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/util.h"

using domi::tensorflow::DataType;
using domi::tensorflow::DT_FLOAT;
using domi::tensorflow::AttrValue;
using domi::tensorflow::NodeDef;
using domi::TENSORFLOW;
using namespace ge::parser;

namespace ge {
// AUTO GEN PLEASE DO NOT MODIFY IT
Status TensorFlowRefSwitchParser::ParseT(const domi::tensorflow::NodeDef *node, RefSwitchOperator *op) {
  // The upper caller guarantees node is not empty.
  domi::tensorflow::AttrValue attr;

  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_T, attr),
                   op->T(domi::TensorAssign::ConvertTensorflowDataType(DT_FLOAT));
                   return SUCCESS);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, "type"), "check Attr T failed");

  domi::tensorflow::DataType tfType = attr.type();
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tfType);
  CHECK_FALSE_EXEC(type != ge::DataType::DT_UNDEFINED,
                   REPORT_CALL_ERROR("E19999", "Data type %s of node %s is not supported",
                                     DataType_Name(tfType).c_str(),
                                     node->name().c_str());
                   GELOGE(FAILED, "Data type %s of node %s is not supported.",
                          DataType_Name(tfType).c_str(), node->name().c_str());
                   return PARAM_INVALID);

  op->T(type);

  return SUCCESS;
}

Status TensorFlowRefSwitchParser::ParseParams(const Message *opSrc, ge::OpDescPtr &opDest) {
  GE_CHECK_NOTNULL(opSrc);
  const NodeDef *node = DOMI_DYNAMIC_CAST<const NodeDef *>(opSrc);
  GE_CHECK_NOTNULL(node);

  RefSwitchOperator op;
  op.Name(node->name());

  GELOGI("RefSwitch Op %s ParseParams Begin.", node->name().c_str());
  GE_RETURN_IF_ERROR(PreParseParams(node, &op));

  GE_RETURN_WITH_LOG_IF_ERROR(ParseT(node, &op), "Parse T for node %s failed.", node->name().c_str());

  GE_RETURN_IF_ERROR(PostParseParams(node, &op));

  Status status = ConvertToOpDesc(op, opDest);

  return status;
}

// AUTO GEN PLEASE DO NOT MODIFY IT
Status TensorFlowRefSwitchParser::PreParseParams(const domi::tensorflow::NodeDef *node, RefSwitchOperator *op) {
  return SUCCESS;
}

Status TensorFlowRefSwitchParser::PostParseParams(const domi::tensorflow::NodeDef *node, RefSwitchOperator *op) {
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, REFSWITCH, TensorFlowRefSwitchParser);
}  // namespace ge
