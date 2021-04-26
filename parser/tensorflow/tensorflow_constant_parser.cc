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

#include "parser/tensorflow/tensorflow_constant_parser.h"
#include <map>
#include <memory>
#include <vector>
#include "parser/common/acl_graph_parser_util.h"
#include "parser/common/op_def/constant_op.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"
#include "register/tensor_assign.h"

using domi::tensorflow::NodeDef;
using domi::TENSORFLOW;
using ge::parser::CONSTANTOP;

namespace ge {
Status TensorFlowConstantParser::ParseDType(const domi::tensorflow::NodeDef *node, ConstantOperator *op) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(op);
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, TENSORFLOW_ATTR_DTYPE, attr),
                   op->DType(domi::TensorAssign::ConvertTensorflowDataType(domi::tensorflow::DT_FLOAT));
                   return SUCCESS);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, TENSORFLOW_ATTR_TYPE_TYPE),
                              "check Attr dtype fail");

  domi::tensorflow::DataType tf_type = attr.type();
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tf_type);

  op->DType(type);

  return SUCCESS;
}

Status TensorFlowConstantParser::ParseValue(const domi::tensorflow::NodeDef *node, const ge::OpDescPtr &opDesc) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(opDesc);
  domi::tensorflow::AttrValue attr_value;
  // Check that the attribute value must exist and get the value of value
  GE_CHK_BOOL_RET_STATUS(TensorFlowUtil::FindAttrValue(node, TENSORFLOW_ATTR_VALUE, attr_value),
                         domi::FAILED, "nodeDef %s Attr %s is not exist.", node->name().c_str(),
                         TENSORFLOW_ATTR_VALUE.c_str());
  // Check that the value attribute must be tensor
  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TENSOR),
                              "check Attr %s failed", TENSORFLOW_ATTR_VALUE.c_str());

  const domi::tensorflow::TensorProto &tensor = attr_value.tensor();

  GeTensorPtr weight = ge::parser::MakeShared<ge::GeTensor>();
  if (weight == nullptr) {
    REPORT_CALL_ERROR("E19999", "New GeTensor failed when parse node:%s", node->name().c_str());
    GELOGE(FAILED, "Create GeTensor fail when parse node:%s", node->name().c_str());
    return FAILED;
  }
  int64_t dataType = 0;
  GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(opDesc, TENSORFLOW_ATTR_DTYPE, dataType), INTERNAL_ERROR,
                         "get dtype fail");
  GE_CHK_STATUS_RET(domi::TensorAssign::SetGeTensorDataType(dataType, weight), "set ge tensor data type fail");

  GE_CHK_STATUS_RET(domi::TensorAssign::SetGeTensor(tensor, weight), "set ge tensor fail");
  GELOGI("TensorFlowConstantParser::ParseValue. TF op node name = %s", opDesc->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::SetTensor(opDesc, ATTR_NAME_WEIGHTS, weight), INTERNAL_ERROR,
                         "set tensor fail");
  return domi::SUCCESS;
}

Status TensorFlowConstantParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  GE_CHECK_NOTNULL(op_dest);
  const NodeDef *node = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  ConstantOperator op;
  op.Name(node->name());

  GE_RETURN_WITH_LOG_IF_ERROR(ParseDType(node, &op), "Parse dtype for node %s failed.", node->name().c_str());
  GE_CHK_STATUS_RET(ConvertToOpDesc(op, op_dest), "ConvertToOpDesc ret fail");
  GE_CHK_STATUS_RET(ParseValue(node, op_dest), "ParseValue ret fail");
  for (const auto &output_desc : op_dest->GetAllOutputsDescPtr()) {
    // Fixed input ND
    output_desc->SetFormat(ge::Format::FORMAT_ND);
    output_desc->SetOriginFormat(ge::Format::FORMAT_ND);
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, CONSTANTOP, TensorFlowConstantParser);
}  // namespace ge
