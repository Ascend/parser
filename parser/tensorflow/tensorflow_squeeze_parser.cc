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

#include "parser/tensorflow/tensorflow_squeeze_parser.h"
#include <memory>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/utils/type_utils.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/acl_graph_parser_util.h"

using domi::tensorflow::AttrValue;
using std::vector;
using std::shared_ptr;
using domi::TENSORFLOW;
using namespace ge::parser;

namespace ge {
Status TensorFlowSqueezeParser::ParseDesc(const domi::tensorflow::AttrValue &attr_value, ge::GeTensorDesc &ge_desc) {
  int32_t tf_datatype = 0;
  auto a_list = attr_value.list();
  GE_CHK_BOOL_RET_STATUS(TensorFlowUtil::ParseFromAttrValueList(ge_desc, a_list, 0, tf_datatype), domi::PARAM_INVALID,
                         "parse ge_desc failed.");
  uint32_t size_type;
  int64_t real_size = 1;
  int64_t tmp_dim = 0;

  auto data_type = ge_desc.GetDataType();
  bool type_ret = ge::TypeUtils::GetDataTypeLength(data_type, size_type);
  GE_IF_BOOL_EXEC(!type_ret,
                  REPORT_CALL_ERROR("E19999", "Data type %s is not supported",
                                    ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
                  GELOGE(domi::PARAM_INVALID, "Can't GetDataTypeLength of data_type: %s",
                         ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
                  return domi::PARAM_INVALID);
  // calculate size
  for (uint32_t j = 0; j < ge_desc.GetShape().GetDimNum(); ++j) {
    tmp_dim = ge_desc.GetShape().GetDim(j);
    GE_IF_BOOL_EXEC(tmp_dim < 0, real_size = tmp_dim * (-1) * real_size; continue;);
    PARSER_INT64_MULCHECK(real_size, tmp_dim);
    real_size *= tmp_dim;
  }
  PARSER_INT64_MULCHECK(real_size, size_type);
  ge::TensorUtils::SetSize(ge_desc, real_size * size_type);
  ge::TensorUtils::SetRealDimCnt(ge_desc, ge_desc.GetShape().GetDimNum());
  GELOGD("after translate tf_desc, datatype: %s, format: %s, real size: %ld, size_type: %u",
         ge::TypeUtils::DataTypeToSerialString(ge_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(ge_desc.GetFormat()).c_str(), real_size * size_type, size_type);
  return SUCCESS;
}

Status TensorFlowSqueezeParser::ParseParams(const Message *op_src, ge::OpDescPtr &op) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);

  const NodeDef *node = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  bool has_axis = true;
  bool has_dims = true;

  domi::tensorflow::AttrValue axis;
  domi::tensorflow::AttrValue dims;
  if (!TensorFlowUtil::FindAttrValue(node, SQUEEZE_ATTR_AXIS, axis)) {
    has_axis = false;
  }
  if (!TensorFlowUtil::FindAttrValue(node, SQUEEZE_ATTR_DIMS, dims)) {
    has_dims = false;
  }
  if (!has_axis && !has_dims) {
    return SUCCESS;
  }
  if (has_axis && has_dims) {
    REPORT_CALL_ERROR("E19999", "In NodeDef %s, Attr %s or %s not exist, check invalid", node->name().c_str(),
                      SQUEEZE_ATTR_AXIS.c_str(), SQUEEZE_ATTR_DIMS.c_str());
    GELOGE(FAILED, "In NodeDef %s dim and axis is error.", node->name().c_str());
    return domi::PARAM_INVALID;
  }

  domi::tensorflow::AttrValue_ListValue values;
  if (has_axis) {
    values = axis.list();
  } else {
    values = dims.list();
  }
  int i = 0;
  int size = values.i_size();
  vector<int32_t> v_result;
  for (i = 0; i < size; i++) {
    int32_t result = values.i(i);
    v_result.push_back(result);
  }
  if (!ge::AttrUtils::SetListInt(op, SQUEEZE_ATTR_AXIS, v_result)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", SQUEEZE_ATTR_AXIS.c_str(),
                      op->GetName().c_str(), op->GetType().c_str());
    GELOGE(FAILED, "Set squeeze axis attr failed");
    return FAILED;
  }

  domi::tensorflow::AttrValue input_attr_value;
  domi::tensorflow::AttrValue output_attr_value;

  GE_IF_BOOL_EXEC(
      GetParserContext().train_flag == true, ge::GeTensorDesc input_desc; ge::GeTensorDesc output_desc;

      if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_INPUT_TENSOR_DESC, input_attr_value)) {
        GE_CHK_BOOL_RET_STATUS(ParseDesc(input_attr_value, input_desc) == SUCCESS, FAILED, "parse input desc failed");
      }

      if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_attr_value)) {
        GE_CHK_BOOL_RET_STATUS(ParseDesc(output_attr_value, output_desc) == SUCCESS, FAILED,
                               "parse output desc failed");
      }

      if (!ge::AttrUtils::SetTensorDesc(op, RESHAPE_ATTR_NAME_INPUT_DESC, input_desc)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", RESHAPE_ATTR_NAME_INPUT_DESC.c_str(),
                          op->GetName().c_str(), op->GetType().c_str());
        GELOGE(FAILED, "Set input desc  failed");
        return FAILED;
      } if (!ge::AttrUtils::SetTensorDesc(op, RESHAPE_ATTR_NAME_OUTPUT_DESC, output_desc)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", RESHAPE_ATTR_NAME_OUTPUT_DESC.c_str(),
                          op->GetName().c_str(), op->GetType().c_str());
        GELOGE(FAILED, "Set output desc  failed");
        return FAILED;
      })
  return SUCCESS;
}
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, SQUEEZE, TensorFlowSqueezeParser);
}  // namespace ge
