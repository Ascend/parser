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

#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "parser/common/op_def/variable_op.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_parser_register.h"

using domi::tensorflow::AttrValue;
using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorShapeProto;
using namespace ge::parser;

namespace ge {
const std::string SERIALIZE_FORMAT = "serialize_format";
/* Original definition of variablev2 operator
node_def {
      name: "Variable_7/Momentum"
      op: "VariableV2"
      device: "/job:localhost/replica:0/task:0/device:CPU:0"
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@Variable_7"
          }
        }
      }
      attr {
        key: "_var_format"
        value {
          s: "4D"
        }
      }
      attr {
        key: "container"
        value {
          s: ""
        }
      }
      attr {
        key: "dtype"
        value {
          type: DT_FLOAT
        }
      }
      attr {
        key: "shape"
        value {
          shape {
            dim {
              size: 10
            }
          }
        }
      }
      attr {
        key: "shared_name"
        value {
          s: ""
        }
      }
    }
*/
static Status ParseSrcType(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, VAR_ATTR_DTYPE, attr),
                   REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                                     node->name().c_str(), VAR_ATTR_DTYPE.c_str());
                   GELOGE(FAILED, "Attr %s does not exist in NodeDef %s.", VAR_ATTR_DTYPE.c_str(),
                          node->name().c_str());
                   return PARAM_INVALID);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, TENSORFLOW_ATTR_TYPE_TYPE),
                              "check Attr type failed");

  domi::tensorflow::DataType tf_type = attr.type();
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tf_type);

  CHECK_FALSE_EXEC(type != ge::DataType::DT_UNDEFINED,
                   REPORT_CALL_ERROR("E19999", "Data type %s of node %s is not supported",
                                     DataType_Name(tf_type).c_str(), node->name().c_str());
                   GELOGE(FAILED, "Data type %s of node %s is not supported.",
                          DataType_Name(tf_type).c_str(), node->name().c_str());
                   return PARAM_INVALID);

  op->SrcType(type);
  return SUCCESS;
}

Status ParseContainer(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, VAR_ATTR_CONTAINER, attr),
                   REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                                     node->name().c_str(), VAR_ATTR_CONTAINER.c_str());
                   GELOGE(FAILED, "Attr %s does not exist in NodeDef %s.", VAR_ATTR_CONTAINER.c_str(),
                          node->name().c_str());
                   return PARAM_INVALID);
  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, TENSORFLOW_ATTR_TYPE_STRING),
                              "check Attr s failed");

  std::string container = attr.s();

  op->Container(container);
  return SUCCESS;
}

Status ParseSharedName(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(
    TensorFlowUtil::FindAttrValue(node, VAR_ATTR_SHARED_NAME, attr),
    REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                      node->name().c_str(), VAR_ATTR_SHARED_NAME.c_str());
    GELOGE(FAILED, "Attr %s does not exist in NodeDef %s.", VAR_ATTR_SHARED_NAME.c_str(), node->name().c_str());
    return PARAM_INVALID);
  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, TENSORFLOW_ATTR_TYPE_STRING),
                              "check Attr s failed");

  std::string shared_name = attr.s();
  op->SharedName(shared_name);

  return SUCCESS;
}

static Status ParseVarName(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  domi::tensorflow::AttrValue attr;
  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, ge::VAR_ATTR_NAME, attr),
                   REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                                     node->name().c_str(), VAR_ATTR_NAME.c_str());
                   GELOGE(FAILED, "Attr %s does not exist in NodeDef %s.", ge::VAR_ATTR_NAME.c_str(),
                          node->name().c_str());
                   return PARAM_INVALID);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, TENSORFLOW_ATTR_TYPE_STRING),
                              "check Attr s failed");

  std::string var_name = attr.s();
  op->SharedName(var_name);

  return SUCCESS;
}

static Status InitOutTensor(const vector<int64_t> &shape, int64_t data_type, ge::GeTensorDesc &out_tensor_desc,
                            ge::Format format) {
  out_tensor_desc.SetFormat(format);

  out_tensor_desc.SetDataType((ge::DataType)data_type);
  ge::TensorUtils::SetReuseInput(out_tensor_desc, false);
  ge::TensorUtils::SetRealDimCnt(out_tensor_desc, shape.size());

  out_tensor_desc.SetShape(ge::GeShape(shape));
  int64_t size = out_tensor_desc.GetShape().GetShapeSize();
  size *= sizeof(float);
  ge::TensorUtils::SetSize(out_tensor_desc, size);
  return SUCCESS;
}

static Status ParseVarShape(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  string node_src_name = node->name();
  domi::tensorflow::AttrValue attr_value;

  if (!TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, attr_value)) {
    REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                      node->name().c_str(), ATTR_NAME_OUTPUT_TENSOR_DESC.c_str());
    GELOGE(FAILED, "In NodeDef %s Attr %s is not exist.", node_src_name.c_str(),
           ge::ATTR_NAME_OUTPUT_TENSOR_DESC.c_str());
    return FAILED;
  }

  ge::GeTensorDesc infer_shape_domi_desc;
  domi::tensorflow::AttrValue_ListValue attr_list = attr_value.list();
  int32_t tf_datatype = 0;
  GE_CHK_BOOL_RET_STATUS(TensorFlowUtil::ParseFromAttrValueList(infer_shape_domi_desc, attr_list, 0, tf_datatype),
                         PARAM_INVALID, "parse domi_desc failed.");

  ge::Format src_format = ge::FORMAT_ND;

  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, VAR_ATTR_SHAPE, attr_value),
                   REPORT_CALL_ERROR("E19999", "In NodeDef:%s attr:%s not exist, check invalid",
                                     node->name().c_str(), VAR_ATTR_SHAPE.c_str());
                   GELOGE(FAILED, "Attr %s does not exist in NodeDef %s.", VAR_ATTR_SHAPE.c_str(),
                          node->name().c_str());
                   return PARAM_INVALID);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_SHAPE),
                              "check Attr s failed");

  const TensorShapeProto &data_shape = attr_value.shape();

  vector<int64_t> var_dims_v;
  for (int32_t i = 0; i < data_shape.dim_size(); i++) {
    var_dims_v.push_back(data_shape.dim(i).size());
  }

  op->VarShape(var_dims_v);

  ge::GeTensorDesc out_tensor_desc;
  GE_RETURN_WITH_LOG_IF_ERROR(InitOutTensor(var_dims_v, op->GetVarSrcType(), out_tensor_desc, src_format),
                              "Init Output Tensor failed");

  op->OutputTensorDesc(out_tensor_desc);

  return SUCCESS;
}

static void ParsePlacement(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  string node_src_name = node->name();
  domi::tensorflow::AttrValue attr_value;
  GELOGI("Start to parse placement, %s", node_src_name.c_str());
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_VARIABLE_PLACEMENT, attr_value)) {
    std::string placement = attr_value.s();
    op->Placement(placement);
  }
}

static void ParseMemType(const domi::tensorflow::NodeDef *node, VariableOperator *op) {
  // The upper caller guarantees input params is not empty.
  string node_src_name = node->name();
  domi::tensorflow::AttrValue attr_value;
  GELOGI("Start to parse mem_type, %s", node_src_name.c_str());
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_OUTPUT_MEMORY_TYPE, attr_value)) {
    uint32_t mem_type = attr_value.i();
    op->MemType(mem_type);
  }
}

Status ParseParams(const Message *op_src, VariableOperator *op) {
  GE_CHECK_NOTNULL(op_src);
  const NodeDef *node = reinterpret_cast<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  string node_op = node->op();
  if (node_op == TEMPORARYVARIABLE) {
    GE_RETURN_IF_ERROR(ParseVarName(node, op));
  } else {
    GE_RETURN_IF_ERROR(ParseContainer(node, op));
    GE_RETURN_IF_ERROR(ParseSharedName(node, op));
  }

  GE_RETURN_IF_ERROR(ParseSrcType(node, op));
  GE_RETURN_IF_ERROR(ParseVarShape(node, op));
  ParsePlacement(node, op);
  ParseMemType(node, op);

  GELOGD("VariabeV2 OP parser params success.op name : %s.", node->name().c_str());

  return SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(VARIABLE, VariableOperator).SetParseParamsFn(ParseParams);

DOMI_REGISTER_TENSORFLOW_PARSER(VARHANDLEOP, VariableOperator).SetParseParamsFn(ParseParams);

DOMI_REGISTER_TENSORFLOW_PARSER(TEMPORARYVARIABLE, VariableOperator).SetParseParamsFn(ParseParams);
}  // namespace ge
