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

#include "parser/common/op_def/frameworkop_op.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_parser_register.h"
#include "proto/tensorflow/tensor_shape.pb.h"

using domi::tensorflow::TensorShapeProto;
using domi::tensorflow::AttrValue;
using domi::TENSORFLOW;
using ge::parser::FRAMEWORKOP;

namespace ge {
Status ParseParams(const Message *op_src, FrameworkOpOperator *op) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);
  const NodeDef *node = reinterpret_cast<const NodeDef *>(op_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  string type = node->op();

  // Parsing input / output desc in attr
  domi::tensorflow::AttrValue input_attr_value;
  domi::tensorflow::AttrValue output_attr_value;
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_INPUT_TENSOR_DESC, input_attr_value)) {
    GE_CHK_STATUS_RET(
      TensorFlowUtil::TransTensorDescriptor(input_attr_value, op, TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG, type),
      "trans input_attr_value failed, op: %s", node->name().c_str());
  } else {
    GELOGD("Frameworkop has no input tensor desc, name:%s, type:%s.", node->name().c_str(), type.c_str());
    /// _Retval constructed from inference function do not has input_tensor_dec
    /// set input tensor desc for adding input tensor desc for op desc
    if (type == "_Retval") {
      ge::GeTensorDesc tensor_desc;
      op->InputTensorDesc(tensor_desc);
    }
  }
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_attr_value)) {
    GE_CHK_STATUS_RET(
      TensorFlowUtil::TransTensorDescriptor(output_attr_value, op, TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG, type),
      "trans output_attr_value failed, op: %s", node->name().c_str());
  } else {
    GELOGD("Frameworkop has no output tensor desc, name:%s, type:%s.", node->name().c_str(), type.c_str());
  }

  // Add index attribute, only Retval needs to be added
  domi::tensorflow::AttrValue index_attr_value;
  GE_IF_BOOL_EXEC(((type == "_Retval") && (TensorFlowUtil::FindAttrValue(node, ATTR_NAME_INDEX, index_attr_value))),
                  op->Index(index_attr_value.i()));

  NodeDef *pkg_node = new (std::nothrow) NodeDef();
  GE_CHECK_NOTNULL(pkg_node);

  pkg_node->CopyFrom(*node);

  domi::tensorflow::AttrValue attr_v;
  // Get the property opdef, if the property does not exist, return failure
  if (TensorFlowUtil::FindAttrValue(pkg_node, ge::ATTR_NAME_FRAMEWORK_OP_DEF, attr_v)) {
    op->TfOpDef(attr_v.s());
  } else {
    GE_CHK_BOOL_EXEC(type == "_Retval",
                     REPORT_INNER_ERROR("E19999", "In NodeDef:%s Attr:opdef is not exist, check invalid",
                                        pkg_node->name().c_str());
                     GE_DELETE_NEW_SINGLE(pkg_node);
                     return PARAM_INVALID, "In NodeDef %s Attr opdef is not exist.", pkg_node->name().c_str());
  }

  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_FRAMEWORK_OP_DEF);
  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_OUTPUT_TENSOR_DESC);
  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_INPUT_TENSOR_DESC);
  pkg_node->mutable_attr()->erase(ge::VAR_ATTR_NAME);

  // Get property func def
  domi::tensorflow::AttrValue func_attr_v;
  GE_IF_BOOL_EXEC(TensorFlowUtil::FindAttrValue(pkg_node, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF, func_attr_v),
                  op->FuncDefPkg(func_attr_v.s());
                  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_FRAMEWORK_FUNC_DEF));
  GELOGD("pkg_node name is %s, op is %s.", pkg_node->name().c_str(), pkg_node->op().c_str());
  if (pkg_node->op() == "DPOP") {
    pkg_node->set_op(pkg_node->name());
  }

  // Serialize nodedef into string and package as a whole
  string serialized_node;
  GE_IF_BOOL_EXEC(!pkg_node->SerializeToString(&serialized_node),
                  REPORT_CALL_ERROR("E19999", "Trans NodeDef:%s(%s) to string failed",
                                    pkg_node->name().c_str(), pkg_node->op().c_str());
                  GELOGE(PARAM_INVALID, "In FrameworkOp trans NodeDef to string failed.");
                  GE_DELETE_NEW_SINGLE(pkg_node); return PARAM_INVALID);

  op->NodeDefPkg(serialized_node);

  string node_def_pkg = op->GetNodeDefPkg();

  GELOGD("In FrameworkOp trans NodeDef to string success.op name : %s. nodedef_pkg [%s]", node->name().c_str(),
         node_def_pkg.c_str());

  // The framework operator of tensorflow preserves its framework type
  op->Frameworktype(TENSORFLOW);

  op->OriginalType(type);

  // Add shape attribute, only variables need to be added
  domi::tensorflow::AttrValue shape_value;
  if (TensorFlowUtil::FindAttrValue(node, VAR_ATTR_SHAPE, shape_value)) {
    vector<int64_t> shape_v;
    TensorShapeProto shape_proto = shape_value.shape();
    for (auto dim : shape_proto.dim()) {
      shape_v.push_back(dim.size());
    }
    op->AttrVector(VAR_ATTR_SHAPE, shape_v);
  }

  GE_DELETE_NEW_SINGLE(pkg_node);
  return SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(FRAMEWORKOP, FrameworkOpOperator).SetParseParamsFn(ParseParams);
}  // namespace ge
