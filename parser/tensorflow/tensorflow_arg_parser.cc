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

#include "parser/common/op_def/arg_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_parser_register.h"

using domi::tensorflow::AttrValue;

namespace ge {
namespace {
const char *const kSerializeFormat = "serialize_format";
}  // namespace
Status ParseParams(const Message *op_src, ArgOpOperator *op) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);
  const NodeDef *node = reinterpret_cast<const NodeDef *>(op_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  domi::tensorflow::AttrValue output_attr_value;
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_attr_value)) {
    GE_CHK_STATUS_RET(
      TensorFlowUtil::TransTensorDescriptor(output_attr_value, op, TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG),
      "trans output_attr_value failed, op: %s", node->name().c_str());
    // For the needs of the Data operator, copy the output description to the input description
    GE_CHK_STATUS_RET(TensorFlowUtil::TransTensorDescriptor(output_attr_value, op, TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG),
                      "trans output_attr_value failed, op: %s", node->name().c_str());

    domi::tensorflow::AttrValue_ListValue attr_list = output_attr_value.list();
    GetParserContext().format =
      static_cast<domi::tagDomiTensorFormat>(attr_list.func(0).attr().at(kSerializeFormat).i());
  } else {
    /// _Arg constructed from inference function do not has input_tensor_dec
    /// set input & output tensor desc for adding input & output tensor desc for op desc
    ge::GeTensorDesc tensor_desc;
    op->InputTensorDesc(tensor_desc);
    op->OutputTensorDesc(tensor_desc);
  }

  domi::tensorflow::AttrValue index_attr_value;
  if (TensorFlowUtil::FindAttrValue(node, ATTR_NAME_INDEX, index_attr_value)) {
    op->Index(index_attr_value.i());
  }

  GELOGI("In _ArgOp trans success.op name : %s.", node->name().c_str());

  return SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(ge::parser::ARG, ArgOpOperator).SetParseParamsFn(ParseParams);
}  // namespace ge
