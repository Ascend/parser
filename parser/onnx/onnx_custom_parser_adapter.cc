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

#include "parser/onnx/onnx_custom_parser_adapter.h"

#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"

using domi::ONNX;
using domi::ParseParamByOpFunc;
using domi::ParseParamFunc;

namespace ge {
Status OnnxCustomParserAdapter::ParseParams(const Message *op_src, ge::Operator &op_dest) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node_src = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  GE_CHECK_NOTNULL(node_src);
  GELOGI("Onnx op node name = %s, op type= %s, parse params.", node_src->name().c_str(), node_src->op_type().c_str());

  ParseParamFunc custom_op_parser =
      domi::OpRegistry::Instance()->GetParseParamFunc(op_dest.GetOpType(), node_src->op_type());
  GE_CHECK_NOTNULL(custom_op_parser);
  if (custom_op_parser(op_src, op_dest) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][Custom_Op_Parser] Custom parser params failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OnnxCustomParserAdapter::ParseParams(const Operator &op_src, Operator &op_dest) {
  ParseParamByOpFunc custom_op_parser = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(op_src.GetOpType());
  GE_CHECK_NOTNULL(custom_op_parser);

  if (custom_op_parser(op_src, op_dest) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][Custom_Op_Parser] failed, node name:%s, type:%s", op_src.GetName().c_str(),
           op_src.GetOpType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_PARSER_ADAPTER_CREATOR(ONNX, OnnxCustomParserAdapter);
}  // namespace ge
