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

#include "parser/tensorflow/tensorflow_no_op_parser.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/op_def/no_op_op.h"
#include "parser/common/op_parser_factory.h"

using domi::TENSORFLOW;
using namespace ge::parser;

namespace ge {
Status TensorFlowNoOpParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  const NodeDef *node = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  NoOpOperator op;
  op.Name(node->name());

  return ConvertToOpDesc(op, op_dest);
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, NOOP, TensorFlowNoOpParser);
}  // namespace ge
