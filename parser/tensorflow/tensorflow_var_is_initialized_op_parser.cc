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
#include "framework/common/util.h"
#include "framework/common/op/ge_op_utils.h"
#include "parser/common/op_def/var_is_initialized_op_op.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_parser_register.h"

using namespace ge::parser;

namespace ge {
Status ParseParams(const Message *op_src, VarIsInitializedOpOperator *op) {
  GE_CHECK_NOTNULL(op_src);
  const NodeDef *node = reinterpret_cast<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  op->Name(node->name());

  return SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(VARISINITIALIZEDOP, VarIsInitializedOpOperator).SetParseParamsFn(ParseParams);

DOMI_REGISTER_TENSORFLOW_PARSER(ISVARIABLEINITIALIZED, VarIsInitializedOpOperator).SetParseParamsFn(ParseParams);
}  // namespace ge