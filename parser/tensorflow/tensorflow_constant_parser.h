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

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_CONSTANT_PARSER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_CONSTANT_PARSER_H_

#include <string>
#include "common/op_def/constant_op.h"
#include "parser/common/data_op_parser.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

using domi::tensorflow::NodeDef;

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowConstantParser : public TensorFlowOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

 private:
  Status ParseDType(const domi::tensorflow::NodeDef *node, ConstantOperator *op);
  Status ParseValue(const domi::tensorflow::NodeDef *node, const ge::OpDescPtr &opDesc);
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_CONSTANT_PARSER_H_
