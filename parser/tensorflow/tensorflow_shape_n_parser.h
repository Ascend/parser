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

#ifndef DOMI_OMG_PARSER_OP_PARSER_TENSORFLOW_SHAPE_N_H_
#define DOMI_OMG_PARSER_OP_PARSER_TENSORFLOW_SHAPE_N_H_

#include "common/op_def/shape_n_op.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

using domi::tensorflow::NodeDef;

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowShapeNParser : public TensorFlowOpParser {
  // AUTO GEN PLEASE DO NOT MODIFY IT
 public:
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

 protected:
  Status PreParseParams(const domi::tensorflow::NodeDef *node, ShapeNOperator *op);
  Status PostParseParams(const domi::tensorflow::NodeDef *node, ShapeNOperator *op);

  Status ParseN(const domi::tensorflow::NodeDef *node, ShapeNOperator *op);
  Status ParseInType(const domi::tensorflow::NodeDef *node, ShapeNOperator *op);
  Status ParseOutType(const domi::tensorflow::NodeDef *node, ShapeNOperator *op);

  // AUTO GEN PLEASE DO NOT MODIFY IT
};
}  // namespace ge

#endif  // DOMI_OMG_PARSER_OP_PARSER_TENSORFLOW_SHAPE_N_H_
