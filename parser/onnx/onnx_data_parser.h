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

#ifndef GE_PARSER_ONNX_ONNX_DATA_PARSER_H_
#define GE_PARSER_ONNX_ONNX_DATA_PARSER_H_

#include <string>
#include <vector>
#include "parser/common/data_op_parser.h"
#include "parser/onnx/onnx_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxDataParser : public OnnxOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::Operator &op_def) override;

 private:
  Status ParseInputFromModel(const Message *op_src, ge::Operator &op_def);

  Status ParseInputFromUser(const ge::Operator &op_def);

  bool IsSubgraphDataOp() {
    return is_subgraph_data_op_;
  }

  int64_t ParseInputTensor(const ge::onnx::AttributeProto &attribute);

  std::vector<int64_t> model_input_dims_v_;

  std::vector<int64_t> user_input_dims_v_;

  bool is_subgraph_data_op_ = false;
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_DATA_PARSER_H_
