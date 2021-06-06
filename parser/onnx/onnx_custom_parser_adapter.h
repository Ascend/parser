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

#ifndef PARSER_ONNX_ONNX_CUSTOM_PARSER_ADAPTER_H_
#define PARSER_ONNX_ONNX_CUSTOM_PARSER_ADAPTER_H_

#include "parser/onnx/onnx_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxCustomParserAdapter : public OnnxOpParser {
 public:
  /// @brief Parsing model file information
  /// @param [in] op_src model data to be parsed
  /// @param [out] op_dest model data after parsing
  /// @return SUCCESS parse successfully
  /// @return FAILED parse failed
  Status ParseParams(const Message *op_src, ge::Operator &op_dest) override;

  Status ParseParams(const Operator &op_src, Operator &op_dest);
};
}  // namespace ge

#endif  // PARSER_ONNX_ONNX_CUSTOM_PARSER_ADAPTER_H_
