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

#ifndef GE_PARSER_ONNX_ONNX_OP_PARSER_H_
#define GE_PARSER_ONNX_ONNX_OP_PARSER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <string>
#include <vector>
#include "framework/omg/parser/op_parser.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "proto/onnx/ge_onnx.pb.h"

using Status = domi::Status;

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxOpParser : public OpParser {
 public:
  /// @brief parse params
  /// @param [in] op_src        op to be parsed
  /// @param [out] op_dest      the parsed op
  /// @return SUCCESS           parse success
  /// @return FAILED            Parse failed
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override {
    return domi::SUCCESS;
  }

  /// @brief parse params
  /// @param [in] op_src        op to be parsed
  /// @param [out] op_dest      the parsed op
  /// @return SUCCESS           parse success
  /// @return FAILED            Parse failed
  Status ParseParams(const Message *op_src, ge::Operator &op_dest) override {
    return domi::SUCCESS;
  }

  /// @brief parsie weight
  /// @param [in] op_src        op to be parsed
  /// @param [out] op_dest      the parsed op
  /// @return SUCCESS           parsing success
  /// @return FAILED            parsing failed
  Status ParseWeights(const Message *op_src, ge::NodePtr &node) override {
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_OP_PARSER_H_
