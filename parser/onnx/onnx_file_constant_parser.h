/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020~2022. All rights reserved.
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

#ifndef GE_PARSER_ONNX_ONNX_FILE_CONSTANT_PARSER_H_
#define GE_PARSER_ONNX_ONNX_FILE_CONSTANT_PARSER_H_

#include "parser/onnx/onnx_op_parser.h"
#include "proto/onnx/ge_onnx.pb.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxFileConstantParser : public OnnxOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::Operator &op_def) override;

 private:
  Status ParsePath(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const;
  Status ParseDataType(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const;
  void ParseShape(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const;
  Status GetTensorProto(const ge::onnx::NodeProto &node_proto, ge::onnx::TensorProto &tensor_proto) const;
  Status SetPathAttr(const ge::onnx::StringStringEntryProto &string_proto, ge::NamedAttrs &attrs) const;
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_FILE_CONSTANT_PARSER_H_
