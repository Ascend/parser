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

#ifndef GE_PARSER_ONNX_ONNX_CONSTANT_PARSER_H_
#define GE_PARSER_ONNX_ONNX_CONSTANT_PARSER_H_

#include <string>
#include "common/util.h"
#include "parser/common/data_op_parser.h"
#include "parser/onnx/onnx_op_parser.h"

using ge::onnx::NodeProto;

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxConstantParser : public OnnxOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::Operator &op_def) override;

 private:
  Status ParseConstFromInput(const ge::onnx::NodeProto *op_src, ge::Operator &op_def);
  Status ParseConvertTensor(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor);
  Status ParseConvertData(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor, int count);
  void ParseConvertDataElements(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor, int count,
                               int64_t data_type);
  Status ParseConvertDataType(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor);

  template <typename T>
  static Status SetTensorData(int32_t val_size, const google::protobuf::RepeatedField<T> &val_vector, int count,
                              Tensor &tensor) {
    bool zeros_like = (count != val_size && val_size == 1);
    T *addr = new (std::nothrow) T[count]();
    GE_CHECK_NOTNULL(addr);
    int minCount = (count > val_size) ? val_size : count;
    if (!zeros_like) {
      for (int32_t i = 0; i < minCount; i++) {
        *(addr + i) = val_vector.Get(i);
      }
      for (int32_t i = minCount; i < count; i++) {
        *(addr + i) = val_vector.Get(minCount - 1);
      }
    } else {
      for (int32_t i = 0; i < count; i++) {
        *(addr + i) = val_vector.Get(0);
      }
    }
    tensor.SetData(reinterpret_cast<uint8_t *>(addr), count * sizeof(T));
    GE_DELETE_NEW_ARRAY(addr);
    return SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_CONSTANT_PARSER_H_
