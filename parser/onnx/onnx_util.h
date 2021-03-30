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

#ifndef PARSER_ONNX_ONNX_UTIL_PARSER_H_
#define PARSER_ONNX_ONNX_UTIL_PARSER_H_

#include "external/graph/types.h"

namespace OnnxDataType {
enum OnnxDataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
};
}

namespace ge {
const char *const kAttrNameValue = "value";
const char *const kAttrNameInput = "input_tensor";
const char *const kAttrNameIndex = "index";
const char *const kAttrNameIsSubgraphOp = "is_subgraph_op";
const char *const kOpTypeConstant = "Constant";
const char *const kOpTypeInput = "Input";

class OnnxUtil {
 public:
  static ge::DataType ConvertOnnxDataType(int64_t onnx_data_type);
  static int64_t CaculateDataSize(int64_t onnx_data_type);
  static void GenUniqueSubgraphName(int subgraph_index, const std::string &original_subgraph_name,
                                    const std::string &parent_node_name, std::string &unique_subgraph_name);
};
}  // namespace ge

#endif //PARSER_ONNX_ONNX_UTIL_PARSER_H_
