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

#include "onnx_util.h"
#include <map>

namespace {
const std::map<uint32_t, ge::DataType> onnx_data_type_map = {
    {OnnxDataType::UNDEFINED, ge::DataType::DT_UNDEFINED}, {OnnxDataType::FLOAT, ge::DataType::DT_FLOAT},
    {OnnxDataType::UINT8, ge::DataType::DT_UINT8},         {OnnxDataType::INT8, ge::DataType::DT_INT8},
    {OnnxDataType::UINT16, ge::DataType::DT_UINT16},       {OnnxDataType::INT16, ge::DataType::DT_INT16},
    {OnnxDataType::INT32, ge::DataType::DT_INT32},         {OnnxDataType::INT64, ge::DataType::DT_INT64},
    {OnnxDataType::STRING, ge::DataType::DT_STRING},       {OnnxDataType::BOOL, ge::DataType::DT_BOOL},
    {OnnxDataType::FLOAT16, ge::DataType::DT_FLOAT16},     {OnnxDataType::DOUBLE, ge::DataType::DT_DOUBLE},
    {OnnxDataType::UINT32, ge::DataType::DT_UINT32},       {OnnxDataType::UINT64, ge::DataType::DT_UINT64},
    {OnnxDataType::COMPLEX64, ge::DataType::DT_COMPLEX64}, {OnnxDataType::COMPLEX128, ge::DataType::DT_COMPLEX128},
    {OnnxDataType::BFLOAT16, ge::DataType::DT_UNDEFINED},
};

const std::map<uint32_t, int64_t> onnx_data_type_size_map = {
    {OnnxDataType::FLOAT, sizeof(float)},     {OnnxDataType::UINT8, sizeof(uint8_t)},
    {OnnxDataType::INT8, sizeof(int8_t)},     {OnnxDataType::UINT16, sizeof(uint16_t)},
    {OnnxDataType::INT16, sizeof(int16_t)},   {OnnxDataType::INT32, sizeof(int32_t)},
    {OnnxDataType::INT64, sizeof(int64_t)},   {OnnxDataType::STRING, sizeof(std::string)},
    {OnnxDataType::BOOL, sizeof(bool)},       {OnnxDataType::FLOAT16, 2},
    {OnnxDataType::DOUBLE, sizeof(double)},   {OnnxDataType::UINT32, sizeof(uint32_t)},
    {OnnxDataType::UINT64, sizeof(uint64_t)}, {OnnxDataType::COMPLEX64, 8},
    {OnnxDataType::COMPLEX128, 16},           {OnnxDataType::BFLOAT16, 2},
};
}

namespace ge {
ge::DataType OnnxUtil::ConvertOnnxDataType(int64_t onnx_data_type) {
  auto search = onnx_data_type_map.find(onnx_data_type);
  if (search != onnx_data_type_map.end()) {
    return search->second;
  } else {
    return ge::DataType::DT_UNDEFINED;
  }
}

int64_t OnnxUtil::CaculateDataSize(int64_t onnx_data_type) {
  auto search = onnx_data_type_size_map.find(onnx_data_type);
  if (search != onnx_data_type_size_map.end()) {
    return search->second;
  } else {
    return ge::DataType::DT_UNDEFINED;
  }
}

void OnnxUtil::GenUniqueSubgraphName(int subgraph_index, const std::string &original_subgraph_name,
                                     const std::string &parent_node_name, std::string &unique_subgraph_name) {
  unique_subgraph_name = parent_node_name + "_" + std::to_string(subgraph_index) + "_" + original_subgraph_name;
}
}  // namespace ge
