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

#include "parser/common/data_op_parser.h"
#include <cstdlib>
#include "parser/common/acl_graph_parser_util.h"
#include "omg/parser/parser_inner_ctx.h"
#include "common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"

namespace {
const int kDataMemAlignSize = 32;
const int kTwoTimesAlign = 2;
const int kDynamicBatchInputSize = -1;
const uint32_t kScalarLength = 1;
}  // namespace

namespace ge {
FMK_FUNC_HOST_VISIBILITY Status DataOpParser::ParseShape(const vector<int64_t> &shape, ge::OpDescPtr op) {
  GE_RETURN_WITH_LOG_IF_FALSE(op != nullptr, "[Check][Param] ParseShape failed for data_op, op is null");

  const string &data_op_name = op->GetName();
  GetParserContext().input_dims.emplace(data_op_name, shape);

  int64_t attr_type = 0;
  ge::DataType data_type;
  if (ge::AttrUtils::GetInt(op, ge::DATA_ATTR_NAME_DATA_TYPE, attr_type)) {
    data_type = static_cast<ge::DataType>(attr_type);
  } else {
    data_type = ge::DT_FLOAT;
  }

  // convert input
  vector<int64_t> def_format_shape(shape);

  ge::GeTensorDesc i_tensor_desc;
  ge::GeTensorDesc o_tensor_desc;
  const unordered_map<string, domiTensorFormat_t> &input_nodes_format_map = GetParserContext().input_nodes_format_map;
  auto map_iter = input_nodes_format_map.find(data_op_name);
  if (map_iter != input_nodes_format_map.end() && map_iter->second == domi::DOMI_TENSOR_NC1HWC0) {
    // Input 5D NC1HWC0
    GE_RETURN_WITH_LOG_IF_ERROR(Init5DInputTensor(def_format_shape, i_tensor_desc),
                                "[Invoke][Init5DInputTensor] failed");
    // Output
    GE_RETURN_WITH_LOG_IF_ERROR(Init5DOutputTensor(def_format_shape, o_tensor_desc),
                                "[Invoke][Init5DOutputTensor] failed");
  } else {
    // No need to consider AIPP here,
    // The adjustdatanodedesc function of model_builder will process the
    // input_desc and output_desc of AIPP's data node.
    // Without AIPP, the data of input float is kept in cctranstensor implementation.
    // The cast operator can not run in the pvmodel simulation environment,
    // so the input data conversion processing maintains the original state.
    // To be modified after AICPU operators support pvmodel.
    if (data_type == ge::DT_FLOAT) {
      // Input
      GE_RETURN_WITH_LOG_IF_ERROR(InitInputTensor(def_format_shape, i_tensor_desc),
                                  "[Invoke][InitInputTensor] failed");
      // Output
      GE_RETURN_WITH_LOG_IF_ERROR(InitOutputTensor(def_format_shape, o_tensor_desc),
                                  "[Invoke][InitOutputTensor] failed");
    } else {
      // Input
      GE_RETURN_WITH_LOG_IF_ERROR(InitNDTensor(def_format_shape, data_type, i_tensor_desc),
                                  "[Invoke][InitNDTensor] failed");
      // Output
      GE_RETURN_WITH_LOG_IF_ERROR(InitNDTensor(def_format_shape, data_type, o_tensor_desc),
                                  "[Invoke][InitNDTensor] failed");
    }
  }
  i_tensor_desc.SetFormat(ge::TypeUtils::DomiFormatToFormat(GetParserContext().format));
  i_tensor_desc.SetOriginFormat(ge::TypeUtils::DomiFormatToFormat(GetParserContext().format));
  o_tensor_desc.SetFormat(ge::TypeUtils::DomiFormatToFormat(GetParserContext().format));
  if (op->AddInputDesc(i_tensor_desc) != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddInputDesc failed for op %s.", op->GetName().c_str());
    GELOGE(domi::INTERNAL_ERROR, "[Invoke][AddInputDesc] failed for op %s.", op->GetName().c_str());
    return FAILED;
  }
  if (op->AddOutputDesc(o_tensor_desc) != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddOutputDesc failed for op %s.", op->GetName().c_str());
    GELOGE(domi::INTERNAL_ERROR, "[Invoke][AddOutputDesc] failed for op %s.", op->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status DataOpParser::Init5DInputTensor(const vector<int64_t> &shape, ge::GeTensorDesc &tensor_desc) {
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetFormat(static_cast<ge::Format>(domi::DOMI_TENSOR_NC1HWC0));
  ge::TensorUtils::SetReuseInput(tensor_desc, false);
  ge::TensorUtils::SetRealDimCnt(tensor_desc, shape.size());
  tensor_desc.SetShape(ge::GeShape(shape));

  int64_t tensor_size = 0;
  ge::graphStatus graph_status = ge::TensorUtils::GetTensorSizeInBytes(tensor_desc, tensor_size);
  if (graph_status != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "GetTensorSizeInBytes failed");
    GELOGE(FAILED, "[Invoke][GetTensorSizeInBytes] failed!");
    return domi::FAILED;
  }
  // Set the actual occupied space size
  ge::TensorUtils::SetSize(tensor_desc, tensor_size);
  return SUCCESS;
}

Status DataOpParser::InitNDTensor(const vector<int64_t> &shape, ge::DataType data_type, ge::GeTensorDesc &tensor_desc) {
  // Fixed input ND
  tensor_desc.SetFormat(static_cast<ge::Format>(DOMI_TENSOR_ND));
  tensor_desc.SetDataType(data_type);
  tensor_desc.SetOriginDataType(data_type);
  ge::TensorUtils::SetReuseInput(tensor_desc, false);
  ge::TensorUtils::SetRealDimCnt(tensor_desc, shape.size());
  tensor_desc.SetShape(ge::GeShape(shape));
  tensor_desc.SetOriginShape(ge::GeShape(shape));

  int64_t size = kScalarLength;
  if (!tensor_desc.GetShape().GetDims().empty()) {
    size = tensor_desc.GetShape().GetShapeSize();
  }
  uint32_t type_size = 0;
  if (ge::TypeUtils::GetDataTypeLength(data_type, type_size)) {
    PARSER_INT64_UINT32_MULCHECK(size, type_size);
    size *= type_size;
  } else {
    PARSER_INT64_UINT32_MULCHECK(size, static_cast<uint32_t>(sizeof(float)));
    size *= sizeof(float);
  }
  ge::TensorUtils::SetSize(tensor_desc, size);
  return SUCCESS;
}

Status DataOpParser::Init5DOutputTensor(const vector<int64_t> &shape, ge::GeTensorDesc &output) {
  output.SetDataType(ge::DT_FLOAT16);
  output.SetFormat(static_cast<ge::Format>(domi::DOMI_TENSOR_NC1HWC0));
  ge::TensorUtils::SetReuseInput(output, false);
  ge::TensorUtils::SetRealDimCnt(output, shape.size());
  output.SetShape(ge::GeShape(shape));

  int64_t output_size = 0;
  ge::graphStatus graph_status = ge::TensorUtils::GetTensorMemorySizeInBytes(output, output_size);
  if (graph_status != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "GetTensorMemorySizeInBytes failed!");
    GELOGE(FAILED, "[Invoke][GetTensorMemorySizeInBytes] failed!");
    return domi::FAILED;
  }
  // Set the actual occupied space size
  ge::TensorUtils::SetSize(output, output_size);
  return SUCCESS;
}

Status DataOpParser::InitInputTensor(const vector<int64_t> &shape, ge::GeTensorDesc &input) {
  input.SetFormat(static_cast<ge::Format>(domiTensorFormat_t(DOMI_TENSOR_ND)));
  input.SetDataType(ge::DT_FLOAT);
  input.SetOriginDataType(ge::DT_FLOAT);
  ge::TensorUtils::SetReuseInput(input, false);

  input.SetShape(ge::GeShape(shape));
  input.SetOriginShape(ge::GeShape(shape));
  int64_t size = 0;
  // No need to check dynamic_batch_size since its first dim is -1.
  if (input.GetShape().GetDim(0) != -1) {
    size = input.GetShape().GetShapeSize();
  }
  PARSER_INT64_UINT32_MULCHECK(size, static_cast<uint32_t>(sizeof(float)));
  ge::TensorUtils::SetSize(input, size * sizeof(float));

  return SUCCESS;
}

Status DataOpParser::InitOutputTensor(const vector<int64_t> &shape, ge::GeTensorDesc &output) {
  int64_t output_size = 0;
  ge::GeShape output_shape = ge::GeShape(shape);
  ge::Format format = ge::FORMAT_ND;
  ge::DataType data_type = ge::DT_FLOAT;
  output.SetFormat(format);
  output.SetDataType(data_type);
  ge::TensorUtils::SetReuseInput(output, false);
  ge::TensorUtils::SetRealDimCnt(output, shape.size());
  output.SetShape(output_shape);

  ge::graphStatus graph_status = ge::TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_size);
  if (graph_status != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "CalcTensorMemSize failed, shape:%s, format:%s, datatype:%s",
                      output_shape.ToString().c_str(),
                      ge::TypeUtils::FormatToSerialString(format).c_str(),
                      ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    GELOGE(FAILED, "[Invoke][CalcTensorMemSize] failed, shape:%s, format:%s, datatype:%s",
           output_shape.ToString().c_str(),
           ge::TypeUtils::FormatToSerialString(format).c_str(),
           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  }

  if (output_size == kDynamicBatchInputSize) {
    GELOGI("After calc tensor memory size, output_mem_size = %ld", output_size);
    return SUCCESS;
  }

  int64_t size = output_size;
  auto valid_max_size = INT64_MAX - kTwoTimesAlign * kDataMemAlignSize;
  if (size > valid_max_size || size < 0) {
    REPORT_INNER_ERROR("E19999", "updated mem size is out of data range [0, %ld], shape:%s, format:%s, datatype:%s",
                       valid_max_size, output_shape.ToString().c_str(),
                       ge::TypeUtils::FormatToSerialString(format).c_str(),
                       ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    GELOGE(FAILED, "[Check][Size] updated mem size is out of data range [0, %ld], shape:%s, format:%s, datatype:%s",
           valid_max_size, output_shape.ToString().c_str(),
           ge::TypeUtils::FormatToSerialString(format).c_str(),
           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  } else {
    size = ((size + kTwoTimesAlign * kDataMemAlignSize - 1) / kDataMemAlignSize) * kDataMemAlignSize;
  }
  // Set the actual occupied space size
  ge::TensorUtils::SetSize(output, size);
  return SUCCESS;
}
}  // namespace ge
