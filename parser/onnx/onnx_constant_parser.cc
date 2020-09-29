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

#include "onnx_constant_parser.h"
#include <map>
#include <vector>
#include "common/ge/ge_util.h"
#include "common/util.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_adapter.h"
#include "parser/common/op_parser_factory.h"
#include "parser/onnx/onnx_util.h"

using ge::onnx::NodeProto;
using ge::onnx::TensorProto;
using domi::ONNX;
using GeShape = ge::GeShape;
using GeTensorDesc = ge::GeTensorDesc;
using namespace ge::parser;

namespace ge {
Status OnnxConstantParser::ParseConvertData(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor, int count) {
  int64_t data_type = tensor_proto.data_type();
  if (ge::OnnxUtil::ConvertOnnxDataType(data_type) == ge::DataType::DT_UNDEFINED) {
    GELOGE(FAILED, "data_type %ld not support.", data_type);
    return FAILED;
  }

  if (count == 0) {
    GELOGI("At least one dim equals zero, result in the count equal to zero.");
    return SUCCESS;
  }

  std::map<uint32_t, int32_t> datatype_val_size_map = {
      {OnnxDataType::INT32, tensor_proto.int32_data_size()},
      {OnnxDataType::INT64, tensor_proto.int64_data_size()},
      {OnnxDataType::STRING, tensor_proto.string_data_size()},
      {OnnxDataType::FLOAT, tensor_proto.float_data_size()},
      {OnnxDataType::DOUBLE, tensor_proto.double_data_size()},
      {OnnxDataType::UINT64, tensor_proto.uint64_data_size()},
      {OnnxDataType::UINT8, 0},
      {OnnxDataType::INT8, 0},
      {OnnxDataType::UINT16, 0},
      {OnnxDataType::INT16, 0},
      {OnnxDataType::BOOL, 0},
      {OnnxDataType::FLOAT16, 0},
      {OnnxDataType::UINT32, 0},
      {OnnxDataType::COMPLEX64, 0},
      {OnnxDataType::COMPLEX128, 0},
      {OnnxDataType::BFLOAT16, 0},
  };

  int32_t datatype_val_size = 0;
  auto iter = datatype_val_size_map.find(data_type);
  if (iter != datatype_val_size_map.end()) {
    datatype_val_size = iter->second;
  } else {
    GELOGE(domi::PARAM_INVALID, "data_type %ld not support.", data_type);
    return FAILED;
  }

  // find raw data
  if (datatype_val_size == 0) {
    if (tensor_proto.raw_data().empty()) {
      GELOGE(domi::PARAM_INVALID, "tensor_proto has no data() elements or raw_data()");
      return FAILED;
    }

    if (data_type == OnnxDataType::STRING) {
      tensor.SetData(tensor_proto.raw_data());
    } else {
      tensor.SetData(reinterpret_cast<const uint8_t *>(tensor_proto.raw_data().c_str()),
                     tensor_proto.raw_data().size());
    }
    GELOGD("Raw data size is : %zu", tensor_proto.raw_data().size());
    return SUCCESS;
  }

  // find _data() elements
  ParseConvertDataElements(tensor_proto, tensor, count, data_type);
  return SUCCESS;
}

void OnnxConstantParser::ParseConvertDataElements(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor,
                                                  int count, int64_t data_type) {
  switch (data_type) {
    case OnnxDataType::INT32:
      (void)SetTensorData(tensor_proto.int32_data_size(), tensor_proto.int32_data(), count, tensor);
      break;
    case OnnxDataType::INT64:
      (void)SetTensorData(tensor_proto.int64_data_size(), tensor_proto.int64_data(), count, tensor);
      break;
    case OnnxDataType::STRING: {
      std::vector<std::string> data;
      for (auto str_data : tensor_proto.string_data()) {
        data.emplace_back(str_data);
      }
      tensor.SetData(data);
      break;
    }
    case OnnxDataType::FLOAT:
      (void)SetTensorData(tensor_proto.float_data_size(), tensor_proto.float_data(), count, tensor);
      break;
    case OnnxDataType::DOUBLE:
      (void)SetTensorData(tensor_proto.double_data_size(), tensor_proto.double_data(), count, tensor);
      break;
    case OnnxDataType::UINT64:
      (void)SetTensorData(tensor_proto.uint64_data_size(), tensor_proto.uint64_data(), count, tensor);
      break;
    default:
      break;
  }
}

Status OnnxConstantParser::ParseConvertTensor(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor) {
  // convert shape and format
  std::vector<int64_t> tmp_shape;
  int count = 1;
  for (int i = 0; i < tensor_proto.dims_size(); i++) {
    tmp_shape.push_back(tensor_proto.dims(i));
    int64_t dim = tmp_shape[i];
    // support weights shape [0],have no weights
    if (dim < 0 || (count != 0 && (dim >= INT64_MAX / count))) {
      GELOGE(FAILED, "Dim size is invalid, dim is less than zero or dim size exceeds INT64_MAX.");
      return FAILED;
    }
    count *= dim;
  };
  TensorDesc tensor_desc = tensor.GetTensorDesc();
  tensor_desc.SetShape(ge::Shape(tmp_shape));
  tensor_desc.SetFormat(static_cast<ge::Format>(GetParserContext().format));
  tensor.SetTensorDesc(tensor_desc);

  // set data
  if (ParseConvertData(tensor_proto, tensor, count) != SUCCESS) {
    GELOGE(FAILED, "Convert ge tensor data and format failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OnnxConstantParser::ParseConvertDataType(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor) {
  int64_t data_type = tensor_proto.data_type();
  ge::DataType type = ge::OnnxUtil::ConvertOnnxDataType(data_type);
  if (type == ge::DataType::DT_UNDEFINED) {
    GELOGE(domi::PARAM_INVALID, "tensor_proto date type %ld is undefined.", data_type);
    return FAILED;
  }

  TensorDesc tensor_desc = tensor.GetTensorDesc();
  tensor_desc.SetDataType(ge::DataType(type));
  tensor.SetTensorDesc(tensor_desc);
  return SUCCESS;
}

Status OnnxConstantParser::ParseConstFromInput(const ge::onnx::NodeProto *op_src, ge::Operator &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);

  // Get const Tensor from node
  Tensor tensor;
  for (auto it : node->attribute()) {
    if (it.name() != ge::kAttrNameValue) {
      continue;
    }
    const ::ge::onnx::TensorProto it_tensor = it.t();
    if (ParseConvertDataType(it_tensor, tensor) != SUCCESS) {
      GELOGE(FAILED, "Convert ge tensor date type failed, attribute name is %s.", it.name().c_str());
      return FAILED;
    }

    if (ParseConvertTensor(it_tensor, tensor) != SUCCESS) {
      GELOGE(FAILED, "Convert ge tensor shape and format failed, attribute name is %s.", it.name().c_str());
      return FAILED;
    }
  }

  op_def.SetAttr(ge::kAttrNameValue, tensor);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_def);
  op_def.UpdateOutputDesc(op_desc->GetOutputNameByIndex(0), tensor.GetTensorDesc());

  return SUCCESS;
}

Status OnnxConstantParser::ParseParams(const Message *op_src, ge::Operator &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("Onnx op node name = %s, op type= %s, parse params", node->name().c_str(), node->op_type().c_str());

  if (ParseConstFromInput(node, op_def) != SUCCESS) {
    GELOGE(FAILED, "Parse constant node %s failed", node->name().c_str());
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(ONNX, CONSTANT, OnnxConstantParser);
}  // namespace ge
