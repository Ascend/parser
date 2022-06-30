/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "onnx_file_constant_parser.h"
#include <vector>

#include "graph/ge_tensor.h"
#include "parser/common/op_parser_factory.h"
#include "parser/onnx/onnx_util.h"
#include "framework/common/util.h"
#include "framework/common/types.h"

using ge::onnx::NodeProto;
using ge::onnx::TensorProto;
using domi::ONNX;
using GeShape = ge::GeShape;
using GeTensorDesc = ge::GeTensorDesc;
using namespace ge::parser;

namespace {
const std::string kAttrShape = "shape";
const std::string kAttrDataType = "dtype";
const std::string kFileConstantPath = "file_constant_path";
const std::string kLocation = "location";
const std::string kOffset = "offset";
const int64_t kOffsetCoefficient = 4096;
const char *const kFileConstant = "FileConstant";
}
namespace ge {
Status OnnxFileConstantParser::ParseParams(const Message *op_src, ge::Operator &op_def) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  GELOGD("Onnx op node name = %s, op type= %s, parse params", node->name().c_str(), node->op_type().c_str());

  ge::onnx::TensorProto tensor_proto;
  if (GetTensorProto(node, tensor_proto) != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "node[%s] get tensor failed", node->name().c_str());
    GELOGE(domi::PARAM_INVALID, "[Get][TensorProto] node[%s] get tensor failed", node->name().c_str());
    return FAILED;
  }
  if (ParseDataType(tensor_proto, op_def) != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "node[%s] parse data type failed", node->name().c_str());
    GELOGE(domi::PARAM_INVALID, "[Parse][Shape] node[%s] parse data type failed", node->name().c_str());
    return FAILED;
  }
  if (ParsePath(tensor_proto, op_def) != SUCCESS) {
    REPORT_INNER_ERROR("E19999", "node[%s] parse file path failed", node->name().c_str());
    GELOGE(domi::PARAM_INVALID, "[Parse][Shape] node[%s] parse file path failed", node->name().c_str());
    return FAILED;
  }
  ParseShape(tensor_proto, op_def);
  return SUCCESS;
}

Status OnnxFileConstantParser::GetTensorProto(const ge::onnx::NodeProto *node_proto,
                                              ge::onnx::TensorProto &tensor_proto) {
  for (const auto &it : node_proto->attribute()) {
    if (it.name() != ge::kAttrNameValue) {
      continue;
    }
    tensor_proto = it.t();
    return SUCCESS;
  }
  REPORT_INNER_ERROR("E19999", "node_proto[%s] get value failed", node_proto->name().c_str());
  GELOGE(ge::PARAM_INVALID, "[Get][TensorProto] node_proto[%s] get value failed", node_proto->name().c_str());
  return FAILED;
}

void OnnxFileConstantParser::ParseShape(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) {
  std::vector<int64_t> tmp_shape;
  for (int i = 0; i < tensor_proto.dims_size(); i++) {
    tmp_shape.push_back(tensor_proto.dims(i));
  }
  op_def.SetAttr(kAttrShape.c_str(), tmp_shape);
}

Status OnnxFileConstantParser::ParseDataType(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) {
  int64_t data_type = tensor_proto.data_type();
  ge::DataType type = ge::OnnxUtil::ConvertOnnxDataType(data_type);
  if (type >= ge::DataType::DT_UNDEFINED) {
    REPORT_INNER_ERROR("E19999", "tensor_proto date type %ld is undefined.", data_type);
    GELOGE(domi::PARAM_INVALID, "[Check][Param] tensor_proto date type %ld is undefined.", data_type);
    return FAILED;
  }

  op_def.SetAttr(kAttrDataType.c_str(), type);
  return SUCCESS;
}

Status OnnxFileConstantParser::ParsePath(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) {
  ge::NamedAttrs attrs;
  for (int32_t i = 0; i < tensor_proto.external_data_size(); ++i) {
    const ge::onnx::StringStringEntryProto &string_proto = tensor_proto.external_data(i);
    if (SetPathAttr(string_proto, attrs) != SUCCESS) {
      REPORT_INNER_ERROR("E19999", "external tensor proto[%s] parse attrs failed.", tensor_proto.name().c_str());
      GELOGE(domi::PARAM_INVALID, "external tensor proto[%s] parse attrs failed.", tensor_proto.name().c_str());
      return FAILED;
    }
  }

  if (!attrs.HasAttr(kLocation)) {
    REPORT_INNER_ERROR("E19999", "external tensor proto[%s] must contain location.", tensor_proto.name().c_str());
    GELOGE(domi::PARAM_INVALID, "external tensor proto[%s] must contain location.", tensor_proto.name().c_str());
    return FAILED;
  }
  op_def.SetAttr(kFileConstantPath.c_str(), attrs);
  return SUCCESS;
}

Status OnnxFileConstantParser::SetPathAttr(const ge::onnx::StringStringEntryProto &string_proto,
                                           ge::NamedAttrs &attrs) {
  if (string_proto.key() == kLocation) {
    AttrUtils::SetStr(attrs, kLocation, string_proto.value());
  } else {
    int64_t value;
    try {
      value = stol(string_proto.value());
    } catch (const std::exception &e) {
      REPORT_INNER_ERROR("E19999", "Convert %s to int64_t value failed:%s", string_proto.value().c_str(), e.what());
      GELOGE(domi::PARAM_INVALID, "Convert %s to int64_t value failed:%s", string_proto.value().c_str(), e.what());
      return FAILED;
    }
    if (string_proto.key() == kOffset) {
      if (std::numeric_limits<int64_t>::max() / kOffsetCoefficient < value) {
        REPORT_INNER_ERROR("E19999", "overflow, kOffsetCoefficient[%ld], value[%ld]", kOffsetCoefficient, value);
        GELOGE(domi::PARAM_INVALID, "overflow, kOffsetCoefficient[%ld], value[%ld]", kOffsetCoefficient, value);
        return FAILED;
      }
      value *= kOffsetCoefficient;
    }
    AttrUtils::SetInt(attrs, string_proto.key(), value);
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(ONNX, kFileConstant, OnnxFileConstantParser);
}  // namespace ge