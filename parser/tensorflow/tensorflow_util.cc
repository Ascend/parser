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

#include "parser/tensorflow/tensorflow_util.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "common/string_util.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/utils/type_utils.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

using domi::tensorflow::DT_INVALID;

namespace ge {
using AttrValueMap = ::google::protobuf::Map<string, domi::tensorflow::AttrValue>;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool TensorFlowUtil::FindAttrValue(
    const domi::tensorflow::NodeDef *node_def, const string &attr_name, domi::tensorflow::AttrValue &attr_value) {
  GE_CHECK_NOTNULL(node_def);
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue> &attr = node_def->attr();
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    attr_value = it->second;
    return true;
  }

  return false;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY domi::Status TensorFlowUtil::CheckAttrHasType(
    const domi::tensorflow::AttrValue &attr_value, const string &type) {
  uint32_t num_set = 0;
#define VALIDATE_FIELD(name, type_string, oneof_case)                                                                \
  do {                                                                                                               \
    if (attr_value.has_list()) {                                                                                     \
      if (attr_value.list().name##_size() > 0) {                                                                     \
        if (type != "list(" type_string ")") {                                                                       \
          GELOGE(FAILED, "GeAttrValue had value with type 'list(" type_string ")'when '%s' expected", type.c_str()); \
          return FAILED;                                                                                             \
        }                                                                                                            \
        ++num_set;                                                                                                   \
      }                                                                                                              \
    } else if (attr_value.value_case() == domi::tensorflow::AttrValue::oneof_case) {                                 \
      if (type != type_string) {                                                                                     \
        GELOGE(FAILED, "GeAttrValue had value with type '" type_string "' when '%s' expected", type.c_str());        \
        return FAILED;                                                                                               \
      }                                                                                                              \
      ++num_set;                                                                                                     \
    }                                                                                                                \
  } while (false)

  VALIDATE_FIELD(s, "string", kS);
  VALIDATE_FIELD(i, "int", kI);
  VALIDATE_FIELD(f, "float", kF);
  VALIDATE_FIELD(b, "bool", kB);
  VALIDATE_FIELD(type, "type", kType);
  VALIDATE_FIELD(shape, "shape", kShape);
  VALIDATE_FIELD(tensor, "tensor", kTensor);
  VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

  if (attr_value.value_case() == domi::tensorflow::AttrValue::kPlaceholder) {
    GELOGE(FAILED, "GeAttrValue had value with unexpected type 'placeholder'");
    return FAILED;
  }

  // Okay to have an empty list, but not to be missing a non-list value.
  if ((num_set == 0) && (!ge::StringUtils::StartWith(type, "list("))) {
    GELOGE(FAILED, "GeAttrValue missing value with expected type '%s'", type.c_str());
    return FAILED;
  }

  // Ref types and DT_INVALID are illegal, and DataTypes must
  // be a valid enum type.
  if (type == "type") {
    if (!domi::tensorflow::DataType_IsValid(attr_value.type())) {
      GELOGE(FAILED, "GeAttrValue has invalid DataType enum: %d", attr_value.type());
      return FAILED;
    }
    if (attr_value.type() == DT_INVALID) {
      GELOGE(FAILED, "GeAttrValue has invalid DataType");
      return FAILED;
    }
  } else if (type == "list(type)") {
    for (auto &as_int : attr_value.list().type()) {
      const domi::tensorflow::DataType dtype = static_cast<domi::tensorflow::DataType>(as_int);
      if (!domi::tensorflow::DataType_IsValid(dtype)) {
        GELOGE(FAILED, "GeAttrValue has invalid DataType enum: %d", as_int);
        return FAILED;
      }
      if (dtype == DT_INVALID) {
        GELOGE(FAILED, "GeAttrValue contains invalid DataType");
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY domi::Status TensorFlowUtil::ParseDataType(
    const NodeDef *node_src, const string &attr_src, domi::tensorflow::DataType &data_type) {
  GE_CHECK_NOTNULL(node_src);

  string node_name = node_src->name();

  // Find the value of attr_src from node_src
  domi::tensorflow::AttrValue attr_value;
  GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(FindAttrValue(node_src, attr_src, attr_value),
                                        "In NodeDef %s Attr %s is not exist.", node_name.c_str(), attr_src.c_str());

  // Check whether the attr_src.value contains the type field
  GE_RETURN_WITH_LOG_IF_ERROR(CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TYPE), "check Attr %s failed",
                              attr_src.c_str());

  data_type = attr_value.type();

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool TensorFlowUtil::ParseFromAttrValueList(
    ge::GeTensorDesc &ge_desc, const domi::tensorflow::AttrValue_ListValue &a_list, int32_t i, int32_t &tf_datatype) {
  const std::string SERIALIZE_FORMAT = "serialize_format";
  const std::string SERIALIZE_DATATYPE = "serialize_datatype";
  const std::string SERIALIZE_SHAPE = "serialize_shape";

  ge_desc.SetFormat(ge::FORMAT_ND);
  ge_desc.SetOriginFormat(ge::FORMAT_ND);

  tf_datatype = a_list.func(i).attr().at(SERIALIZE_DATATYPE).i();
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tf_datatype);
  GE_CHK_BOOL_RET_STATUS(type != ge::DataType::DT_UNDEFINED, PARAM_INVALID,
                         "In FrameworkOp translate datatype:%d failed, domi cann't support.", tf_datatype);
  ge_desc.SetDataType(type);
  int shape_dim_dim = a_list.func(i).attr().at(SERIALIZE_SHAPE).list().i_size();
  vector<int64_t> data_dim;
  for (int j = 0; j < shape_dim_dim; j++) {
    data_dim.push_back(a_list.func(i).attr().at(SERIALIZE_SHAPE).list().i(j));
  }
  ge_desc.SetShape(ge::GeShape(data_dim));
  ge_desc.SetOriginShape(ge::GeShape(data_dim));
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY domi::Status TensorFlowUtil::TransTensorDescriptor(
    const domi::tensorflow::AttrValue &attr_value, ParserOperator *op, const uint32_t io, const string &type) {
  GE_CHECK_NOTNULL(op);
  if (!attr_value.has_list()) {
    return PARAM_INVALID;
  }

  vector<int32_t> tf_in_type;
  vector<int32_t> tf_out_type;
  // list contain many TensorDescriptors
  domi::tensorflow::AttrValue_ListValue a_list = attr_value.list();
  for (int32_t i = 0; i < a_list.func_size(); i++) {
    ge::GeTensorDesc ge_desc;
    int32_t tf_datatype = 0;
    GE_CHK_BOOL_RET_STATUS(ParseFromAttrValueList(ge_desc, a_list, i, tf_datatype), PARAM_INVALID,
                           "parse ge_desc failed.");

    uint32_t size_type = 1;
    int64_t tmp_dim = 0;

    auto data_type = ge_desc.GetDataType();
    GE_CHK_BOOL_RET_STATUS(ge::TypeUtils::GetDataTypeLength(data_type, size_type), PARAM_INVALID,
                           "dataType no define size , parse ge_desc failed.");
    // get size
    for (uint32_t j = 0; j < ge_desc.GetShape().GetDimNum(); ++j) {
      tmp_dim = ge_desc.GetShape().GetDim(j);

      // The shape infered by fusedbatchnormgrad and mean calling tensorflow is not accurate.
      // Here, special treatment is given to the two operators.
      // Adjust shape to fit resnet50 network only.
      GE_IF_BOOL_EXEC((type == ge::parser::FUSEDBATCHNORMGRAD) && (tmp_dim == 0), ge_desc.SetShape(ge::GeShape());
                      break;);
      GE_IF_BOOL_EXEC((type == ge::parser::MEAN) && (tmp_dim == 0), vector<int64_t> data_dim = {tmp_dim};
                      ge_desc.SetShape(ge::GeShape(data_dim)); break;);
    }
    ge::TensorUtils::SetRealDimCnt(ge_desc, ge_desc.GetShape().GetDimNum());

    GELOGD("IO:%d: after translate tf_desc, datatype: %s, format: %s, size_type: %u", io,
           ge::TypeUtils::DataTypeToSerialString(ge_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(ge_desc.GetFormat()).c_str(), size_type);

    if (io == TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG) {
      op->InputTensorDesc(ge_desc);
      tf_in_type.push_back(tf_datatype);
    } else if (io == TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG) {
      op->OutputTensorDesc(ge_desc);
      tf_out_type.push_back(tf_datatype);
    }
  }
  op->AttrVector(ge::T_IN_DATATYPE, tf_in_type);
  op->AttrVector(ge::T_OUT_DATATYPE, tf_out_type);
  return SUCCESS;
}
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void TensorFlowUtil::AddNodeAttr(
    const string &attr_name, const domi::tensorflow::AttrValue &value, domi::tensorflow::NodeDef *node_def) {
  GE_CHK_BOOL_TRUE_EXEC_INFO(node_def == nullptr, return, "input parameter is null.");
  node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_name, value));
}
}  // namespace ge
