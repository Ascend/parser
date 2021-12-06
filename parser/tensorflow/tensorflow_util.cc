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
/***************************TensorFlow attribute type, constant definition*******************************************/
const std::string TENSORFLOW_ATTR_TYPE_STRING = "string";
const std::string TENSORFLOW_ATTR_TYPE_INT = "int";
const std::string TENSORFLOW_ATTR_TYPE_FLOAT = "float";
const std::string TENSORFLOW_ATTR_TYPE_BOOL = "bool";
const std::string TENSORFLOW_ATTR_TYPE_TYPE = "type";
const std::string TENSORFLOW_ATTR_TYPE_SHAPE = "shape";
const std::string TENSORFLOW_ATTR_TYPE_TENSOR = "tensor";
const std::string TENSORFLOW_ATTR_TYPE_FUNC = "func";

const std::string TENSORFLOW_ATTR_LIST_TYPE_STRING = "list(string)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_INT = "list(int)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_FLOAT = "list(float)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_BOOL = "list(bool)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_TYPE = "list(type)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_SHAPE = "list(shape)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_TENSOR = "list(tensor)";
const std::string TENSORFLOW_ATTR_LIST_TYPE_FUNC = "list(func)";

/***************************constant definition*******************************************/
const std::string TENSORFLOW_ATTR_OUTPUT_OP = "output_op";

const std::string TENSORFLOW_ATTR_T = "T";
const std::string TENSORFLOW_ATTR_N = "N";
const std::string TENSORFLOW_ATTR_DATA_FORMAT = "data_format";
const std::string TENSORFLOW_ATTR_PADDING = "padding";
const std::string TENSORFLOW_ATTR_KSIZE = "ksize";
const std::string TENSORFLOW_ATTR_STRIDES = "strides";
const std::string TENSORFLOW_ATTR_DILATIONS = "dilations";
const std::string TENSORFLOW_ATTR_DTYPE = "dtype";
const std::string TENSORFLOW_ATTR_VALUE = "value";
const std::string TENSORFLOW_ATTR_TRANSINPUT = "transpose_a";
const std::string TENSORFLOW_ATTR_TRANSWEIGHT = "transpose_b";
const std::string TENSORFLOW_ATTR_SHAPE = "shape";
const std::string TENSORFLOW_ATTR_TIDX = "Tidx";
const std::string TENSORFLOW_ATTR_TPADDINGS = "Tpaddings";
const std::string TENSORFLOW_ATTR_TMULTIPLES = "Tmultiples";
const std::string TENSORFLOW_ATTR_TINDICES = "Tindices";
const std::string TENSORFLOW_ATTR_TPARAMS = "Tparams";
const std::string TENSORFLOW_ATTR_TAXIS = "Taxis";
const std::string TENSORFLOW_ATTR_DSTT = "DstT";
const std::string TENSORFLOW_ATTR_SRCT = "SrcT";
const std::string TENSORFLOW_ATTR_PERM = "perm";
const std::string TENSORFLOW_ATTR_INDEX = "Index";
const std::string TENSORFLOW_ATTR_TSHAPE = "Tshape";
const std::string TENSORFLOW_ATTR_AXIS = "Axis";
const std::string TENSORFLOW_ATTR_BIAS = "bias";
const std::string TENSORFLOW_ATTR_DEPTH_RADIUS = "depth_radius";
const std::string TENSORFLOW_ATTR_ALPHA = "alpha";
const std::string TENSORFLOW_ATTR_BETA = "beta";
const std::string TENSORFLOW_ATTR_MODE = "mode";

// op:Const
const std::string TENSORFLOWF_NODE_OP_CONST = "Const";
const std::string TENSORFLOWF_NODE_OP_IDENTITY = "Identity";
const std::string TENSORFLOWF_NODE_OP_SWITCH = "Switch";
const std::string TENSORFLOWF_NODE_OP_PLACEHOLDER = "Placeholder";
const std::string TENSORFLOWF_NODE_OP_ADDN = "AddN";
const std::string TENSORFLOWF_NODE_OP_MATMUL = "MatMul";
const std::string TENSORFLOWF_NODE_OP_RELU = "Relu";
const std::string TENSORFLOWF_NODE_OP_SHAPE = "Shape";
const std::string TENSORFLOWF_NODE_OP_TRANSPOSE = "Transpose";
const std::string TENSORFLOWF_NODE_OP_MERGE = "Merge";

// data_format
const std::string TENSORFLOWF_TENSOR_NCHW = "NCHW";
const std::string TENSORFLOWF_TENSOR_NHWC = "NHWC";

const int TENSORFLOW_CONV_STRIDE_NUM = 4;
const int TENSORFLOW_CONV_DILATION_NUM = 4;

// padding
const std::string TENSORFLOWF_OP_PADDING_VALID = "VALID";
const std::string TENSORFLOWF_OP_PADDING_SAME = "SAME";

// normal input size
const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_MATMUL = 2;
const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_RESHAPE = 1;
const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_POOL = 1;

// normal weight size
const uint32_t TENSORFLOW_NORMAL_WEIGHT_SIZE_MATMUL = 1;
const uint32_t TENSORFLOW_NORMAL_WEIGHT_SIZE_RESHAPE = 1;

// input or output
const uint32_t TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG = 1;
const uint32_t TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG = 2;

using AttrValueMap = ::google::protobuf::Map<std::string, domi::tensorflow::AttrValue>;
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool TensorFlowUtil::FindAttrValue(
    const domi::tensorflow::NodeDef *const node_def, const std::string &attr_name,
    domi::tensorflow::AttrValue &attr_value) {
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
    const domi::tensorflow::AttrValue &attr_value, const std::string &type) {
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
      if (type != (type_string)) {                                                                                     \
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
    const NodeDef *node_src, const std::string &attr_src, domi::tensorflow::DataType &data_type) {
  GE_CHECK_NOTNULL(node_src);

  std::string node_name = node_src->name();

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
  std::vector<int64_t> data_dim;
  for (int j = 0; j < shape_dim_dim; j++) {
    data_dim.push_back(a_list.func(i).attr().at(SERIALIZE_SHAPE).list().i(j));
  }
  ge_desc.SetShape(ge::GeShape(data_dim));
  ge_desc.SetOriginShape(ge::GeShape(data_dim));
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY domi::Status TensorFlowUtil::TransTensorDescriptor(
    const domi::tensorflow::AttrValue &attr_value, ParserOperator *const op,
    const uint32_t io, const std::string &type) {
  GE_CHECK_NOTNULL(op);
  if (!attr_value.has_list()) {
    return PARAM_INVALID;
  }
  std::vector<int32_t> tf_in_type;
  std::vector<int32_t> tf_out_type;
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
      GE_IF_BOOL_EXEC((type == ge::parser::MEAN) && (tmp_dim == 0), std::vector<int64_t> data_dim = {tmp_dim};
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
    const std::string &attr_name, const domi::tensorflow::AttrValue &value, domi::tensorflow::NodeDef *const node_def) {
  GE_CHK_BOOL_TRUE_EXEC_INFO(node_def == nullptr, return, "input parameter is null.");
  node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_name, value));
}
}  // namespace ge
