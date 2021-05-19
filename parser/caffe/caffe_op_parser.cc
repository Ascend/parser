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

#include "parser/caffe/caffe_op_parser.h"
#include <memory>
#include "parser/common/op_parser_factory.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/omg/parser/parser_types.h"

using namespace ge::parser;

using domi::CAFFE;

namespace ge {
Status CaffeOpParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) { return SUCCESS; }

Status CaffeOpParser::ParseWeights(const Message *op_src, ge::NodePtr &node) { return SUCCESS; }

Status CaffeOpParser::AddConstInput(ge::NodePtr &node) { return SUCCESS; }

void CaffeOpParser::ConvertShape(const BlobProto &proto, std::vector<int64_t> &shape) {
  shape.clear();

  if (proto.has_num() || proto.has_channels() || proto.has_height() || proto.has_width()) {
    // Compatible with old formats, shape description: (num, channels, height, width)
    shape.push_back(proto.num());
    shape.push_back(proto.channels());
    shape.push_back(proto.height());
    shape.push_back(proto.width());
  } else {
    // The shape of the new format is described with "repeated Int64 dim"
    for (int i = 0; i < proto.shape().dim_size(); ++i) {
      shape.push_back(proto.shape().dim(i));
    }
  }
}

Status CaffeOpParser::ConvertWeight(const BlobProto &proto, const string &lay_name, ge::GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(weight);
  std::vector<int64_t> shape_vec;
  ConvertShape(proto, shape_vec);
  ge::GeShape shape(shape_vec);
  // Calculate the number of data in weight
  int count = 1;
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    int dim = shape.GetDim(i);
    if (dim <= 0) {
      REPORT_INNER_ERROR("E19999", "Convert weight fail, dim:%d of layer:%s <=0, check invalid", dim, lay_name.c_str());
      GELOGE(FAILED, "[Check][Size]Convert weight fail, dim:%d of layer:%s <=0, check invalid", dim, lay_name.c_str());
      return FAILED;
    }

    if (dim >= INT64_MAX / count) {
      REPORT_INNER_ERROR("E19999", "Convert weight fail, shape:%s of layer:%s will overflow after multi",
                         shape.ToString().c_str(), lay_name.c_str());
      GELOGE(FAILED, "[Check][Size]Convert weight fail, Blob size exceeds INT64_MAX, dim:%d, count:%d, layer name:%s",
             dim, count, lay_name.c_str());
      return FAILED;
    }

    count *= dim;
  }
  return ParseWeightType(proto, shape, count, lay_name, weight);
}

Status CaffeOpParser::ParseWeightType(const BlobProto &proto, const ge::GeShape &shape, int size,
                                      const string &lay_name, ge::GeTensorPtr &weight) {
  // Extract weight data and store it in weightdef by float type
  GE_CHECK_NOTNULL(weight);
  ge::DataType dtype = ge::DT_FLOAT;
  if (proto.double_data_size() > 0) {
    // Convert by double type
    if (size != proto.double_data_size()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11033", {"opname", "blobsize", "reason"},
                                                      {lay_name, std::to_string(proto.double_data_size()),
                                                       "it does not match shape size[" + std::to_string(size) + "]"});
      GELOGE(FAILED, "[Check][Param]Convert weight fail, Blob size does not match shape size, "
             "shape size:%d, blob size:%d, layer name:%s", size, proto.double_data_size(), lay_name.c_str());
      return FAILED;
    }
    std::unique_ptr<float[]> buf(new (std::nothrow) float[size]());
    GE_CHECK_NOTNULL(buf);
    for (int i = 0; i < size; ++i) {
      buf[i] = proto.double_data(i);
    }
    GE_IF_BOOL_EXEC(weight->SetData(reinterpret_cast<uint8_t *>(buf.get()), size * sizeof(float)) != ge::GRAPH_SUCCESS,
                    GELOGW("SetData failed for GeTensor."););  // no need to return
  } else if (proto.int8_data().length() > 0) {
    if (size != static_cast<int>(proto.int8_data().length())) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11033", {"opname", "blobsize", "reason"},
                                                      {lay_name, std::to_string(proto.int8_data().length()),
                                                       "it does not match shape size[" + std::to_string(size) + "]"});
      GELOGE(FAILED, "[Check][Param]Convert weight failed, Blob size does not match shape size, "
             "shape size:%d, blob size:%ld, layer name:%s", size, proto.int8_data().length(), lay_name.c_str());
      return FAILED;
    }
    const char *data_ptr = proto.int8_data().data();
    GE_CHECK_NOTNULL(data_ptr);
    GE_IF_BOOL_EXEC(
      weight->SetData(reinterpret_cast<const uint8_t *>(data_ptr), size * sizeof(int8_t)) != ge::GRAPH_SUCCESS,
      GELOGW("SetData failed for GeTensor."););  // no need to return
    dtype = ge::DT_INT8;
  } else if (proto.int32_data_size() > 0) {
    if (size != proto.int32_data_size()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11033", {"opname", "blobsize", "reason"},
                                                      {lay_name, std::to_string(proto.int32_data_size()),
                                                       "it does not match shape size[" + std::to_string(size) + "]"});
      GELOGE(FAILED, "[Check][Param]Convert weight failed, Blob size does not match shape size, "
             "shape size:%d, blob size:%d, layer name:%s", size, proto.int32_data_size(), lay_name.c_str());
      return FAILED;
    }
    std::unique_ptr<int32_t[]> int32_weight_buf(new (std::nothrow) int32_t[size]());
    GE_CHECK_NOTNULL(int32_weight_buf);
    for (int i = 0; i < size; ++i) {
      int32_weight_buf[i] = proto.int32_data(i);
    }
    GE_IF_BOOL_EXEC(
      weight->SetData(reinterpret_cast<uint8_t *>(int32_weight_buf.get()), size * sizeof(int32_t)) != ge::GRAPH_SUCCESS,
      GELOGW("SetData failed for GeTensor."););  // no need to return
    dtype = ge::DT_INT32;
  } else if (proto.uint64_data_size() > 0) {
    if (size != proto.uint64_data_size()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11033", {"opname", "blobsize", "reason"},
                                                      {lay_name, std::to_string(proto.uint64_data_size()),
                                                       "it does not match shape size[" + std::to_string(size) + "]"});
      GELOGE(FAILED, "[Check][Param]Convert weight failed, Blob size does not match shape size, "
             "shape size:%d, blob size:%d, layer name:%s", size, proto.uint64_data_size(), lay_name.c_str());
      return FAILED;
    }
    std::unique_ptr<uint64_t[]> uint64_weight_buf(new (std::nothrow) uint64_t[size]());
    GE_CHECK_NOTNULL(uint64_weight_buf);
    for (int i = 0; i < size; ++i) {
      uint64_weight_buf[i] = proto.uint64_data(i);
    }
    GE_IF_BOOL_EXEC(weight->SetData(reinterpret_cast<uint8_t *>(uint64_weight_buf.get()), size * sizeof(uint64_t)) !=
                      ge::GRAPH_SUCCESS,
                    GELOGW("SetData failed for GeTensor."););  // no need to return
    dtype = ge::DT_UINT64;
  } else {
    // Convert by float type
    if (size != proto.data_size()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11033", {"opname", "blobsize", "reason"},
                                                      {lay_name, std::to_string(proto.data_size()),
                                                       "it does not match shape size[" + std::to_string(size) + "]"});
      GELOGE(FAILED, "[Check][Param]Convert weight fail, Blob size does not match shape size, "
             "shape size:%d, blob.data_size:%d, layer name:%s", size, proto.data_size(), lay_name.c_str());
      return FAILED;
    }
    const float *data_ptr = proto.data().data();
    GE_CHECK_NOTNULL(data_ptr);
    GE_IF_BOOL_EXEC(
      weight->SetData(reinterpret_cast<const uint8_t *>(data_ptr), size * sizeof(float)) != ge::GRAPH_SUCCESS,
      GELOGW("SetData failed for GeTensor."););  // no need to return
  }
  ge::GeTensorDesc weight_desc = ge::GeTensorDesc();
  weight_desc.Update(shape, ge::FORMAT_NCHW, dtype);
  weight->SetTensorDesc(weight_desc);
  return SUCCESS;
}

// Dropout's corresponding op_parser is registered as caffeopparser, optimized in optimization stage.
REGISTER_OP_PARSER_CREATOR(CAFFE, DROPOUT, CaffeOpParser);

// A new operator added by framework in OM model is used to
// collect and arrange all outputs in the order of the original model's output
// Net output operator does not need special processing in the parse stage,
// and directly registers in the op_parser file
REGISTER_OP_PARSER_CREATOR(CAFFE, NETOUTPUT, CaffeOpParser);
}  // namespace ge
