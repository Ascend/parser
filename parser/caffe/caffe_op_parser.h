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

#ifndef PARSER_CAFFE_CAFFE_OP_PARSER_H_
#define PARSER_CAFFE_CAFFE_OP_PARSER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "common/util.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/parser/op_parser.h"
#include "proto/caffe/caffe.pb.h"

using domi::caffe::ArgMaxParameter;
using domi::caffe::BatchNormParameter;
using domi::caffe::BlobProto;
using domi::caffe::BlobShape;
using domi::caffe::ConcatParameter;
using domi::caffe::ConvolutionParameter;
using domi::caffe::DetectionOutputParameter;
using domi::caffe::EltwiseParameter;
using domi::caffe::FillerParameter;
using domi::caffe::InnerProductParameter;
using domi::caffe::LayerParameter;
using domi::caffe::PoolingParameter;
using domi::caffe::PReLUParameter;
using domi::caffe::ReshapeParameter;
using domi::caffe::ROIAlignParameter;
using domi::caffe::TanHParameter;
using domi::caffe::UpsampleParameter;

namespace ge {
/**
 * @ingroup ge_omg
 * @brief Used to parse Caffe operator information
 */
class PARSER_FUNC_VISIBILITY CaffeOpParser : public OpParser {
 public:
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

  Status ParseParams(const Message *op_src, ge::Operator &op_dest) override {
    return domi::SUCCESS;
  }

  /**
   * @ingroup ge_omg
   * @brief parse weight information of the operation
   * @param [in] op_src Weight data to be parsed
   * @param [out] graph Weight data after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
  Status ParseWeights(const Message *op_src, ge::NodePtr &node) override;

  /**
   * @ingroup ge_omg
   * @brief add const input node
   * @param [in] node to add const input
   * @param [out] node after add const input
   * @return SUCCESS add const input successfully
   * @return FAILED add const input failed
   * @author
   */
  virtual Status AddConstInput(ge::NodePtr &node);

 protected:
  /**
   * @ingroup ge_omg
   * @brief Convert blob proto to weight definition
   * @param [in] proto Weight data to be parsed
   * @param [out] weight Weight data after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  static Status ConvertWeight(const BlobProto &proto, const string &lay_name, ge::GeTensorPtr &weight);

  /**
   * @ingroup ge_omg
   * @brief Convert blob proto to shape definition
   * @param [in] proto Shape information before conversion
   * @param [out] shape Save converted shape information
   */
  static void ConvertShape(const BlobProto &proto, std::vector<int64_t> &shape);

 private:
  /**
   * @ingroup ge_omg
   * @brief Convert blob proto to weight definition
   * @param [in] proto Weight data to be parsed
   * @param [out] weight Weight data after parsing
   * @return SUCCESS parse weight type successfully
   * @return FAILED parse failed
   */
  static Status ParseWeightType(const BlobProto &proto, const ge::GeShape &shape,
                                int size, const string &lay_name, ge::GeTensorPtr &weight);
};
}  // namespace ge

#endif  // PARSER_CAFFE_CAFFE_OP_PARSER_H_
