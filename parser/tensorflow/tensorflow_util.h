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

#ifndef OMG_PARSER_TENSORFLOW_TENSORFLOW_UTIL_H_
#define OMG_PARSER_TENSORFLOW_TENSORFLOW_UTIL_H_

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "parser/common/op_def/operator.h"
#include "external/graph/attr_value.h"
#include "external/graph/graph.h"
#include "external/graph/operator.h"
#include "framework/omg/parser/parser_types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "proto/tensorflow/graph.pb.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::FunctionDef;
using domi::tensorflow::AttrValue_ListValue;
using domi::tensorflow::FunctionDefLibrary;

namespace ge {
/***************************TensorFlow attribute type, constant definition*******************************************/
extern const std::string TENSORFLOW_ATTR_TYPE_STRING;
extern const std::string TENSORFLOW_ATTR_TYPE_INT;
extern const std::string TENSORFLOW_ATTR_TYPE_FLOAT;
extern const std::string TENSORFLOW_ATTR_TYPE_BOOL;
extern const std::string TENSORFLOW_ATTR_TYPE_TYPE;
extern const std::string TENSORFLOW_ATTR_TYPE_SHAPE;
extern const std::string TENSORFLOW_ATTR_TYPE_TENSOR;
extern const std::string TENSORFLOW_ATTR_TYPE_FUNC;

extern const std::string TENSORFLOW_ATTR_LIST_TYPE_STRING;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_INT;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_FLOAT;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_BOOL;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_TYPE;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_SHAPE;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_TENSOR;
extern const std::string TENSORFLOW_ATTR_LIST_TYPE_FUNC;

/***************************constant definition*******************************************/
extern const std::string TENSORFLOW_ATTR_OUTPUT_OP;

extern const std::string TENSORFLOW_ATTR_T;
extern const std::string TENSORFLOW_ATTR_N;
extern const std::string TENSORFLOW_ATTR_DATA_FORMAT;
extern const std::string TENSORFLOW_ATTR_PADDING;
extern const std::string TENSORFLOW_ATTR_KSIZE;
extern const std::string TENSORFLOW_ATTR_STRIDES;
extern const std::string TENSORFLOW_ATTR_DILATIONS;
extern const std::string TENSORFLOW_ATTR_DTYPE;
extern const std::string TENSORFLOW_ATTR_VALUE;
extern const std::string TENSORFLOW_ATTR_TRANSINPUT;
extern const std::string TENSORFLOW_ATTR_TRANSWEIGHT;
extern const std::string TENSORFLOW_ATTR_SHAPE;
extern const std::string TENSORFLOW_ATTR_TIDX;
extern const std::string TENSORFLOW_ATTR_TPADDINGS;
extern const std::string TENSORFLOW_ATTR_TMULTIPLES;
extern const std::string TENSORFLOW_ATTR_TINDICES;
extern const std::string TENSORFLOW_ATTR_TPARAMS;
extern const std::string TENSORFLOW_ATTR_TAXIS;
extern const std::string TENSORFLOW_ATTR_DSTT;
extern const std::string TENSORFLOW_ATTR_SRCT;
extern const std::string TENSORFLOW_ATTR_PERM;
extern const std::string TENSORFLOW_ATTR_INDEX;
extern const std::string TENSORFLOW_ATTR_TSHAPE;
extern const std::string TENSORFLOW_ATTR_AXIS;
extern const std::string TENSORFLOW_ATTR_BIAS;
extern const std::string TENSORFLOW_ATTR_DEPTH_RADIUS;
extern const std::string TENSORFLOW_ATTR_ALPHA;
extern const std::string TENSORFLOW_ATTR_BETA;
extern const std::string TENSORFLOW_ATTR_MODE;

// op:Const
extern const std::string TENSORFLOWF_NODE_OP_CONST;
extern const std::string TENSORFLOWF_NODE_OP_IDENTITY;
extern const std::string TENSORFLOWF_NODE_OP_SWITCH;
extern const std::string TENSORFLOWF_NODE_OP_PLACEHOLDER;
extern const std::string TENSORFLOWF_NODE_OP_ADDN;
extern const std::string TENSORFLOWF_NODE_OP_MATMUL;
extern const std::string TENSORFLOWF_NODE_OP_RELU;
extern const std::string TENSORFLOWF_NODE_OP_SHAPE;
extern const std::string TENSORFLOWF_NODE_OP_TRANSPOSE;
extern const std::string TENSORFLOWF_NODE_OP_MERGE;

// data_format
extern const std::string TENSORFLOWF_TENSOR_NCHW;
extern const std::string TENSORFLOWF_TENSOR_NHWC;

extern const int TENSORFLOW_CONV_STRIDE_NUM;
extern const int TENSORFLOW_CONV_DILATION_NUM;

// padding
extern const std::string TENSORFLOWF_OP_PADDING_VALID;
extern const std::string TENSORFLOWF_OP_PADDING_SAME;

// normal input size
extern const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_MATMUL;
extern const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_RESHAPE;
extern const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_POOL;

// normal weight size
extern const uint32_t TENSORFLOW_NORMAL_WEIGHT_SIZE_MATMUL;
extern const uint32_t TENSORFLOW_NORMAL_WEIGHT_SIZE_RESHAPE;

// input or output
extern const uint32_t TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG;
extern const uint32_t TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG;

class TensorFlowUtil {
 public:
  /**
  * @ingroup domi_omg
  * @brief find the corresponding AttrValue in NodeDef
  * @param [in] nodeDef      nodedef object to find
  * @param [in] attr_name    attribute name
  * @param [out] attr_value  attribute value
  * @return true             attribute exists
  * @return false            attribute does not exist
  *
  */
  static bool FindAttrValue(const domi::tensorflow::NodeDef *const node_def, const std::string &attr_name,
                            domi::tensorflow::AttrValue &attr_value);

  /**
  * @ingroup domi_omg
  * @brief Check the actual type and expected type of the AttrValue, int, float, list (int), list (bool), etc.
  * @param [in]         attr_value  attrValue to check
  * @param [in]         type  expected attribute type
  * @return SUCCESS     success
  * @return FAILED      failed
  *
  */
  static domi::Status CheckAttrHasType(const domi::tensorflow::AttrValue &attr_value, const std::string &type);

  /**
   * @ingroup domi_omg
   * @brief  parsing data types
   * @param [in] node_src      node to be parsed
   * @param [in] attr_src      attribute to be parsed
   * @param [out] data_type    parsed data type
   * @return SUCCESS           Parsing success
   * @return FAILED            parsing failed
   *
   */
  static domi::Status ParseDataType(const NodeDef *node_src,
                                    const std::string &attr_src,
                                    domi::tensorflow::DataType &data_type);

  /**
   * @ingroup domi_omg
   * @brief  parsing data types
   * @param [in] attr_value    attr in NodeDef to be converted
   * @param [out] op           the parsed information is stored in the properties of the parent class
   * @return SUCCESS           conversion success
   * @return FAILED            conversion failed
   *
   */
  static domi::Status TransTensorDescriptor(const domi::tensorflow::AttrValue &attr_value,
                                            ParserOperator *const op,
                                            const uint32_t io,
                                            const std::string &type = "");
  /*
  * @brief 添加NodeDef属性
   * @param [in] attr_name  attribute name
   * @param [in] attr_value  attribute Value Object
   * @param [out] node_def
   * @return void
   *
   */
  static void AddNodeAttr(const std::string &attr_name,
                          const domi::tensorflow::AttrValue &value,
                          domi::tensorflow::NodeDef *const node_def);

  static domi::Status ClearUnusedParam(ge::ComputeGraphPtr &graph);

  static bool ParseFromAttrValueList(ge::GeTensorDesc &ge_desc,
                                     const domi::tensorflow::AttrValue_ListValue &a_list,
                                     int32_t i,
                                     int32_t &tf_datatype);
};
}  // namespace ge
#endif  // OMG_PARSER_TENSORFLOW_TENSORFLOW_UTIL_H_
