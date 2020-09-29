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
using std::string;
using std::vector;
using domi::tensorflow::NodeDef;
using domi::tensorflow::FunctionDef;
using domi::tensorflow::AttrValue_ListValue;
using domi::tensorflow::FunctionDefLibrary;

namespace ge {
/***************************TensorFlow attribute type, constant definition*******************************************/
static const string TENSORFLOW_ATTR_TYPE_STRING = "string";
static const string TENSORFLOW_ATTR_TYPE_INT = "int";
static const string TENSORFLOW_ATTR_TYPE_FLOAT = "float";
static const string TENSORFLOW_ATTR_TYPE_BOOL = "bool";
static const string TENSORFLOW_ATTR_TYPE_TYPE = "type";
static const string TENSORFLOW_ATTR_TYPE_SHAPE = "shape";
static const string TENSORFLOW_ATTR_TYPE_TENSOR = "tensor";
static const string TENSORFLOW_ATTR_TYPE_FUNC = "func";

static const string TENSORFLOW_ATTR_LIST_TYPE_STRING = "list(string)";
static const string TENSORFLOW_ATTR_LIST_TYPE_INT = "list(int)";
static const string TENSORFLOW_ATTR_LIST_TYPE_FLOAT = "list(float)";
static const string TENSORFLOW_ATTR_LIST_TYPE_BOOL = "list(bool)";
static const string TENSORFLOW_ATTR_LIST_TYPE_TYPE = "list(type)";
static const string TENSORFLOW_ATTR_LIST_TYPE_SHAPE = "list(shape)";
static const string TENSORFLOW_ATTR_LIST_TYPE_TENSOR = "list(tensor)";
static const string TENSORFLOW_ATTR_LIST_TYPE_FUNC = "list(func)";

/***************************constant definition*******************************************/
static const string TENSORFLOW_ATTR_OUTPUT_OP = "output_op";

static const string TENSORFLOW_ATTR_T = "T";
static const string TENSORFLOW_ATTR_N = "N";
static const string TENSORFLOW_ATTR_DATA_FORMAT = "data_format";
static const string TENSORFLOW_ATTR_PADDING = "padding";
static const string TENSORFLOW_ATTR_KSIZE = "ksize";
static const string TENSORFLOW_ATTR_STRIDES = "strides";
static const string TENSORFLOW_ATTR_DILATIONS = "dilations";
static const string TENSORFLOW_ATTR_DTYPE = "dtype";
static const string TENSORFLOW_ATTR_VALUE = "value";
static const string TENSORFLOW_ATTR_TRANSINPUT = "transpose_a";
static const string TENSORFLOW_ATTR_TRANSWEIGHT = "transpose_b";
static const string TENSORFLOW_ATTR_SHAPE = "shape";
static const string TENSORFLOW_ATTR_TIDX = "Tidx";
static const string TENSORFLOW_ATTR_TPADDINGS = "Tpaddings";
static const string TENSORFLOW_ATTR_TMULTIPLES = "Tmultiples";
static const string TENSORFLOW_ATTR_TINDICES = "Tindices";
static const string TENSORFLOW_ATTR_TPARAMS = "Tparams";
static const string TENSORFLOW_ATTR_TAXIS = "Taxis";
static const string TENSORFLOW_ATTR_DSTT = "DstT";
static const string TENSORFLOW_ATTR_SRCT = "SrcT";
static const string TENSORFLOW_ATTR_PERM = "perm";
static const string TENSORFLOW_ATTR_INDEX = "Index";
static const string TENSORFLOW_ATTR_TSHAPE = "Tshape";
static const string TENSORFLOW_ATTR_AXIS = "Axis";
static const string TENSORFLOW_ATTR_BIAS = "bias";
static const string TENSORFLOW_ATTR_DEPTH_RADIUS = "depth_radius";
static const string TENSORFLOW_ATTR_ALPHA = "alpha";
static const string TENSORFLOW_ATTR_BETA = "beta";
static const string TENSORFLOW_ATTR_MODE = "mode";

// op:Const
static const string TENSORFLOWF_NODE_OP_CONST = "Const";
static const string TENSORFLOWF_NODE_OP_IDENTITY = "Identity";
static const string TENSORFLOWF_NODE_OP_SWITCH = "Switch";
static const string TENSORFLOWF_NODE_OP_PLACEHOLDER = "Placeholder";
static const string TENSORFLOWF_NODE_OP_ADDN = "AddN";
static const string TENSORFLOWF_NODE_OP_MATMUL = "MatMul";
static const string TENSORFLOWF_NODE_OP_RELU = "Relu";
static const string TENSORFLOWF_NODE_OP_SHAPE = "Shape";
static const string TENSORFLOWF_NODE_OP_TRANSPOSE = "Transpose";
static const string TENSORFLOWF_NODE_OP_MERGE = "Merge";

// data_format
static const string TENSORFLOWF_TENSOR_NCHW = "NCHW";
static const string TENSORFLOWF_TENSOR_NHWC = "NHWC";

static const int TENSORFLOW_CONV_STRIDE_NUM = 4;
static const int TENSORFLOW_CONV_DILATION_NUM = 4;

// padding
static const string TENSORFLOWF_OP_PADDING_VALID = "VALID";
static const string TENSORFLOWF_OP_PADDING_SAME = "SAME";

// normal input size
static const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_MATMUL = 2;
static const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_RESHAPE = 1;
static const uint32_t TENSORFLOW_NORMAL_INPUT_SIZE_POOL = 1;

// normal weight size
static const uint32_t TENSORFLOW_NORMAL_WEIGHT_SIZE_MATMUL = 1;
static const uint32_t TENSORFLOW_NORMAL_WEIGHT_SIZE_RESHAPE = 1;

// input or output
static const uint32_t TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG = 1;
static const uint32_t TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG = 2;

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
  static bool FindAttrValue(const domi::tensorflow::NodeDef *nodeDef, const string &attr_name,
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
  static domi::Status CheckAttrHasType(const domi::tensorflow::AttrValue &attr_value, const string &type);

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
                                    const string &attr_src,
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
                                            ParserOperator *op,
                                            const uint32_t io,
                                            const string &type = "");
  /*
  * @brief 添加NodeDef属性
   * @param [in] attr_name  attribute name
   * @param [in] attr_value  attribute Value Object
   * @param [out] node_def
   * @return void
   *
   */
  static void AddNodeAttr(const string &attr_name,
                          const domi::tensorflow::AttrValue &value,
                          domi::tensorflow::NodeDef *node_def);

  static domi::Status ClearUnusedParam(ge::ComputeGraphPtr &graph);

  static bool ParseFromAttrValueList(ge::GeTensorDesc &ge_desc,
                                     const domi::tensorflow::AttrValue_ListValue &a_list,
                                     int32_t i,
                                     int32_t &tf_datatype);
};
}  // namespace ge
#endif  // OMG_PARSER_TENSORFLOW_TENSORFLOW_UTIL_H_
