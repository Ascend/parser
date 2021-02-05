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

#ifndef OMG_PARSER_TENSORFLOW_TENSORFLOW_FUSION_OP_PARSER_H_
#define OMG_PARSER_TENSORFLOW_TENSORFLOW_FUSION_OP_PARSER_H_

#include <vector>
#include "graph/ge_tensor.h"
#include "omg/parser/op_parser.h"
#include "parser/tensorflow/tensorflow_fusionop_util.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"

using std::vector;
using google::protobuf::Message;
using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;

namespace ge {
/**
 * @ingroup domi_omg
 * @brief Used to parse TensorFlow operator information
 */
class PARSER_FUNC_VISIBILITY TensorFlowFusionOpParser : public TensorFlowOpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief Analytic operator parameters
   * @param [in] v_input_const Operator parameters to be parsed
   * @param [out] op_dest Parsed model data
   * @return SUCCESS Parsing success
   * @return FAILED Parsing failed
   */
  virtual Status ParseParams(const vector<const NodeDef *> &v_input_const, ge::NodePtr &node);

  /**
   * @ingroup domi_omg
   * @brief Analytic operator parameters
   * @param [in] op_src Parameter data to be parsed
   * @param [out] graph Parsed parameter data
   * @return SUCCESS Parsing success
   * @return FAILED Parsing failed
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) final;

 protected:
  /**
   * @ingroup domi_omg
   * @brief Parse parameters from const op
   * @param [in] op_src Model data to be parsed
   * @param [out] op_dest Parsed model data
  * @return SUCCESS Parsing success
  * @return FAILED Parsing failed
   *
   */
  // template <class T>
  Status ParseParamFromConst(const NodeDef *input_const, int32_t &param);

  Status ParseParamFromConst(const NodeDef *nodeDef, int32_t &param, int index);

  Status ParseParamFromConst(const NodeDef *input_const, float &param);

  Status ParseParamFromConst(const NodeDef *nodeDef, float &param, int index);

  Status GetTensorFromNode(const NodeDef *nodeDef, TensorProto &tensor);

  Status ParseHalfFromConst(const NodeDef *node_def, float &param, int index = 0);

  Status ParseWeightFromConst(const NodeDef *node_def, ge::GeTensorPtr &weight);
};
}  // namespace ge

#endif  // OMG_PARSER_TENSORFLOW_TENSORFLOW_FUSION_OP_PARSER_H_
