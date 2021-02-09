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

#ifndef OMG_PARSER_TENSORFLOW_TENSORFLOW_OP_PARSER_H_
#define OMG_PARSER_TENSORFLOW_TENSORFLOW_OP_PARSER_H_

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

#include <string>
#include <vector>
#include "framework/omg/parser/op_parser.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "register/tensor_assign.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"


using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using google::protobuf::int32;
using google::protobuf::int64;
using google::protobuf::Message;
using std::string;
using std::vector;
using Status = domi::Status;
using domi::tensorflow::AttrValue;
using domi::tensorflow::DataType;
using domi::tensorflow::DT_BOOL;
using domi::tensorflow::DT_FLOAT;
using domi::tensorflow::DT_INT32;
using domi::tensorflow::DT_INT64;
using domi::tensorflow::DT_INVALID;
using domi::tensorflow::TensorShapeProto;
using domi::tensorflow::TensorShapeProto_Dim;

namespace ge {
/**
 * @ingroup domi_omg
 * @brief used to parse TensorFlow operator information
 */
class PARSER_FUNC_VISIBILITY TensorFlowOpParser : public OpParser {
 public:

  /**
   * @ingroup domi_omg
   * @brief parse params
   * @param [in] op_src        op to be parsed
   * @param [out] op_dest      the parsed op
   * @return SUCCESS           parse success
   * @return FAILED            Parse failed
   *
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override {
    return domi::SUCCESS;
  }

  /**
   * @ingroup domi_omg
   * @brief parse params
   * @param [in] op_src        op to be parsed
   * @param [out] op_dest      the operator
   * @return SUCCESS           parse success
   * @return FAILED            Parse failed
   *
   */
  Status ParseParams(const Message *op_src, ge::Operator &op_dest) override {
    return domi::SUCCESS;
  }

  /**
   * @ingroup domi_omg
   * @brief parsie weight
   * @param [in] op_src        op to be parsed
   * @param [out] op_dest      the parsed op
   * @return SUCCESS           parsing success
   * @return FAILED            parsing failed
   *
   */
  Status ParseWeights(const Message *op_src, ge::NodePtr &node) final {
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // OMG_PARSER_TENSORFLOW_TENSORFLOW_OP_PARSER_H_
