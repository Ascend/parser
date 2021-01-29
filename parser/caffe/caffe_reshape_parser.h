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

#ifndef PARSER_CAFFE_CAFFE_RESHAPE_PARSER_H_
#define PARSER_CAFFE_CAFFE_RESHAPE_PARSER_H_

#include "parser/caffe/caffe_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY CaffeReshapeParser : public CaffeOpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op) override;

  /**
   * @ingroup domi_omg
   * @brief parse weight of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
  Status ParseWeights(const Message *op_src, ge::OpDescPtr &op);

  /**
   * @ingroup domi_omg
   * @brief add const input node
   * @param [in] node to add const input
   * @param [out] node after add const input
   * @return SUCCESS add const input successfully
   * @return FAILED add const input failed
   * @author
   */
  Status AddConstInput(ge::NodePtr &node) override;
};
}  // namespace ge

#endif  // PARSER_CAFFE_CAFFE_RESHAPE_PARSER_H_
