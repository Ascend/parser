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

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_PARSER_ADAPTER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_PARSER_ADAPTER_H_

#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowCustomParserAdapter : public TensorFlowOpParser {
 public:
  /**
  * @ingroup domi_omg
  * @brief Parsing model file information
  * @param [in] op_src model data to be parsed
  * @param [out] op_dest model data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

   /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] op_dest params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   * @author
   */
   Status ParseParams(const Operator &op_src, ge::OpDescPtr &op_dest);
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_CUSTOM_PARSER_ADAPTER_H_
