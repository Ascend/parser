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

#ifndef PARSER_TENSORFLOW_TENSORFLOW_RESHAPE_PARSER_H_
#define PARSER_TENSORFLOW_TENSORFLOW_RESHAPE_PARSER_H_

#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowReshapeParser : public TensorFlowOpParser {
 private:
  Status ParseDesc(const domi::tensorflow::AttrValue &attr_value, ge::GeTensorDesc &ge_desc);

 public:
  /**
  * @ingroup domi_omg
  * @brief parse weight information
  * @param [in] v_input_const weight data to be parsed
  * @param [out] op_dest weight data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;
};
}  // namespace ge

#endif  // PARSER_TENSORFLOW_TENSORFLOW_RESHAPE_PARSER_H_
