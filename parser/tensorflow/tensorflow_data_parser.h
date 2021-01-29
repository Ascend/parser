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

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_DATA_PARSER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_DATA_PARSER_H_

#include <string>
#include <vector>
#include "parser/common/data_op_parser.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY TensorFlowDataParser : public TensorFlowOpParser, public DataOpParser {
 public:
  /**
  * @ingroup domi_omg
  * @brief parse weight
  * @param [in] v_input_const weight data to be parsed
  * @param [out] op_dest weight data after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_def) override;

 private:
  /**
  * @ingroup domi_omg
  * @brief Parsing input from model
  * @param [in] op_src model to be parsed
  * @param [out] op_def input information after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseInputFromModel(const Message *op_src, ge::OpDescPtr &op_def);

  /**
  * @ingroup domi_omg
  * @brief parse input set by users
  * @param [in] op_src model to be parsed
  * @param [out] op_def input information after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  * @author
  */
  Status ParseInputFromUser(const Message *op_src, const ge::OpDescPtr &op_def);

  /**
  * @ingroup domi_omg
  * @brief Check whether the input shape entered by the user matches the input shape defined by the model
  * @return SUCCESS match
  * @return FAILED not match
  * @author
  */
  Status CheckInputShape(const std::string &name);

  std::vector<int64_t> model_input_dims_v;

  std::vector<int64_t> user_input_dims_v;
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_DATA_PARSER_H_
