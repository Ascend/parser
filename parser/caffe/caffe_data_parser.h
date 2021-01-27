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

#ifndef PARSER_CAFFE_CAFFE_DATA_PARSER_H_
#define PARSER_CAFFE_CAFFE_DATA_PARSER_H_

#include <string>
#include <vector>
#include "parser/caffe/caffe_op_parser.h"
#include "parser/common/data_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY CaffeDataParser : public CaffeOpParser, public DataOpParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief parse params of the operation
   * @param [in] op_src params to be parsed
   * @param [out] graph params after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op) override;

 private:
  /**
   * @ingroup domi_omg
   * @brief Get the output dimension according to the input dimension
   * @param [in] name the name of the input layer
   * @param [in] input_dims the dimension of the input layer
   * @param [out] op_def op after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status GetOutputDesc(const std::string &name, int dim_size,
                       const std::vector<int64_t> &input_dims, ge::OpDescPtr &op);

  // caffe data layer type could be type of `Input` or `DummyData`
  Status ParseParamsForInput(const domi::caffe::LayerParameter *layer, ge::OpDescPtr &op);
  Status ParseParamsForDummyData(const domi::caffe::LayerParameter *layer, ge::OpDescPtr &op);
};
}  // namespace ge

#endif  // PARSER_CAFFE_CAFFE_DATA_PARSER_H_
