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

// AUTO GEN PLEASE DO NOT MODIFY IT
#ifndef DOMI_OP_CONSTANT_OP_H_
#define DOMI_OP_CONSTANT_OP_H_
#include "parser/common/op_def/operator.h"
#include "framework/omg/parser/parser_types.h"

namespace ge {
class ConstantOperator : public ParserOperator {
 public:
  ConstantOperator();
  ~ConstantOperator();

  ConstantOperator &Name(const std::string &name);
  ConstantOperator &VectorAttr(std::string key, std::vector<int64_t> &value);

  ConstantOperator &DType(ge::DataType t);
  ge::DataType GetDType() const;
};
}  // namespace ge

#endif  // DOMI_OP_CONSTANT_OP_H_ AUTO GEN PLEASE DO NOT MODIFY IT
