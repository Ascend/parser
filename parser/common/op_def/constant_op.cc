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

#include "common/op_def/constant_op.h"
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ConstantOperator::ConstantOperator() : ParserOperator("Constant") {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ConstantOperator::~ConstantOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ConstantOperator &ConstantOperator::Name(const std::string &name) {
  ParserOperator::Name(name);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ConstantOperator &ConstantOperator::VectorAttr(
    std::string key, std::vector<int64_t> &value) {
  Attr(key, value);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ConstantOperator &ConstantOperator::DType(ge::DataType t) {
  Attr(VAR_ATTR_DTYPE, (int64_t)t);
  return *this;
}

ge::DataType ConstantOperator::GetDType() const { return (ge::DataType)GetIntAttr(VAR_ATTR_DTYPE); }
}  // namespace ge
