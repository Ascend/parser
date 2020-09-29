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
#include "common/op_def/no_op_op.h"
#include <string>

namespace ge {
FMK_FUNC_HOST_VISIBILITY NoOpOperator::NoOpOperator() : ParserOperator("NoOp") {}

FMK_FUNC_HOST_VISIBILITY NoOpOperator::~NoOpOperator() {}

FMK_FUNC_HOST_VISIBILITY NoOpOperator &NoOpOperator::Name(const std::string &name) {
  ParserOperator::Name(name);
  return *this;
}
}  // namespace ge
