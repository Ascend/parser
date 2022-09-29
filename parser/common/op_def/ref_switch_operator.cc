/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "common/op_def/ref_switch_operator.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY RefSwitchOperator::RefSwitchOperator() : ParserOperator("RefSwitch") {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY RefSwitchOperator::~RefSwitchOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY RefSwitchOperator &RefSwitchOperator::T(ge::DataType t) {
  Attr("T", static_cast<int64_t>(t));
  return *this;
}
}  // namespace ge  AUTO GEN PLEASE DO NOT MODIFY IT