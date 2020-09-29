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

#include "common/op_def/fill_op.h"
#include "framework/common/fmk_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator::FillOperator() : ParserOperator("Fill") {}

FMK_FUNC_DEV_VISIBILITY FillOperator::~FillOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator &FillOperator::DataType(int64_t dataType) {
  Attr("T", static_cast<int64_t>(dataType));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator &FillOperator::Alpha(float alpha) {
  Attr("alpha", static_cast<float>(alpha));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator &FillOperator::Beta(float beta) {
  Attr("beta", static_cast<float>(beta));
  return *this;
}

int64_t FillOperator::GetDataType() const { return GetIntAttr("T"); }

float FillOperator::GetAlpha() const { return GetFloatAttr("alpha"); }

float FillOperator::GetBeta() const { return GetFloatAttr("beta"); }
}  // namespace ge
