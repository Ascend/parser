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
#include "common/op_def/shape_n_op.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/omg/parser/parser_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY ShapeNOperator::ShapeNOperator() : ParserOperator("ShapeN") {}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator::~ShapeNOperator() {}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::Name(const std::string &name) {
  ParserOperator::Name(name);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::N(int64_t n) {
  Attr(SHAPEN_ATTR_N, n);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY int64_t ShapeNOperator::GetN() const { return GetIntAttr(SHAPEN_ATTR_N); }

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::InType(ge::DataType t) {
  Attr(SHAPEN_ATTR_IN_TYPE, (int64_t)t);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY ge::DataType ShapeNOperator::GetInType() const {
  return (ge::DataType)GetIntAttr(SHAPEN_ATTR_IN_TYPE);
}

FMK_FUNC_HOST_VISIBILITY ShapeNOperator &ShapeNOperator::OutType(ge::DataType t) {
  Attr(SHAPEN_ATTR_OUT_TYPE, (int64_t)t);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY ge::DataType ShapeNOperator::GetOutType() const {
  return (ge::DataType)GetIntAttr(SHAPEN_ATTR_OUT_TYPE);
}
}  // namespace ge
