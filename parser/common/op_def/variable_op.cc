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

#include "parser/common/op_def/variable_op.h"

#include "graph/debug/ge_attr_define.h"

namespace ge {
VariableOperator::VariableOperator() : ParserOperator(ge::parser::VARIABLE) {}

VariableOperator::~VariableOperator() {}

VariableOperator &VariableOperator::Name(const std::string &name) {
  ParserOperator::Name(name);
  return *this;
}

VariableOperator &VariableOperator::Container(const std::string &container) {
  Attr(VAR_ATTR_CONTAINER, container);
  return *this;
}

VariableOperator &VariableOperator::SharedName(const std::string &sharedname) {
  Attr(VAR_ATTR_SHARED_NAME, sharedname);
  return *this;
}

VariableOperator &VariableOperator::Placement(const std::string &placement) {
  Attr(ATTR_VARIABLE_PLACEMENT, placement);
  return *this;
}

VariableOperator &VariableOperator::MemType(const uint32_t &mem_type) {
  Attr(ATTR_OUTPUT_MEMORY_TYPE, mem_type);
  return *this;
}

VariableOperator &VariableOperator::SrcType(const int64_t &dtype) {
  Attr(VAR_ATTR_DTYPE, dtype);
  return *this;
}

VariableOperator &VariableOperator::VarShape(const std::vector<int64_t> &shape_value) {
  Attr(VAR_ATTR_SHAPE, shape_value);
  return *this;
}

int64_t VariableOperator::GetVarSrcType() const { return GetIntAttr(VAR_ATTR_DTYPE); }
}  //  namespace ge
