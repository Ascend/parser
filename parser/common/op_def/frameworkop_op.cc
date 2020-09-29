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

#include "common/op_def/frameworkop_op.h"
#include <string>
#include "framework/common/fmk_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator::FrameworkOpOperator()
    : ParserOperator("FrameworkOp") {}

FrameworkOpOperator::~FrameworkOpOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::Name(
    const std::string &name) {
  ParserOperator::Name(name);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::Index(int64_t index) {
  Attr(RETVAL_ATTR_NAME_INDEX, static_cast<int64_t>(index));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::NodeDefPkg(
    const std::string &nodedef_pkg) {
  Attr_bt(ATTR_NAME_FRAMEWORK_NODE_DEF, nodedef_pkg);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::Frameworktype(
    int64_t framework_type) {
  Attr(ATTR_NAME_FRAMEWORK_FWK_TYPE, static_cast<int64_t>(framework_type));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::TfOpDef(
    const std::string &opdef_string) {
  Attr(ATTR_NAME_FRAMEWORK_OP_DEF, opdef_string);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::OriginalType(
    const std::string &type) {
  Attr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::FuncDefPkg(const std::string &func_string) {
  Attr_bt(ATTR_NAME_FRAMEWORK_FUNC_DEF, func_string);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY int64_t FrameworkOpOperator::GetFrameworkType() const {
  return GetIntAttr(ATTR_NAME_FRAMEWORK_FWK_TYPE);
}

FMK_FUNC_HOST_VISIBILITY std::string FrameworkOpOperator::GetNodeDefPkg() const {
  return GetStringAttr(ATTR_NAME_FRAMEWORK_NODE_DEF);
}
}  // namespace ge
