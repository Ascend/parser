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

#ifndef DOMI_OP_FRAMEWORKOP_OP_OPERATOR_H_
#define DOMI_OP_FRAMEWORKOP_OP_OPERATOR_H_
#include "graph/debug/ge_attr_define.h"
#include "parser/common/op_def/operator.h"

namespace ge {
class FrameworkOpOperator : public ParserOperator {
 public:
  FrameworkOpOperator();

  ~FrameworkOpOperator();

  FrameworkOpOperator &Name(const std::string &name);

  FrameworkOpOperator &OriginalType(const std::string &type);

  FrameworkOpOperator &NodeDefPkg(const std::string &nodedef_pkg);

  FrameworkOpOperator &Frameworktype(int64_t framework_type);

  FrameworkOpOperator &TfOpDef(const std::string &opdef_string);

  FrameworkOpOperator &Index(int64_t index);

  FrameworkOpOperator &FuncDefPkg(const std::string &func_string);

  int64_t GetFrameworkType() const;

  std::string GetNodeDefPkg() const;
};
}  // namespace ge

#endif  // DOMI_OP_FRAMEWORKOP_OP_OPERATOR_H_
