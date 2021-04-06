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

#include "subgraph_adapter_factory.h"
#include "framework/common/debug/ge_log.h"

namespace ge{
SubgraphAdapterFactory* SubgraphAdapterFactory::Instance() {
  static SubgraphAdapterFactory instance;
  return &instance;
}

std::shared_ptr<SubgraphAdapter> SubgraphAdapterFactory::CreateSubgraphAdapter(
    const std::string &op_type) {
  // First look for CREATOR_FUN based on OpType, then call CREATOR_FUN to create SubgraphAdapter.
  auto iter = subgraph_adapter_creator_map_.find(op_type);
  if (iter != subgraph_adapter_creator_map_.end()) {
    return iter->second();
  }

  GELOGW("SubgraphAdapterFactory::CreateSubgraphAdapter: Not supported type: %s", op_type.c_str());
  return nullptr;
}

// This function is only called within the constructor of the global SubgraphAdapterRegisterar object,
// and does not involve concurrency, so there is no need to lock it
void SubgraphAdapterFactory::RegisterCreator(const std::string &type, CREATOR_FUN fun) {
  std::map<std::string, CREATOR_FUN> *subgraph_adapter_creator_map = &subgraph_adapter_creator_map_;
  GELOGD("SubgraphAdapterFactory::RegisterCreator: op type:%s.", type.c_str());
  (*subgraph_adapter_creator_map)[type] = fun;
}
}  // namespace ge
