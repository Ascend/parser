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

#ifndef PARSER_TENSORFLOW_SCOPE_SCOPE_PASS_MANAGER_H_
#define PARSER_TENSORFLOW_SCOPE_SCOPE_PASS_MANAGER_H_

#include <vector>
#include "external/register/scope/scope_fusion_pass_register.h"
#include "proto/tensorflow/graph.pb.h"

using std::shared_ptr;
using std::unique_ptr;

namespace ge {
/**
 * @ingroup domi_omg
 * @brief manage passes
 */
class ScopePassManager {
 public:
  ScopePassManager() : scope_graph_(nullptr) {}
  ScopePassManager(const ScopePassManager &scope_pass_manager) = delete;
  ScopePassManager &operator=(const ScopePassManager &scope_pass_manager) = delete;
  ~ScopePassManager() {}

  shared_ptr<ScopeGraph> BuildScopeGraph(domi::tensorflow::GraphDef *graph_def);

  domi::Status AddPass(unique_ptr<ScopeBasePass> &pass);
  domi::Status Run(shared_ptr<ScopeGraph> &graph);

  std::shared_ptr<ScopeGraph> scope_graph_;

 private:
  std::vector<unique_ptr<ScopeBasePass>> graph_passes_;
};
}  // namespace ge

#endif  // PARSER_TENSORFLOW_SCOPE_SCOPE_PASS_MANAGER_H_
