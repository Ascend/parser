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

#ifndef GE_GRAPH_PASSES_ITERATOR_FUSION_PASS_H_
#define GE_GRAPH_PASSES_ITERATOR_FUSION_PASS_H_

#include "common/graph_pass.h"
#include "register/register_fmk_types.h"

namespace ge {
class IteratorFusionPass : public GraphPass {
 public:
  explicit IteratorFusionPass(domi::FrameworkType type) : fmk_type_(type) {}

  ~IteratorFusionPass() override {};

  Status Run(ge::ComputeGraphPtr graph) final;

 private:
  domi::FrameworkType fmk_type_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_ITERATOR_FUSION_PASS_H_
