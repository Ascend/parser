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

#include "iterator_fusion_pass.h"

#include <memory>

#include "framework/omg/parser/parser_types.h"
#include "common/util.h"
#include "graph_optimizer.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
Status IteratorFusionPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  domi::FrameworkType fmk_type = static_cast<domi::FrameworkType>(fmk_type_);
  std::unique_ptr<ParserGraphOptimizer> graph_optimizer(new (std::nothrow) ParserGraphOptimizer(graph, fmk_type));
  if (graph_optimizer == nullptr) {
    REPORT_CALL_ERROR("E19999", "New ParserGraphOptimizer failed");
    return FAILED;
  }

  graph_optimizer->SetLocalFmkopFlag(local_fmk_op_flag_);
  return graph_optimizer->FusionFmkop();
}
}  // namespace ge
