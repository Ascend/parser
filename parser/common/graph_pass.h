/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef PARSER_COMMON_GRAPH_PASS_H_
#define PARSER_COMMON_GRAPH_PASS_H_

#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "common/pass.h"

namespace ge {
///
/// @ingroup domi_omg
/// @brief graph pass
/// @author
///
class GraphPass : public Pass<ge::ComputeGraph> {
 public:
  ///
  /// run graph pass
  /// @param [in] graph graph to be optimized
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimized failed
  /// @author
  ///
  virtual Status Run(ge::ComputeGraphPtr graph) = 0;
  virtual Status ClearStatus() { return SUCCESS; };
};
}  // namespace ge

#endif  // PARSER_COMMON_GRAPH_PASS_H_
