/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef PARSER_COMMON_PARSER_UTILS_H_
#define PARSER_COMMON_PARSER_UTILS_H_

#include "graph/graph.h"
#include "graph/node.h"
#include "external/ge/ge_api_error_codes.h"

namespace ge {
class ParserUtils {
 public:
  static Status ExpandOneToManyGraph(Graph &graph);

 private:
  static Status ExpandNodeToSubgraph(const Graph &subgraph, const NodePtr &node, Graph &graph);
  static Status HandleInputContext(const NodePtr &node,
                                   const std::vector<NodePtr> &input_nodes,
                                   const ComputeGraphPtr &compute_graph);
  static Status HandleOutputContext(const NodePtr &node, 
                                    const std::vector<std::pair<NodePtr, int32_t>> &out_node_index);
};
}  // namespace ge
#endif  // PARSER_COMMON_PARSER_UTILS_H_
