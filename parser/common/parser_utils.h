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

#include <unordered_map>
#include "graph/graph.h"
#include "graph/node.h"
#include "external/ge/ge_api_error_codes.h"

namespace ge {
class ParserUtils {
 public:
  using OutputNodeInfo = std::pair<std::string, int32_t>;
  using OutputMapping = std::unordered_map<std::string, OutputNodeInfo>;
  static Status ExpandOneToManyGraph(const Graph &graph, OutputMapping &output_mapping);
  static string GenOutputKey(const OutputNodeInfo &node_info);
  static void UpdateOutputNodeInfo(const OutputMapping &final_output_nodes, OutputNodeInfo &output_node_info);
  static void UpdateOutputCtx(const OutputMapping &final_output_nodes, OutputMapping &tensor_to_nodes);
  static std::string GetOperatorName(const Operator &op);
  static std::string GetOperatorType(const Operator &op);
  static std::string GetGraphName(const Graph &graph);

 private:
  static Status ExpandNodeToSubgraph(const Graph &subgraph, const NodePtr &node, const Graph &graph,
                                     OutputMapping &output_mapping);
  static Status HandleInputContext(const NodePtr &node,
                                   const std::vector<NodePtr> &input_nodes,
                                   const ComputeGraphPtr &compute_graph);
  static Status HandleOutputContext(const NodePtr &node, 
                                    const std::vector<std::pair<NodePtr, int32_t>> &out_node_index,
                                    OutputMapping &output_mapping);
};
}  // namespace ge
#endif  // PARSER_COMMON_PARSER_UTILS_H_
