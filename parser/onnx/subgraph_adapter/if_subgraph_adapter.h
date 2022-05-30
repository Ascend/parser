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

#ifndef GE_PARSER_ONNX_SUBGRAPH_ADAPTER_IF_SUBGRAPH_ADAPTER_H_
#define GE_PARSER_ONNX_SUBGRAPH_ADAPTER_IF_SUBGRAPH_ADAPTER_H_

#include <set>
#include <string>
#include "subgraph_adapter.h"

namespace ge {
class PARSER_FUNC_VISIBILITY IfSubgraphAdapter : public SubgraphAdapter {
 public:
  domi::Status AdaptAndFindAllSubgraphs(ge::onnx::NodeProto *parent_node,
                                        std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                        std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph,
                                        const std::string &parent_graph_name = "") override;

 private:
  domi::Status ParseIfNodeSubgraphs(ge::onnx::NodeProto *parent_node, std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                    std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph,
                                    const std::string &parent_graph_name);
  domi::Status GetSubgraphsAllInputs(ge::onnx::GraphProto &onnx_graph, std::set<std::string> &all_inputs) const;
  void AddInputNodeForGraph(const std::set<std::string> &all_inputs, ge::onnx::GraphProto &onnx_graph) const;
  void AddInputForParentNode(const std::set<std::string> &all_inputs, ge::onnx::NodeProto &parent_node) const;
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_SUBGRAPH_ADAPTER_IF_SUBGRAPH_ADAPTER_H_
