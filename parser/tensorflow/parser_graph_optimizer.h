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

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "omg/omg_inner_types.h"

namespace ge {
class ParserGraphOptimizer {
 public:
  explicit ParserGraphOptimizer(ge::ComputeGraphPtr graph, domi::FrameworkType type = domi::TENSORFLOW)
      : graph_(graph), fmktype_(type) {}

  ~ParserGraphOptimizer() {}

  domi::Status FusionFmkop();

 private:
  ge::ComputeGraphPtr graph_;
  domi::FrameworkType fmktype_;

  domi::Status FindFmkNodeCluser(std::unordered_map<std::string, std::vector<ge::NodePtr>> &node_cluser_Map) const;

  domi::Status MarkForFusion(std::unordered_map<std::string, std::vector<ge::NodePtr>> &node_cluster_map);

  domi::Status GetFusionCluster(const bool has_get_next, const bool has_dyn_get_next,
                                unordered_map<string, vector<NodePtr>> &node_cluster_map);

  domi::Status UpdateGraph(std::vector<ge::NodePtr> &nodes);

  static domi::Status InsertNode(ge::ComputeGraphPtr sub_graph, std::vector<ge::NodePtr> &nodes,
                                 std::vector<ge::InDataAnchorPtr> &input_anchors,
                                 std::vector<ge::OutDataAnchorPtr> &output_anchors,
                                 std::map<ge::OutDataAnchorPtr, std::vector<ge::InDataAnchorPtr>> &output_in_map,
                                 std::vector<ge::InControlAnchorPtr> &input_control_anchors,
                                 std::vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                 std::unordered_map<std::string, ge::NodePtr> &node_map);

  domi::Status LinkInnerAnchor(std::unordered_map<std::string, ge::NodePtr> &node_map) const;

  static domi::Status RebuildOutputAnchors(std::vector<ge::OutDataAnchorPtr> &output_anchors,
                                           ge::OpDescPtr fusion_op_desc);

  static domi::Status RebuildInputAnchors(std::vector<ge::InDataAnchorPtr> &input_anchors,
                                          ge::OpDescPtr fusion_op_desc);

  static domi::Status RebuildFusionNode(std::vector<ge::InDataAnchorPtr> &input_anchors,
                                        std::vector<ge::OutDataAnchorPtr> &output_anchors,
                                        std::map<ge::OutDataAnchorPtr, std::vector<ge::InDataAnchorPtr>> &output_in_map,
                                        std::vector<ge::InControlAnchorPtr> &input_control_anchors,
                                        std::vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                        ge::NodePtr fusion_node);
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_