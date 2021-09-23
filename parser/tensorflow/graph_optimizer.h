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

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/omg/parser/parser_types.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "omg/omg_inner_types.h"

using std::map;
using std::string;
using std::unordered_map;
using std::vector;

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

  domi::Status FindFmkNodeCluser(unordered_map<string, vector<ge::NodePtr>> &node_cluser_Map);

  domi::Status MarkForFusion(unordered_map<string, vector<ge::NodePtr>> &node_cluser_Map);

  domi::Status UpdateGraph(vector<ge::NodePtr> &nodes);

  domi::Status InsertNode(ge::ComputeGraphPtr sub_graph, vector<ge::NodePtr> &nodes,
                          vector<ge::InDataAnchorPtr> &input_anchors, vector<ge::OutDataAnchorPtr> &output_anchors,
                          map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                          vector<ge::InControlAnchorPtr> &input_control_anchors,
                          vector<ge::OutControlAnchorPtr> &output_control_anchors,
                          unordered_map<string, ge::NodePtr> &node_map);

  domi::Status LinkInnerAnchor(unordered_map<string, ge::NodePtr> &node_map);

  domi::Status RebuildOutputAnchors(vector<ge::OutDataAnchorPtr> &output_anchors, ge::OpDescPtr fusion_op_desc);

  domi::Status RebuildInputAnchors(vector<ge::InDataAnchorPtr> &input_anchors, ge::OpDescPtr fusion_op_desc);

  domi::Status RebuildFusionNode(vector<ge::InDataAnchorPtr> &input_anchors,
                                 vector<ge::OutDataAnchorPtr> &output_anchors,
                                 map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                                 vector<ge::InControlAnchorPtr> &input_control_anchors,
                                 vector<ge::OutControlAnchorPtr> &output_control_anchors, ge::NodePtr fusion_node);

};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
