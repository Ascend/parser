/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

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

#include "parser_utils.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_registry.h"

namespace ge {
namespace {
bool HasOneNonDataNode(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  int32_t non_data_nums = 0;
  for (const auto& node : graph->GetDirectNode()) {
    if (node->GetType() != parser::DATA) {
      non_data_nums++;
    }
  }
  GELOGD("Graph has non data node num is %d", non_data_nums);
  return (non_data_nums == 1);
}
Status HandleNewOp(const NodePtr &node,
                   const ComputeGraphPtr &compute_graph,
                   const ComputeGraphPtr &sub_compute_graph,
                   const NodePtr &new_node,
                   bool no_need_change_name) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(new_node);
  if (new_node->SetOwnerComputeGraph(compute_graph) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SetOwnerComputeGraph failed for node:%s", new_node->GetName().c_str());
    GELOGE(FAILED, "[Set][OwnerComputeGraph] for node:%s failed.", new_node->GetName().c_str());
    return FAILED;
  }
  auto op_desc = new_node->GetOpDesc();
  string new_name;
  if (no_need_change_name) {
    new_name = node->GetName();
  } else {
    static std::atomic_long new_node_index(0);
    new_name = "PartitionedCall_" + new_node->GetName() + "_" + to_string(new_node_index++);
  }
  op_desc->SetName(new_name);
  std::vector<std::string> node_name_vec = { node->GetName() };
  if (!ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
      std::move(node_name_vec))) {
    GELOGW("Set %s to %s fail.", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str(), op_desc->GetName().c_str());
  }
  // handle control op
  const auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
  for (size_t i = 0UL; i < sub_graph_names.size(); i++) {
    auto branch_graph = sub_compute_graph->GetSubgraph(sub_graph_names[i]);
    GE_CHECK_NOTNULL(branch_graph);
    branch_graph->SetParentNode(new_node);
    branch_graph->SetParentGraph(compute_graph);
    compute_graph->AddSubGraph(branch_graph);
  }
  GELOGD("Handle new node[%s] for node[%s] success.", new_node->GetName().c_str(), node->GetName().c_str());
  return SUCCESS;
}
}

Status ParserUtils::ExpandOneToManyGraph(const Graph &graph, OutputMapping &output_mapping) {
  GELOGD("Begin to run ParserUtils::ExpandOneToManyGraph.");
  for (const auto &ge_node : graph.GetDirectNode()) {
    NodePtr node = NodeAdapter::GNode2Node(ge_node);
    GE_CHECK_NOTNULL(node);
    std::string ori_type;
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, ori_type);
    domi::ParseOpToGraphFunc parse_op_to_graph_func =
        domi::OpRegistry::Instance()->GetParseOpToGraphFunc(node->GetType(), ori_type);
    if (parse_op_to_graph_func == nullptr) {
      GELOGD("node:%s type:%s ori type:%s has no parse_op_to_graph_func.",
             node->GetName().c_str(), node->GetType().c_str(), ori_type.c_str());
      continue;
    }
    GELOGI("node:%s type:%s ori type:%s has registered one to many parser func.",
           node->GetName().c_str(), node->GetType().c_str(), ori_type.c_str());
    Graph subgraph("one_to_many_graph");
    Operator op = OpDescUtils::CreateOperatorFromNode(node);
    Status ret = parse_op_to_graph_func(op, subgraph);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Get one to many graph failed for op:%s.", GetOperatorName(op).c_str());
      GELOGE(FAILED, "[Invoke][ParseOpToGraphFunc]Get one to many graph failed for op:%s.",
             GetOperatorName(op).c_str());
      return FAILED;
    }
    ret = ExpandNodeToSubgraph(subgraph, node, graph, output_mapping);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Invoke][ExpandNodeToSubgraph]Expand one to many graph failed for op:%s.",
             GetOperatorName(op).c_str());
      return FAILED;
    }
  }
  GELOGD("Run ParserUtils::ExpandOneToManyGraph success.");
  return SUCCESS;
}

Status ParserUtils::ExpandNodeToSubgraph(const Graph &subgraph, const NodePtr &node, const Graph &graph,
                                         OutputMapping &output_mapping) {
  ComputeGraphPtr sub_compute_graph = GraphUtils::GetComputeGraph(subgraph);
  GE_CHECK_NOTNULL(sub_compute_graph);
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  // add subgraph node to graph.
  bool no_need_change_name = HasOneNonDataNode(sub_compute_graph);
  std::vector<NodePtr> input_nodes;
  for (const auto &sub_node : sub_compute_graph->GetDirectNode()) {
    auto new_node = compute_graph->AddNode(sub_node);
    GE_CHECK_NOTNULL(new_node);
    if (HandleNewOp(node, compute_graph, sub_compute_graph, new_node, no_need_change_name) != SUCCESS) {
      GELOGE(FAILED, "[Handle][NewOp][%s] for node[%s] failed.", new_node->GetName().c_str(), node->GetName().c_str());
      return FAILED;
    }
    if (new_node->GetType() == ge::parser::DATA) {
      input_nodes.emplace_back(new_node);
    }
  }

  // handle input context.
  Status ret = HandleInputContext(node, input_nodes, compute_graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Run][HandleInputContext] failed, node:%s.", node->GetName().c_str());
    return FAILED;
  }

  // handle output context.
  std::vector<std::pair<NodePtr, int32_t>> out_node_index = sub_compute_graph->GetGraphOutNodesInfo();
  ret = HandleOutputContext(node, out_node_index, output_mapping);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Run][HandleOutputContext] failed, node:%s.", node->GetName().c_str());
    return FAILED;
  }

  graphStatus graph_status = GraphUtils::RemoveNodeWithoutRelink(compute_graph, node);
  if (graph_status != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove node:%s from graph:%s failed.", node->GetName().c_str(),
                      compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Remove][Node] %s from graph:%s failed.", node->GetName().c_str(),
           compute_graph->GetName().c_str());
    return FAILED;
  }
  graph_status = compute_graph->TopologicalSorting();
  if (graph_status != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "TopologicalSorting failed, graph:%s.", compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Invoke][TopologicalSorting] failed, graph:%s.", compute_graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ParserUtils::HandleInputContext(const NodePtr &node,
                                       const std::vector<NodePtr> &input_nodes,
                                       const ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(node);
  for (const auto &in_n : input_nodes) {
    GE_CHECK_NOTNULL(in_n);
    int index;
    if (!AttrUtils::GetInt(in_n->GetOpDesc(), ATTR_NAME_INDEX, index)) {
      REPORT_INNER_ERROR("E19999", "GetInt failed, node:%s", in_n->GetName().c_str());
      GELOGE(FAILED, "[Get][AttrIndex] of node:%s failed.", in_n->GetName().c_str());
      return FAILED;
    }
    GELOGD("Begin to handle input node:%s with index:%d.", in_n->GetName().c_str(), index);
    // get node's in data anchor and peer out anchor
    auto node_in_anchor = node->GetInDataAnchor(index);
    GE_CHECK_NOTNULL(node_in_anchor);
    auto src_out_anchor = node_in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    auto data_out_anchor = in_n->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(data_out_anchor);
    for (const auto &peer_in_anchor : data_out_anchor->GetPeerInDataAnchors()) {
      // add data edge
      graphStatus ret = GraphUtils::RemoveEdge(data_out_anchor, peer_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed.",
                          data_out_anchor->GetOwnerNode()->GetName().c_str(),
                          peer_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(FAILED, "[Remove][Edge] from %s to %s failed.", data_out_anchor->GetOwnerNode()->GetName().c_str(),
               peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      ret = GraphUtils::RemoveEdge(src_out_anchor, node_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "remove edge from %s to %s failed.",
                          src_out_anchor->GetOwnerNode()->GetName().c_str(),
                          node_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(FAILED, "[Remove][Edge] from %s to %s failed.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
               node_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      ret = GraphUtils::AddEdge(src_out_anchor, peer_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "add edge from %s to %s failed.",
                          src_out_anchor->GetOwnerNode()->GetName().c_str(),
                          peer_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(FAILED, "[Add][Edge] from %s to %s failed.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
               peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }

      // add control edge
      if (node->GetInControlAnchor() != nullptr) {
        for (const auto &out_anchor : node->GetInControlAnchor()->GetPeerAnchors()) {
          if (GraphUtils::AddEdge(out_anchor, peer_in_anchor->GetOwnerNode()->GetInControlAnchor()) != GRAPH_SUCCESS) {
            REPORT_CALL_ERROR("E19999", "add control edge from %s to %s failed.",
                              out_anchor->GetOwnerNode()->GetName().c_str(),
                              peer_in_anchor->GetOwnerNode()->GetName().c_str());
            GELOGE(FAILED, "[Invoke][AddEdge]add control edge from %s to %s failed.",
                   out_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
            return FAILED;
          }
        }
      }
    }
    graphStatus ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, in_n);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "RemoveNodeWithoutRelink failed, graph:%s, node:%s.",
                        compute_graph->GetName().c_str(), in_n->GetName().c_str());
      GELOGE(FAILED, "[Remove][Node] %s failed, graph:%s.", in_n->GetName().c_str(), compute_graph->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ParserUtils::HandleOutputContext(const NodePtr &node,
                                        const std::vector<std::pair<NodePtr, int32_t>> &out_node_index,
                                        OutputMapping &output_mapping) {
  GE_CHECK_NOTNULL(node);
  GELOGD("The size of output node is %zu", out_node_index.size());
  for (size_t index = 0; index < out_node_index.size(); index++) {
    auto node_out_anchor = node->GetOutDataAnchor(index);
    if (node_out_anchor == nullptr) {
      continue;
    }

    NodePtr out_node = out_node_index[index].first;
    int32_t out_index = out_node_index[index].second;
    GELOGD("Begin to handle output node: %s[%d] with index:%zu", out_node->GetName().c_str(), out_index, index);
    std::string key = GenOutputKey({node->GetName(), index});
    output_mapping[key] = std::make_pair(out_node->GetName(), out_index);
    auto src_out_anchor = out_node->GetOutDataAnchor(out_index); // get out node's out anchor.
    GE_CHECK_NOTNULL(src_out_anchor);
    for (const auto &dest_in_anchor : node_out_anchor->GetPeerInDataAnchors()) {
      graphStatus ret = GraphUtils::RemoveEdge(node_out_anchor, dest_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "remove edge from node %s to node %s failed.",
                          node_out_anchor->GetOwnerNode()->GetName().c_str(),
                          dest_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(FAILED, "[Remove][Edge] from node %s to node %s failed.",
               node_out_anchor->GetOwnerNode()->GetName().c_str(),
               dest_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
      ret = GraphUtils::AddEdge(src_out_anchor, dest_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add edge from %s to %s failed.",
                          src_out_anchor->GetOwnerNode()->GetName().c_str(),
                          dest_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(FAILED, "[Add][Edge] from %s to %s failed.", src_out_anchor->GetOwnerNode()->GetName().c_str(),
               dest_in_anchor->GetOwnerNode()->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

string ParserUtils::GenOutputKey(const OutputNodeInfo &node_info) {
  return node_info.first + ":" + std::to_string(node_info.second);
}

void ParserUtils::UpdateOutputNodeInfo(const OutputMapping &final_output_nodes, OutputNodeInfo &output_node_info) {
  std::string key = ParserUtils::GenOutputKey(output_node_info);
  auto iter = final_output_nodes.find(key);
  if (iter != final_output_nodes.end()) {
    output_node_info = iter->second;
    GELOGD("Update output node info, origin[%s], now[%s].",
           key.c_str(), ParserUtils::GenOutputKey(output_node_info).c_str());
  }
}

void ParserUtils::UpdateOutputCtx(const OutputMapping &final_output_nodes, OutputMapping &tensor_to_nodes) {
  for (auto &tensor_to_node : tensor_to_nodes) {
    std::string tensor_name = tensor_to_node.first;
    auto &output_node_info = tensor_to_node.second;
    UpdateOutputNodeInfo(final_output_nodes, output_node_info);
  }
}

std::string ParserUtils::GetOperatorName(const Operator &op) {
  AscendString name;
  (void)op.GetName(name);
  return name.GetString() == nullptr ? "" : std::string(name.GetString());
}

std::string ParserUtils::GetOperatorType(const Operator &op) {
  AscendString type;
  (void)op.GetOpType(type);
  return type.GetString() == nullptr ? "" : std::string(type.GetString());
}

std::string ParserUtils::GetGraphName(const Graph &graph) {
  AscendString name;
  (void)graph.GetName(name);
  return name.GetString() == nullptr ? "" : std::string(name.GetString());
}
}  // namespace ge