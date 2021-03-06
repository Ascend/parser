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
Status HandleNewOp(const NodePtr &node, const ComputeGraphPtr &compute_graph, const NodePtr &new_node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(new_node);
  if (new_node->SetOwnerComputeGraph(compute_graph) != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Set owner graph for node:%s failed.", new_node->GetName().c_str());
    return FAILED;
  }
  auto op_desc = new_node->GetOpDesc();
  static std::atomic_long new_node_index(0);
  auto new_name = "PartitionedCall_" + new_node->GetName() + "_" + to_string(new_node_index++);
  op_desc->SetName(new_name);
  bool ret = ge::AttrUtils::SetListStr(op_desc,
                                       ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                       std::move(std::vector<std::string>{node->GetName()}));
  if (!ret) {
    GELOGW("Set %s to %s fail.", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str(), op_desc->GetName().c_str());
  }
  GELOGD("Handle new op[%s] for node[%s] success.", new_node->GetName().c_str(), node->GetName().c_str());
  return SUCCESS;
}
}

Status ParserUtils::ExpandOneToManyGraph(Graph &graph) {
  GELOGD("Begin run ParserUtils::ExpandOneToManyGraph.");
  for (const auto &gn : graph.GetDirectNode()) {
    NodePtr n = NodeAdapter::GNode2Node(gn);
    GE_CHECK_NOTNULL(n);
    std::string ori_type;
    (void)AttrUtils::GetStr(n->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, ori_type);
    domi::ParseOpToGraphFunc parse_op_to_graph_func =
        domi::OpRegistry::Instance()->GetParseOpToGraphFunc(n->GetType(), ori_type);
    if (parse_op_to_graph_func == nullptr) {
      GELOGD("node:%s type:%s ori type:%s has no parse_op_to_graph_func.",
             n->GetName().c_str(), n->GetType().c_str(), ori_type.c_str());
      continue;
    }
    GELOGI("node:%s type:%s ori type:%s has registered one to many parser func.",
           n->GetName().c_str(), n->GetType().c_str(), ori_type.c_str());
    Graph subgraph("one_to_many_graph");
    Operator op = OpDescUtils::CreateOperatorFromNode(n);
    Status ret = parse_op_to_graph_func(op, subgraph);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Get one to many graph failed for op:%s.", op.GetName().c_str());
      return FAILED;
    }
    ret = ExpandNodeToSubgraph(subgraph, n, graph);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Expand one to many graph failed for op:%s.", op.GetName().c_str());
      return FAILED;
    }
  }
  GELOGD("run ParserUtils::ExpandOneToManyGraph success.");
  return SUCCESS;
}

Status ParserUtils::ExpandNodeToSubgraph(const Graph &subgraph, const NodePtr &node, Graph &graph) {
  ComputeGraphPtr sub_compute_graph = GraphUtils::GetComputeGraph(subgraph);
  GE_CHECK_NOTNULL(sub_compute_graph);
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  // add subgraph node to graph.
  std::vector<NodePtr> input_nodes;
  for (const auto &n : sub_compute_graph->GetDirectNode()) {
    auto new_node = compute_graph->AddNode(n);
    GE_CHECK_NOTNULL(new_node);
    if (HandleNewOp(node, compute_graph, new_node) != SUCCESS) {
      GELOGE(FAILED, "Handle new op[%s] for node[%s] failed.", new_node->GetName().c_str(), node->GetName().c_str());
      return FAILED;
    }
    if (new_node->GetType() == ge::parser::DATA) {
      input_nodes.emplace_back(new_node);
    }
  }

  // handle input context.
  Status ret = HandleInputContext(node, input_nodes, compute_graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "run ParserUtils::HandleInputContext failed.");
    return FAILED;
  }

  // handle output context.
  std::vector<std::pair<NodePtr, int32_t>> out_node_index = sub_compute_graph->GetGraphOutNodesInfo();
  ret = HandleOutputContext(node, out_node_index);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "run ParserUtils::HandleOutputContext failed.");
    return FAILED;
  }

  graphStatus graph_status = GraphUtils::RemoveNodeWithoutRelink(compute_graph, node);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Remove node:%s failed.", node->GetName().c_str());
    return FAILED;
  }
  graph_status = compute_graph->TopologicalSorting();
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Topological sorting failed.");
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
      GELOGE(FAILED, "Get attr index of node:%s failed.", in_n->GetName().c_str());
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
        GELOGE(FAILED, "remove data out anchor and peer in anchor failed.");
        return FAILED;
      }
      ret = GraphUtils::RemoveEdge(src_out_anchor, node_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(FAILED, "remove node in anchor and peer out anchor failed.");
        return FAILED;
      }
      ret = GraphUtils::AddEdge(src_out_anchor, peer_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(FAILED, "link node's peer out anchor and data's peer in anchor failed.");
        return FAILED;
      }

      // add control edge
      if (node->GetInControlAnchor() != nullptr) {
        for (const auto &out_anchor : node->GetInControlAnchor()->GetPeerAnchors()) {
          graphStatus ret = GraphUtils::AddEdge(out_anchor, peer_in_anchor->GetOwnerNode()->GetInControlAnchor());
          if (ret != GRAPH_SUCCESS) {
            GELOGE(FAILED, "add control edge failed.");
            return FAILED;
          }
        }
      }
    }
    graphStatus ret = GraphUtils::RemoveNodeWithoutRelink(compute_graph, in_n);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(FAILED, "remove node:%s failed.", in_n->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ParserUtils::HandleOutputContext(const NodePtr &node,
                                        const std::vector<std::pair<NodePtr, int32_t>> &out_node_index) {
  GE_CHECK_NOTNULL(node);
  GELOGD("The size of out node is %zu", out_node_index.size());
  for (size_t index = 0; index < out_node_index.size(); index++) {
    auto node_out_anchor = node->GetOutDataAnchor(index);
    if (node_out_anchor == nullptr) {
      continue;
    }

    NodePtr out_node = out_node_index[index].first;
    int32_t out_index = out_node_index[index].second;
    GELOGD("Begin to handle output node:%s[%d] with index:%zu", out_node->GetName().c_str(), out_index, index);
    auto src_out_anchor = out_node->GetOutDataAnchor(out_index); // get out node's out anchor.
    GE_CHECK_NOTNULL(src_out_anchor);
    for (const auto &dest_in_anchor : node_out_anchor->GetPeerInDataAnchors()) {
      graphStatus ret = GraphUtils::RemoveEdge(node_out_anchor, dest_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(FAILED, "remove node's out anchor and peer in anchor failed.");
        return FAILED;
      }
      ret = GraphUtils::AddEdge(src_out_anchor, dest_in_anchor);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(FAILED, "link node's peer out anchor and out node's out anchor failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge