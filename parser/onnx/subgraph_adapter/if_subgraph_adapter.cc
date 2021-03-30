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

#include "if_subgraph_adapter.h"
#include "subgraph_adapter_factory.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"

namespace ge{
namespace {
const std::map<std::string, int> kAttrNameToIndex = {{"then_branch", 0}, {"else_branch", 1}};
const int kIfNodeAttrSize = 2;
}
Status IfSubgraphAdapter::AdaptAndFindAllSubgraphs(ge::onnx::NodeProto *parent_node,
    std::vector<ge::onnx::GraphProto *> &onnx_graphs,
    std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph) {
  GE_CHECK_NOTNULL(parent_node);
  GELOGI("Onnx parent node name=%s, op type=%s, adapt subgraph.", parent_node->name().c_str(),
         parent_node->op_type().c_str());

  auto ret = ParseIfNodeSubgraphs(parent_node, onnx_graphs, name_to_onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Node] Parse if node failed.");
    REPORT_CALL_ERROR("E19999", "[Parse][Node] Parse if node:%s failed.", parent_node->name().c_str());
    return ret;
  }

  return SUCCESS;
}

Status IfSubgraphAdapter::ParseIfNodeSubgraphs(ge::onnx::NodeProto *parent_node,
                                               std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                               std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph) {
  if (parent_node->attribute_size() != kIfNodeAttrSize) {
    GELOGE(FAILED, "[Parse][Node] Invalid graph, if node attribute size:%d must be 2.", parent_node->attribute_size());
    REPORT_INNER_ERROR("E19999", "Invalid graph, if node attribute size:%d must be 2.", parent_node->attribute_size());
    return FAILED;
  }

  GELOGD("node attribute size:%d.", parent_node->attribute_size());
  std::set<std::string> all_inputs;
  // for onnx graph, the first attribute may be else branch and the second attribute may be then branch
  for (int i = 0; i < parent_node->attribute_size(); i++) {
    ge::onnx::AttributeProto *attribute = parent_node->mutable_attribute(i);
    GE_CHECK_NOTNULL(attribute);
    std::string attr_name = attribute->name();
    auto itr = kAttrNameToIndex.find(attr_name);
    if (itr == kAttrNameToIndex.end()) {
      GELOGE(FAILED, "[Parse][Attribute] Invalid attribute name:%s, it should be then_branch or else_branch.",
             attr_name.c_str());
      REPORT_INNER_ERROR("E19999", "Invalid attribute name:%s, it should be then_branch or else_branch.",
                         attr_name.c_str());
      return FAILED;
    }
    std::string unique_subgraph_name;
    OnnxUtil::GenUniqueSubgraphName(itr->second, itr->first, parent_node->name(), unique_subgraph_name);
    GELOGI("Adapt if node attribute:%s, subgraph name:%s.", attr_name.c_str(), unique_subgraph_name.c_str());
    ge::onnx::GraphProto *onnx_graph = attribute->mutable_g();
    name_to_onnx_graph[unique_subgraph_name] = onnx_graph;
    onnx_graphs.emplace_back(onnx_graph);

    auto ret = GetSubgraphsAllInputs(*onnx_graph, all_inputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][Inputs] Get subgraph all inputs failed, attr_name:%s.", attr_name.c_str());
      REPORT_INNER_ERROR("E19999", "Get subgraph all inputs failed, attr_name:%s.", attr_name.c_str());
      return ret;
    }
  }

  for (auto &onnx_graph : onnx_graphs) {
    AddInputNodeForGraph(all_inputs, *onnx_graph);
  }

  AddInputForParentNode(all_inputs, *parent_node);
  return SUCCESS;
}

Status IfSubgraphAdapter::GetSubgraphsAllInputs(ge::onnx::GraphProto &onnx_graph,
                                                std::set<std::string> &all_inputs) {
  std::set<std::string> graph_inputs;
  std::set<std::string> graph_outputs;
  for (int i = 0; i < onnx_graph.node_size(); i++) {
    ge::onnx::NodeProto *node_proto = onnx_graph.mutable_node(i);
    for (int j = 0; j < node_proto->input_size(); j++) {
      graph_inputs.emplace(node_proto->input(j));
    }
    for (int j = 0; j < node_proto->output_size(); j++) {
      graph_outputs.emplace(node_proto->output(j));
    }
  }

  for (const auto &input : graph_inputs) {
    auto out_iter = graph_outputs.find(input);
    if (out_iter == graph_outputs.end()) {
      // Record input node need to be constructed
      all_inputs.emplace(input);
    }
  }

  return SUCCESS;
}

void IfSubgraphAdapter::AddInputNodeForGraph(const std::set<std::string> &all_inputs,
                                             ge::onnx::GraphProto &onnx_graph) {
  for (const auto &input_name : all_inputs) {
    ge::onnx::ValueInfoProto *value_info = onnx_graph.add_input();
    value_info->set_name(input_name);
  }
}

void IfSubgraphAdapter::AddInputForParentNode(const std::set<std::string> &all_inputs,
                                              ge::onnx::NodeProto &parent_node) {
  for (const auto &input_name : all_inputs) {
    parent_node.add_input(input_name);
  }
}
REGISTER_SUBGRAPH_ADAPTER_CREATOR(IF, IfSubgraphAdapter);
}  // namespace ge
