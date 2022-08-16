/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "auto_mapping_subgraph_io_index_func.h"
#include <vector>
#include "external/register/register.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "register/register_fmk_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"

namespace ge {
namespace {
std::vector<NodePtr> FindNodesByType(const ge::ComputeGraphPtr &graph, const std::string &type) {
  std::vector<NodePtr> nodes;
  for (const auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    std::string node_type = NodeUtils::GetNodeType(node);
    GELOGI("Find node %s, node type is %s.", node->GetName().c_str(), node_type.c_str());
    if (node_type == type) {
      nodes.push_back(node);
      continue;
    }
  }
  return nodes;
}

Status AutoMappingSubgraphIndexByOutputNodesInfo(const ge::ComputeGraphPtr &compute_graph,
    const std::function<Status(int netoutput_index, int &parent_output_index)> &output) {
  const auto &out_nodes_info = compute_graph->GetGraphOutNodesInfo();
  for (size_t i = 0; i < out_nodes_info.size(); ++i) {
    const auto &out_node = out_nodes_info[i].first;
    int32_t output_index = out_nodes_info[i].second;
    int64_t index = static_cast<int64_t>(i);
    int parent_index = -1;
    auto ret = output(index, parent_index);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Get parent output index %ld failed, node:%s", index, out_node->GetName().c_str());
      GELOGE(FAILED, "[Get][ParentOutputIndex] Get parent output index %ld failed, node:%s",
             index, out_node->GetName().c_str());
      return FAILED;
    }
    auto op_desc = out_node->GetOpDesc();
    if (op_desc == nullptr) {
      GELOGE(FAILED, "[Get][OpDesc] Op desc is null!");
      return FAILED;
    }
    auto output_desc = op_desc->MutableOutputDesc(output_index);
    if (output_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "Can not find output tensor desc from node:%s, index %d",
                        out_node->GetName().c_str(), output_index);
      GELOGE(FAILED, "[Get][OutputDesc] Can not find output tensor desc from node:%s, index %d",
             out_node->GetName().c_str(), output_index);
      return FAILED;
    }
    if (!ge::AttrUtils::SetInt(output_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_INNER_ERROR("E19999", "Set attr:%s of op:%s failed, parent_index:%d",
                         ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), out_node->GetName().c_str(), parent_index);
      GELOGE(FAILED, "[Set][Attr] Set attr:%s of op:%s failed, parent_index:%d",
             ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), out_node->GetName().c_str(), parent_index);
      return FAILED;
    }
    GELOGI("Generate subgraph output map for subgraph %s, out node index %ld, parent node index %d, node name:%s",
           compute_graph->GetName().c_str(), index, parent_index, out_node->GetName().c_str());
  }

  return SUCCESS;
}

Status AutoMappingSubgraphIndexByDataNode(const ge::ComputeGraphPtr &compute_graph,
                                          const std::function<Status(int data_index, int &parent_input_index)> &input) {
  auto nodes = FindNodesByType(compute_graph, "Data");
  for (size_t i = 0; i < nodes.size(); ++i) {
    int parent_index = -1;
    int index = -1;
    if (!ge::AttrUtils::GetInt(nodes[i]->GetOpDesc(), ge::ATTR_NAME_INDEX, index)) {
      REPORT_INNER_ERROR("E19999", "Get attr:index failed, op_name:%s", nodes[i]->GetName().c_str());
      GELOGE(FAILED, "[Get][Attr] Get attr:index failed, op_name:%s", nodes[i]->GetName().c_str());
      return FAILED;
    }
    GELOGI("Get index %d from data[%zu], node:%s", index, i, nodes[i]->GetName().c_str());
    auto ret = input(index, parent_index);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Get data index failed, op_name:%s", nodes[i]->GetName().c_str());
      GELOGE(FAILED, "[Get][ParentInputIndex] Get data index failed, op_name:%s", nodes[i]->GetName().c_str());
      return FAILED;
    }
    if (!ge::AttrUtils::SetInt(nodes[i]->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_INNER_ERROR("E19999", "Set attr:%s failed, op_name:%s, ",
                         ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), nodes[i]->GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] Set attr:%s failed, op_name:%s, ",
             ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), nodes[i]->GetName().c_str());
      return FAILED;
    }
    GELOGI("Generate subgraph input map for subgraph %s, data index %zu, parent node index %d",
           compute_graph->GetName().c_str(), i, parent_index);
  }
  return SUCCESS;
}
}

Status AutoMappingSubgraphIndexByDataNodeAndOutputNodesInfo(
    const ge::Graph &graph,
    const std::function<Status(int data_index, int &parent_input_index)> &input,
    const std::function<Status(int netoutput_index, int &parent_output_index)> &output) {
  GE_CHECK_NOTNULL(input);
  GE_CHECK_NOTNULL(output);
  auto compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  auto ret = AutoMappingSubgraphIndexByDataNode(compute_graph, input);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Auto mapping graph:%s input index failed,", compute_graph->GetName().c_str());
    GELOGE(ret, "[Mapping][InputIndex] Auto mapping graph:%s input index failed,", compute_graph->GetName().c_str());
    return ret;
  }
  ret = AutoMappingSubgraphIndexByOutputNodesInfo(compute_graph, output);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Auto mapping graph:%s output index failed,", compute_graph->GetName().c_str());
    GELOGE(ret, "[Mapping][OutputIndex] Auto mapping graph:%s output index failed,", compute_graph->GetName().c_str());
    return ret;
  }

  return SUCCESS;
}

domi::Status AutoMappingSubgraphDataFormat(const NodePtr &parent_node, const ge::Graph &graph) {
  GE_CHECK_NOTNULL(parent_node);
  const auto &parent_op_desc = parent_node->GetOpDesc();
  GE_CHECK_NOTNULL(parent_op_desc);
  const auto &compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  const auto data_nodes = FindNodesByType(compute_graph, DATA);
  for (size_t i = 0U; i < data_nodes.size(); ++i) {
    const auto &data_op_desc = data_nodes[i]->GetOpDesc();
    GE_CHECK_NOTNULL(data_op_desc);
    int32_t index = -1;
    // when this function has been called, PARENT_INDEX has not been set
    if (!ge::AttrUtils::GetInt(data_op_desc, ge::ATTR_NAME_INDEX, index)) {
      REPORT_INNER_ERROR("E19999", "Get attr:index failed, op_name:%s", data_nodes[i]->GetName().c_str());
      GELOGE(FAILED, "[Get][Attr] Get attr:index failed, op_name:%s", data_nodes[i]->GetName().c_str());
      return FAILED;
    }
    GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(index) < parent_op_desc->GetAllInputsSize(), PARAM_INVALID,
                           "[Check][Index] failed, index=%d should less than %zu.", index,
                           parent_op_desc->GetAllInputsSize());
    // set data format by node input desc
    const auto input_format = parent_op_desc->GetInputDesc(index).GetFormat();
    const auto input_original_format = parent_op_desc->GetInputDesc(index).GetOriginFormat();
    const auto input_desc = data_op_desc->MutableInputDesc(0U);
    const auto output_desc = data_op_desc->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHECK_NOTNULL(output_desc);
    input_desc->SetFormat(input_format);
    input_desc->SetOriginFormat(input_original_format);
    output_desc->SetFormat(input_format);
    output_desc->SetOriginFormat(input_original_format);
    GELOGD("Set index %d of data[%zu], node:%s, format:%d->%d, original "
           "format:%d->%d, from parent node:%s, node_type:%s",
           index, i, data_nodes[i]->GetName().c_str(),
           static_cast<int32_t>(output_desc->GetFormat()),
           static_cast<int32_t>(input_format),
           static_cast<int32_t>(output_desc->GetOriginFormat()),
           static_cast<int32_t>(input_original_format),
           parent_node->GetName().c_str(), parent_node->GetType().c_str());
  }
  return SUCCESS;
}
}  // namespace ge

namespace domi {
REGISTER_AUTOMAPPING_SUBGRAPH_IO_INDEX_FUNC(ONNX, ge::AutoMappingSubgraphIndexByDataNodeAndOutputNodesInfo);
}  // namespace domi