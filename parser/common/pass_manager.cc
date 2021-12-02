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

#include "common/util.h"
#include "parser/common/pass_manager.h"
#include "framework/omg/parser/parser_types.h"
#include "parser/common/acl_graph_parser_util.h"
#include "graph/utils/node_utils.h"
#include "omg/omg_inner_types.h"

namespace ge {
namespace parser {
const std::vector<std::pair<std::string, GraphPass *>> &PassManager::GraphPasses() const {
  return names_to_graph_passes_;
}

Status PassManager::AddPass(const string &pass_name, GraphPass *const pass) {
  GE_CHECK_NOTNULL(pass);
  names_to_graph_passes_.emplace_back(pass_name, pass);
  return SUCCESS;
}

Status PassManager::Run(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  return Run(graph, names_to_graph_passes_);
}

Status PassManager::Run(const ComputeGraphPtr &graph,
                        std::vector<std::pair<std::string, GraphPass *>> &names_to_passes) {
  GE_CHECK_NOTNULL(graph);
  bool not_changed = true;

  for (auto &pass_pair : names_to_passes) {
    const auto &pass = pass_pair.second;
    const auto &pass_name = pass_pair.first;
    GE_CHECK_NOTNULL(pass);

    PARSER_TIMESTAMP_START(PassRun);
    Status status = pass->Run(graph);
    if (status == SUCCESS) {
      not_changed = false;
    } else if (status != NOT_CHANGED) {
      GELOGE(status, "Pass Run failed on graph %s", graph->GetName().c_str());
      return status;
    }
    for (const auto &subgraph :graph->GetAllSubgraphs()) {
      GE_CHECK_NOTNULL(subgraph);
      GE_CHK_STATUS_RET(pass->ClearStatus(), "[Invoke][ClearStatus]pass clear status failed for subgraph %s",
                        subgraph->GetName().c_str());
      string subgraph_pass_name = pass_name + "::" + graph->GetName();
      PARSER_TIMESTAMP_START(PassRunSubgraph);
      status = pass->Run(subgraph);
      PARSER_TIMESTAMP_END(PassRunSubgraph, subgraph_pass_name.c_str());
      if (status == SUCCESS) {
        not_changed = false;
      } else if (status != NOT_CHANGED) {
        REPORT_CALL_ERROR("E19999", "Pass Run failed on subgraph %s, pass name:%s",
                          subgraph->GetName().c_str(), subgraph_pass_name.c_str());
        GELOGE(status, "[Invoke][Run]Pass Run failed on subgraph %s, pass name:%s",
               subgraph->GetName().c_str(), subgraph_pass_name.c_str());
        return status;
      }
    }
    PARSER_TIMESTAMP_END(PassRun, pass_name.c_str());
  }

  return not_changed ? NOT_CHANGED : SUCCESS;
}

PassManager::~PassManager() {
  for (auto &pass_pair : names_to_graph_passes_) {
    auto &pass = pass_pair.second;
    GE_DELETE_NEW_SINGLE(pass);
  }
}
}  // namespace parser
}  // namespace ge
