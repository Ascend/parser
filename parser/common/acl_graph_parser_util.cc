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

#include "parser/common/acl_graph_parser_util.h"

#include <dlfcn.h>
#include <cstdlib>
#include "common/string_util.h"
#include "common/types.h"
#include "common/debug/log.h"
#include "common/ge/tbe_plugin_manager.h"
#include "common/op/ge_op_utils.h"
#include "common/util.h"

#include "ge/ge_api_types.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/type_utils.h"
#include "omg/parser/parser_inner_ctx.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/register_tbe.h"

namespace {
static string GetSoPath() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(&GetSoPath), &dl_info) == 0) {
    GELOGW("Failed to read so_path!");
    return string();
  } else {
    std::string so_path = dl_info.dli_fname;
    char path[PATH_MAX] = {0};
    if (so_path.length() >= PATH_MAX) {
      GELOGW("File path is too long!");
      return string();
    }
    if (realpath(so_path.c_str(), path) == nullptr) {
      GELOGW("Failed to get realpath of %s", so_path.c_str());
      return string();
    }

    so_path = path;
    so_path = so_path.substr(0, so_path.rfind('/') + 1);
    return so_path;
  }
}

static void GetOpsProtoPath(string &opsproto_path) {
  GELOGD("Start to get ops proto path schedule.");
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    string path = path_env;
    string file_path = ge::RealPath(path.c_str());
    if (file_path.empty()) {
      GELOGE(ge::FAILED, "File path %s is invalid.", path.c_str());
      return;
    }
    opsproto_path = (path + "/op_proto/custom/" + ":") + (path + "/op_proto/built-in/");
    GELOGI("Get opsproto so path from env : %s", path.c_str());
    return;
  }
  string path_base = GetSoPath();
  GELOGI("path_base is %s", path_base.c_str());
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  opsproto_path = (path_base + "ops/op_proto/custom/" + ":") + (path_base + "ops/op_proto/built-in/");
}
}  // namespace

namespace ge {
domi::Status AclGrphParseUtil::GetOutputLeaf(NodePtr node,
                                             std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  ge::OpDescPtr tmpDescPtr = node->GetOpDesc();
  if (tmpDescPtr == nullptr) {
    GELOGE(domi::FAILED, "Get outnode op desc fail.");
    return domi::FAILED;
  }
  size_t size = tmpDescPtr->GetOutputsSize();
  if (node->GetType() != NETOUTPUT) {
    for (size_t index = 0; index < size; ++index) {
      output_nodes_info.push_back(std::make_pair(node, index));
    }
  } else {
    const auto in_anchors = node->GetAllInDataAnchors();
    for (auto in_anchor : in_anchors) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr) {
        GELOGE(domi::FAILED, "Get leaf node op desc fail.");
        return domi::FAILED;
      }
      auto out_node = out_anchor->GetOwnerNode();
      output_nodes_info.push_back(std::make_pair(out_node, out_anchor->GetIdx()));
    }
  }
  return SUCCESS;
}

void AclGrphParseUtil::GetOutputNodesNameAndIndex(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                                  std::vector<std::string> &output_nodes_name) {
  output_nodes_name.clear();
  if (domi::GetContext().out_top_names.empty()) {
    // tf process, no top name.
    for (const auto output_node_info : output_nodes_info) {
      std::string node_name = output_node_info.first->GetName();
      int32_t index = output_node_info.second;
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
    return;
  }
  // caffe process reserved place;
}

domi::Status AclGrphParseUtil::SetDefaultOutputNode(ge::Graph &graph) {
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(FAILED, "compute_graph is nullptr.");
    return FAILED;
  }

  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  std::vector<std::string> output_nodes_name;

  for (ge::NodePtr node : compute_graph->GetDirectNode()) {
    if (!node->GetInAllNodes().empty() && node->GetOutAllNodes().empty()) {
      Status ret = AclGrphParseUtil::GetOutputLeaf(node, output_nodes_info);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "find leaf fail.");
        return FAILED;
      }
    }
  }

  AclGrphParseUtil::GetOutputNodesNameAndIndex(output_nodes_info, output_nodes_name);
  compute_graph->SetGraphOutNodesInfo(output_nodes_info);
  domi::GetContext().net_out_nodes = output_nodes_name;
  GELOGI("Set graph %s default output node success.", graph.GetName().c_str());
  return SUCCESS;
}

domi::Status AclGrphParseUtil::LoadOpsProtoLib() {
  string opsproto_path;
  GetOpsProtoPath(opsproto_path);
  GELOGI("Get opsproto path is %s", opsproto_path.c_str());
  OpsProtoManager *manager = OpsProtoManager::Instance();
  map<string, string> option_tmp;
  option_tmp.emplace(std::pair<string, string>(string("ge.opsProtoLibPath"), opsproto_path));
  bool is_proto_init = manager->Initialize(option_tmp);
  if (!is_proto_init) {
    GELOGE(FAILED, "Load ops_proto lib failed, ops proto path is invalid.");
    return FAILED;
  }
  return SUCCESS;
}

void AclGrphParseUtil::SaveCustomCaffeProtoPath() {
  GELOGD("Enter save custom caffe proto path.");
  std::string path_base = GetSoPath();
  path_base = path_base.substr(0, path_base.rfind('/'));
  path_base = path_base.substr(0, path_base.rfind('/') + 1);
  ge::GetParserContext().caffe_proto_path = path_base + "include/proto/";

  string custom_op_path;
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  if (path_env != nullptr) {
    std::string path = path_env;
    custom_op_path = path + "/framework/custom/caffe/";
    GELOGI("Get custom proto path from env : %s", path_env);
    GetParserContext().custom_proto_path = custom_op_path;
    return;
  }
  custom_op_path = path_base + "ops/framework/custom/caffe/";
  ge::GetParserContext().custom_proto_path = custom_op_path;
  return;
}

// Initialize PARSER, load custom op plugin
// options will be used later for parser decoupling
domi::Status AclGrphParseUtil::AclParserInitialize(const std::map<std::string, std::string> &options) {
  GELOGT(TRACE_INIT, "AclParserInitialize start");
  // check init status
  if (parser_initialized) {
    GELOGW("AclParserInitialize is called more than once");
    return SUCCESS;
  }

  // load custom op plugin
  TBEPluginManager::Instance().LoadPluginSo(options);

  // load and save custom op proto for prediction
  (void)LoadOpsProtoLib();
  SaveCustomCaffeProtoPath();

  auto op_registry = domi::OpRegistry::Instance();
  if (op_registry == nullptr) {
    GELOGE(FAILED, "Get OpRegistry instance failed");
    return FAILED;
  }

  auto it = options.find(ge::FRAMEWORK_TYPE);
  if (it == options.end()) {
    GELOGE(FAILED, "Can not find ge.frameworkType in options");
    return FAILED;
  }
  std::string fmk_type = it->second;
  GELOGD("frameworkType is %s", fmk_type.c_str());
  std::vector<OpRegistrationData> registrationDatas = op_registry->registrationDatas;
  GELOGI("The size of registrationDatas in parser is: %zu", registrationDatas.size());
  for (OpRegistrationData &reg_data : registrationDatas) {
    if (std::to_string(reg_data.GetFrameworkType()) == fmk_type) {
      (void)OpRegistrationTbe::Instance()->Finalize(reg_data, false);
      (void)domi::OpRegistry::Instance()->Register(reg_data);
    }
  }

  // set init status
  if (!parser_initialized) {
    // Initialize success, first time calling initialize
    parser_initialized = true;
  }

  GELOGT(TRACE_STOP, "AclParserInitialize finished");
  return SUCCESS;
}
}  // namespace ge
