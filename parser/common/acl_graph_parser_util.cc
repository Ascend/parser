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
#include <regex.h>

#include <cstdlib>
#include <ctime>
#include <fstream>

#include "common/string_util.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_types.h"
#include "ge/ge_api_types.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/type_utils.h"
#include "omg/parser/parser_inner_ctx.h"
#include "parser/common/register_tbe.h"
#include "tbe_plugin_loader.h"

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;
using namespace ge::parser;

namespace {
const std::string kGraphDefaultName = "domi_default";
/// The maximum length of the file.
/// Based on the security coding specification and the current actual (protobuf) model size, it is determined as 2G-1
const int kMaxFileSizeLimit = INT_MAX;
const int kMaxBuffSize = 256;
const int kProtoReadBytesLimit = INT_MAX;    // Max size of 2 GB minus 1 byte.
const int kWarningThreshold = 536870912 * 2; // 536870912 represent 512M

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
    string file_path = ge::parser::RealPath(path.c_str());
    if (file_path.empty()) {
      REPORT_INNER_ERROR("E19999", "File path %s is invalid.", path.c_str());
      GELOGE(ge::FAILED, "[Get][Path] File path %s is invalid.", path.c_str());
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

static void GetAclParams(const std::map<ge::AscendString, ge::AscendString> &parser_params, const string &key,
                         string &value) {
  for (auto &ele : parser_params) {
    const char *key_ascend = ele.first.GetString();
    if (key_ascend == nullptr) {
      GELOGW("Input options key is null, Please check!");
      continue;
    }

    string key_str = key_ascend;
    if (key == key_str) {
      const char *value_ascend = ele.second.GetString();
      if (value_ascend == nullptr) {
        value = "";
      } else {
        value = value_ascend;
      }
      return;
    }
  }
  value = "";
  return;
}

static bool CheckDigitStr(std::string &str) {
  for (char c : str) {
    if (!isdigit(c)) {
      REPORT_CALL_ERROR("E19999", "param str:%s is not positive integer", str.c_str());
      GELOGE(domi::FAILED, "[Check][Param] Value[%s] is not positive integer", str.c_str());
      return false;
    }
  }
  return true;
}
} // namespace

namespace ge {
static bool CheckInputTrueOrFalse(const std::string &s, const std::string &atc_param) {
  if ((s == "true") || (s == "false")) {
    return true;
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage("E10005", {"parameter", "value"}, {atc_param, s});
    GELOGE(PARAM_INVALID, "[Check][Param] Input parameter[%s]'s value[%s] must be true or false.",
           atc_param.c_str(), s.c_str());
    return false;
  }
}

static Status CheckOutNode(ge::OpDescPtr op_desc, int32_t index) {
  int32_t out_size = op_desc->GetOutputsSize();
  if (index < 0 || index >= out_size) {
    GELOGE(domi::FAILED, "[Check][Param]out_node [%s] output index:%d must be smaller "
           "than node output size:%d and can not be negative!", op_desc->GetName().c_str(), index, out_size);
    std::string fail_reason = "output index:" + to_string(index) +
                              " must be smaller than output size:" + to_string(out_size) + " and can not be negative!";
    ErrorManager::GetInstance().ATCReportErrMessage("E10003", {"parameter", "value", "reason"},
                                                    {"out_nodes", op_desc->GetName(), fail_reason});
    return domi::FAILED;
  }
  return domi::SUCCESS;
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
    REPORT_INNER_ERROR("E19999", "OpsProtoManager init failed because ops proto path:%s is invalid.",
                       opsproto_path.c_str());
    GELOGE(FAILED, "[Invoke][Initialize] Load ops_proto lib failed, ops proto path:%s is invalid.",
           opsproto_path.c_str());
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
  TBEPluginLoader::Instance().LoadPluginSo(options);

  // load and save custom op proto for prediction
  (void)LoadOpsProtoLib();
  SaveCustomCaffeProtoPath();

  auto op_registry = domi::OpRegistry::Instance();
  if (op_registry == nullptr) {
    REPORT_CALL_ERROR("E19999", "Call OpRegistry::Instance failed, ret nullptr.");
    GELOGE(FAILED, "[Get][OpRegistry] instance failed");
    return FAILED;
  }

  auto it = options.find(ge::FRAMEWORK_TYPE);
  if (it == options.end()) {
    REPORT_INNER_ERROR("E19999", "Can not find ge.frameworkType in param options");
    GELOGE(FAILED, "[Check][Param]Can not find ge.frameworkType in options");
    return FAILED;
  }
  std::string fmk_type = it->second;
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

void AclGrphParseUtil::SetDefaultFormat() {
  if (ge::GetParserContext().type == domi::TENSORFLOW) {
    ge::GetParserContext().format = domi::DOMI_TENSOR_NHWC;
  } else {
    ge::GetParserContext().format = domi::DOMI_TENSOR_NCHW;
  }
}

domi::Status AclGrphParseUtil::ParseAclOutputNodes(const string &out_nodes) {
  try {
    // parse output node
    if (!out_nodes.empty()) {
      ge::GetParserContext().out_nodes_map.clear();
      ge::GetParserContext().user_out_nodes.clear();
      ge::GetParserContext().user_out_nodes_top_vec.clear();

      vector<string> nodes_v = StringUtils::Split(out_nodes, ';');
      for (const string &node : nodes_v) {
        vector<string> key_value_v = StringUtils::Split(node, ':');
        if (key_value_v.size() != 2) { // The size must be 2.
          if (key_value_v.size() == 1 && ge::GetParserContext().type == domi::CAFFE) {
            ge::GetParserContext().user_out_nodes_top_vec.push_back(node);
            continue;
          }
          ErrorManager::GetInstance().ATCReportErrMessage(
              "E10001", {"parameter", "value", "reason"},
              {"out_nodes", node, "the correct format is \"node_name1:0; node_name1:1; node_name2:0\""});
          GELOGE(PARAM_INVALID, "[Check][Param] The input format of out_nodes is invalid, the correct format is "
                 "\"node_name1:0; node_name1:1; node_name2:0\", while the actual input is %s.",
                 node.c_str());
          return PARAM_INVALID;
        }
        if (!ge::GetParserContext().user_out_nodes_top_vec.empty()) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                          {"out_nodes", out_nodes, "is not all index or top_name"});
          GELOGE(PARAM_INVALID, "[Check][Param] This out_nodes str must be all index or top_name, "
                 "while the actual input is %s", out_nodes.c_str());
          return PARAM_INVALID;
        }
        // stoi: The method may throw an exception: invalid_argument/out_of_range
        if (!CheckDigitStr(key_value_v[1])) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                          {"out_nodes", out_nodes, "is not positive integer"});
          GELOGE(PARAM_INVALID, "[Check][Param] This str:%s must be digit string, while the actual input is %s",
                 key_value_v[1].c_str(), out_nodes.c_str());
          return PARAM_INVALID;
        }

        auto iter = ge::GetParserContext().out_nodes_map.find(key_value_v[0]);
        int32_t index = stoi(StringUtils::Trim(key_value_v[1]));
        GELOGD("Get output info: node[%s] and index[%d]", key_value_v[0].c_str(), index);
        if (iter != ge::GetParserContext().out_nodes_map.end()) {
          iter->second.emplace_back(index);
        } else {
          std::vector<int32_t> index_v;
          index_v.emplace_back(index);
          ge::GetParserContext().out_nodes_map.emplace(key_value_v[0], index_v);
        }
        ge::GetParserContext().user_out_nodes.push_back(std::make_pair(key_value_v[0], index));
      }
    }
  } catch (std::invalid_argument &) {
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid of out_nodes: %s ", out_nodes.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"}, {"out_nodes", out_nodes});
    return PARAM_INVALID;
  } catch (std::out_of_range &) {
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid of out_nodes: %s ", out_nodes.c_str());
    ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"}, {"out_nodes", out_nodes});
    return PARAM_INVALID;
  }
  return SUCCESS;
}

domi::Status AclGrphParseUtil::ParseAclOutputFp16NodesFormat(const string &is_output_fp16) {
  if (is_output_fp16.empty()) {
    return SUCCESS;
  }

  vector<domiTensorFormat_t> &output_formats = ge::GetParserContext().output_formats;
  output_formats.clear();
  vector<string> node_format_vec = StringUtils::Split(is_output_fp16, ',');
  for (auto &is_fp16 : node_format_vec) {
    StringUtils::Trim(is_fp16);
    if (!CheckInputTrueOrFalse(is_fp16, "is_output_adjust_hw_layout")) {
      GELOGE(PARAM_INVALID, "[Check][Param]Invalid Param, is_output_adjust_hw_layout "
             "only support true/false: but is [%s]", is_output_fp16.c_str());
      return PARAM_INVALID;
    }
    if (is_fp16 == "false") {
      output_formats.push_back(DOMI_TENSOR_ND);
    } else if (is_fp16 == "true") {
      output_formats.push_back(domi::DOMI_TENSOR_NC1HWC0);
    }
  }
  return SUCCESS;
}

domi::Status AclGrphParseUtil::ParseAclEnableScope(const string &enable_scope_fusion_passes) {
  ge::GetParserContext().enable_scope_fusion_passes.clear();
  if (enable_scope_fusion_passes.empty()) {
    return SUCCESS;
  }
  ge::GetParserContext().enable_scope_fusion_passes = enable_scope_fusion_passes;
  return SUCCESS;
}

void AclGrphParseUtil::AddAttrsForInputNodes(const vector<string> &adjust_fp16_format_vec,
                                             const string &fp16_nodes_name, uint32_t index, OpDescPtr &op_desc) {
  if (AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_DATATYPE, TypeUtils::DataTypeToSerialString(DT_FLOAT16))) {
    if ((index < adjust_fp16_format_vec.size()) && (adjust_fp16_format_vec[index] == "true")) {
      GELOGI("This node [%s] should be set NC1HWC0", fp16_nodes_name.c_str());
      if (!AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_FORMAT, TypeUtils::FormatToSerialString(FORMAT_NC1HWC0))) {
        GELOGW("This node [%s] set NC1HWC0 failed", fp16_nodes_name.c_str());
      }
    }
  }
}

domi::Status AclGrphParseUtil::ParseAclInputFp16Nodes(const ComputeGraphPtr &graph, const string &input_fp16_nodes,
                                                      const string &is_input_adjust_hw_layout) {
  GE_CHECK_NOTNULL(graph);
  vector<string> adjust_fp16_format_vec;
  if (!is_input_adjust_hw_layout.empty()) {
    adjust_fp16_format_vec = StringUtils::Split(is_input_adjust_hw_layout, ',');
    for (auto &s : adjust_fp16_format_vec) {
      StringUtils::Trim(s);
      if (!CheckInputTrueOrFalse(s, "is_input_adjust_hw_layout")) {
        GELOGE(PARAM_INVALID, "[Check][Param] Invalid Param, is_input_adjust_hw_layout "
               "only support true/false: but is [%s]", is_input_adjust_hw_layout.c_str());
        return PARAM_INVALID;
      }
    }
  }
  if (input_fp16_nodes.empty()) {
    return SUCCESS;
  }
  GELOGI("The input_fp16_nodes is set %s", input_fp16_nodes.c_str());
  vector<string> input_fp16_nodes_vec = StringUtils::Split(input_fp16_nodes, ';');
  for (uint32_t i = 0; i < input_fp16_nodes_vec.size(); ++i) {
    ge::NodePtr node = graph->FindNode(input_fp16_nodes_vec[i]);
    if (node == nullptr) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                      {"input_fp16_nodes", input_fp16_nodes_vec[i]});
      GELOGE(PARAM_INVALID, "[Check][Param] Input parameter[input_fp16_nodes]'s opname[%s] is not exist in model",
             input_fp16_nodes_vec[i].c_str());
      return PARAM_INVALID;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetType() != ge::parser::DATA) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10017", {"parameter", "opname"},
                                                      {"input_fp16_nodes", input_fp16_nodes_vec[i]});
      GELOGE(PARAM_INVALID, "[Check][Param] Input parameter[input_fp16_nodes]'s opname[%s] is not a input opname",
             input_fp16_nodes_vec[i].c_str());
      return PARAM_INVALID;
    }
    AddAttrsForInputNodes(adjust_fp16_format_vec, input_fp16_nodes_vec[i], i, op_desc);
  }
  return SUCCESS;
}

void AclGrphParseUtil::GetOutputNodesNameAndIndex(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                                  std::vector<std::string> &output_nodes_name) {
  output_nodes_name.clear();
  if (ge::GetParserContext().out_top_names.empty()) {
    // tf process, no top name.
    for (const auto output_node_info : output_nodes_info) {
      std::string node_name = output_node_info.first->GetName();
      int32_t index = output_node_info.second;
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
    return;
  }
  // caffe process, need add top name after node_name:index
  for (size_t i = 0; i < output_nodes_info.size(); ++i) {
    std::string node_name = output_nodes_info[i].first->GetName();
    int32_t index = output_nodes_info[i].second;
    if (i < ge::GetParserContext().out_top_names.size()) {
      output_nodes_name.push_back(node_name + ":" + std::to_string(index) + ":" +
                                  ge::GetParserContext().out_top_names[i]);
    } else {
      GELOGW("Get top name of node [%s] fail.", node_name.c_str());
      output_nodes_name.push_back(node_name + ":" + std::to_string(index));
    }
  }
}

domi::Status AclGrphParseUtil::GetOutputLeaf(NodePtr node,
                                             std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  ge::OpDescPtr tmpDescPtr = node->GetOpDesc();
  if (tmpDescPtr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node has no opdesc.");
    GELOGE(domi::FAILED, "[Get][OpDesc] param node has no opdesc.");
    return domi::FAILED;
  }
  size_t size = tmpDescPtr->GetOutputsSize();
  if (node->GetType() != ge::parser::NETOUTPUT) {
    for (size_t index = 0; index < size; ++index) {
      output_nodes_info.push_back(std::make_pair(node, index));
      GELOGD("Get output leaf node:%s.", node->GetName().c_str());
    }
  } else {
    const auto in_anchors = node->GetAllInDataAnchors();
    for (auto in_anchor : in_anchors) {
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      if (out_anchor == nullptr) {
        REPORT_INNER_ERROR("E19999", "Get leaf node op desc fail.");
        GELOGE(domi::FAILED, "[Invoke][GetPeerOutAnchor] Get leaf node op desc fail.");
        return domi::FAILED;
      }
      auto out_node = out_anchor->GetOwnerNode();
      output_nodes_info.push_back(std::make_pair(out_node, out_anchor->GetIdx()));
    }
  }
  return SUCCESS;
}

domi::Status AclGrphParseUtil::GetDefaultOutInfo(ge::ComputeGraphPtr &compute_graph,
                                                 std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info) {
  std::vector<std::pair<std::string, int32_t>> default_out_nodes = ge::GetParserContext().default_out_nodes;
  if (ge::GetParserContext().type == domi::CAFFE && !default_out_nodes.empty()) {
    for (uint32_t i = 0; i < default_out_nodes.size(); ++i) {
      ge::NodePtr out_node = compute_graph->FindNode(default_out_nodes[i].first);
      if (out_node == nullptr) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                        {"out_nodes", default_out_nodes[i].first});
        GELOGE(domi::FAILED, "[Check][Param] Can not find out_nodes(%d) (%s) in graph.",
               i, default_out_nodes[i].first.c_str());
        return domi::FAILED;
      }
      output_nodes_info.push_back(std::make_pair(out_node, default_out_nodes[i].second));
      GELOGD("Get default output node:%s.", out_node->GetName().c_str());
    }
    return domi::SUCCESS;
  }

  for (ge::NodePtr node : compute_graph->GetDirectNode()) {
    if (!node->GetInAllNodes().empty() && node->GetOutAllNodes().empty()) {
      Status ret = GetOutputLeaf(node, output_nodes_info);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "[Invoke][GetOutputLeaf] Find leaf fail.");
    }
  }
  return domi::SUCCESS;
}

domi::Status AclGrphParseUtil::SetOutputNodeInfo(ge::Graph &graph,
                                                 const std::map<AscendString, AscendString> &parser_params) {
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  std::vector<std::pair<std::string, int32_t>> user_out_nodes = ge::GetParserContext().user_out_nodes;
  std::vector<domiTensorFormat_t> output_formats = ge::GetParserContext().output_formats;
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  std::vector<std::string> output_nodes_name;

  // User declared outputs
  for (uint32_t i = 0; i < user_out_nodes.size(); ++i) {
    ge::NodePtr out_node = compute_graph->FindNode(user_out_nodes[i].first);
    if (out_node == nullptr) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                      {"out_nodes", user_out_nodes[i].first});
      GELOGE(domi::FAILED, "[Check][Param] Can not find out_nodes(%d) (%s) in graph.",
             i, user_out_nodes[i].first.c_str());
      return domi::FAILED;
    }
    auto op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (CheckOutNode(op_desc, user_out_nodes[i].second) != SUCCESS) {
      GELOGE(domi::FAILED, "[CheckOut][Node] (%s) fail.", user_out_nodes[i].first.c_str());
      return domi::FAILED;
    }

    // add user_define_output_nodes attr.
    (void)ge::AttrUtils::SetStr(op_desc, ATTR_ATC_USER_DEFINE_OUTPUT_NODES, "true");

    if (i < output_formats.size()) {
      if (output_formats[i] == domi::DOMI_TENSOR_NC1HWC0) {
        GELOGI("The output node [%s] should be set NC1HWC0", user_out_nodes[i].first.c_str());
        vector<string> output_fp16_5hd_vec;
        (void)ge::AttrUtils::GetListStr(op_desc, "_user_defined_output_fp16_5hd", output_fp16_5hd_vec);
        output_fp16_5hd_vec.push_back(std::to_string(user_out_nodes[i].second) + ":" + "NC1HWC0");
        (void)ge::AttrUtils::SetListStr(op_desc, "_user_defined_output_fp16_5hd", output_fp16_5hd_vec);
      }
    }
    output_nodes_info.push_back(std::make_pair(out_node, user_out_nodes[i].second));
  }
  // default output node (leaf)
  if (user_out_nodes.empty()) {
    if (GetDefaultOutInfo(compute_graph, output_nodes_info) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "GetDefaultOutInfo failed for graph:%s", graph.GetName().c_str());
      GELOGE(domi::FAILED, "[Invoke][GetDefaultOutInfo] failed, graph:%s.", graph.GetName().c_str());
      return domi::FAILED;
    }
  }
  GetOutputNodesNameAndIndex(output_nodes_info, output_nodes_name);
  compute_graph->SetGraphOutNodesInfo(output_nodes_info);
  ge::GetParserContext().net_out_nodes = output_nodes_name;
  GELOGI("Set graph %s output node success.", graph.GetName().c_str());
  return domi::SUCCESS;
}

domi::Status AclGrphParseUtil::CheckOptions(const std::map<AscendString, AscendString> &parser_params) {
  for (auto &ele : parser_params) {
    const char *key_ascend = ele.first.GetString();
    if (key_ascend == nullptr) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"},
                                                      {"parser_params", "null AscendString"});
      GELOGE(PARAM_INVALID, "[Check][Param] Input options key is null, Please check!");
      return PARAM_INVALID;
    }

    string key_str = key_ascend;
    auto it = ge::ir_option::ir_parser_suppported_options.find(key_str);
    if (it == ge::ir_option::ir_parser_suppported_options.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10016", {"parameter", "opname"}, {"parser_params", key_str});
      GELOGE(PARAM_INVALID, "[Check][Param] Input options include unsupported option(%s).Please check!", key_ascend);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

domi::Status AclGrphParseUtil::ParseParamsBeforeGraph(const std::map<AscendString, AscendString> &parser_params,
                                                      string &graph_name) {
  GELOGI("Parse graph user options start.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(CheckOptions(parser_params) != SUCCESS,
                                 return PARAM_INVALID, "[Check][Options] Parse paragrams invalid, graph:%s.",
                                 graph_name.c_str());
  // support paragrams: out_nodes, is_output_adjust_hw_layout, output, enable_scope_fusion_passes
  SetDefaultFormat();

  string out_nodes;
  GetAclParams(parser_params, ge::ir_option::OUT_NODES, out_nodes);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ParseAclOutputNodes(out_nodes) != SUCCESS,
                                 return PARAM_INVALID,
                                 "[Invoke][ParseAclOutputNodes] Parse out_nodes failed, graph:%s.", graph_name.c_str());

  string is_output_adjust_hw_layout;
  GetAclParams(parser_params, ge::ir_option::IS_OUTPUT_ADJUST_HW_LAYOUT, is_output_adjust_hw_layout);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ParseAclOutputFp16NodesFormat(is_output_adjust_hw_layout) != SUCCESS,
                                 return PARAM_INVALID,
                                 "[Invoke][ParseAclOutputFp16NodesFormat] Parse is_output_adjust_hw_layout failed, "
                                 "graph:%s.", graph_name.c_str());

  string tmp_name;
  GetAclParams(parser_params, ge::ir_option::OUTPUT, tmp_name);
  graph_name = tmp_name.empty() ? (kGraphDefaultName + "_" + ge::parser::CurrentTimeInStr()) : tmp_name;

  string enable_scope_fusion_passes;
  GetAclParams(parser_params, ge::ir_option::ENABLE_SCOPE_FUSION_PASSES, enable_scope_fusion_passes);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ParseAclEnableScope(enable_scope_fusion_passes) != SUCCESS,
                                 return PARAM_INVALID,
                                 "[Invoke][ParseAclEnableScope] Parse enable_scope_fusion_passes failed, graph:%s.",
                                 graph_name.c_str());

  return SUCCESS;
}

domi::Status AclGrphParseUtil::ParseParamsAfterGraph(ge::Graph &graph,
                                                     const std::map<AscendString, AscendString> &parser_params) {
  // support paragrams: input_fp16_nodes, is_input_adjust_hw_layout,
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  string input_fp16_nodes;
  GetAclParams(parser_params, ge::ir_option::INPUT_FP16_NODES, input_fp16_nodes);

  string is_input_adjust_hw_layout;
  GetAclParams(parser_params, ge::ir_option::IS_INPUT_ADJUST_HW_LAYOUT, is_input_adjust_hw_layout);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      ParseAclInputFp16Nodes(compute_graph, input_fp16_nodes, is_input_adjust_hw_layout) != SUCCESS,
      return PARAM_INVALID, "[Invoke][ParseAclInputFp16Nodes] Parse input_fp16_nodes failed, graph:%s",
      compute_graph->GetName().c_str());

  return SUCCESS;
}

namespace parser {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string RealPath(const char *path) {
  if (path == nullptr) {
    GELOGE(ge::FAILED, "path pointer is NULL.");
    return "";
  }
  if (strlen(path) >= PATH_MAX) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"}, {path, std::to_string(PATH_MAX)});
    GELOGE(ge::FAILED, "[Check][Param] Path[%s] len is too long, it must be less than %d", path, PATH_MAX);
    return "";
  }
  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  std::string res;
  char resolved_path[PATH_MAX] = {0};
  if (realpath(path, resolved_path) != nullptr) {
    res = resolved_path;
  }

  return res;
}

// Get file length
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY long GetFileLength(const std::string &input_file) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(input_file.empty(),
                                 REPORT_INNER_ERROR("E19999", "input_file path is null, check invalid.");
                                 return -1, "[Check][Param] input_file path is null.");

  std::string real_path = RealPath(input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(),
                                 REPORT_INPUT_ERROR("E19000", std::vector<std::string>({"path", "errmsg"}),
                                                    std::vector<std::string>({real_path, strerror(errno)}));
                                 return -1, "[Get][Path] input_file path '%s' not valid", input_file.c_str());
  unsigned long long file_length = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mmGetFileSize(input_file.c_str(), &file_length) != EN_OK,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"},
                                                                                 {input_file, strerror(errno)});
                                 return -1, "[Open][File] [%s] failed. %s", input_file.c_str(), strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_length == 0 || file_length > kMaxFileSizeLimit),
                                 REPORT_INPUT_ERROR(
                                     "E19015", std::vector<std::string>({"file", "size", "maxsize"}),
                                     std::vector<std::string>({input_file, std::to_string(file_length),
                                                              std::to_string(kMaxFileSizeLimit)}));
                                 return -1, "[Check][Param] File[%s] size %lld is out of range(0,%d).",
                                 input_file.c_str(), file_length, kMaxFileSizeLimit);
  return static_cast<long>(file_length);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY uint64_t GetCurrentTimestamp() {
  struct timeval tv{};
  int ret = gettimeofday(&tv, nullptr);
  GE_LOGE_IF(ret != 0, "[Func][GetTimeOfDay] may failed: ret=%d", ret);
  auto total_use_time = tv.tv_usec + tv.tv_sec * 1000000;  // 1000000: seconds to microseconds
  return static_cast<uint64_t>(total_use_time);
}

static bool ReadProtoFromCodedInputStream(CodedInputStream &coded_stream, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(proto == nullptr,
                                 REPORT_INNER_ERROR("E19999", "param proto is nullptr, check invalid");
                                 return false, "[Check][Param] incorrect parameter. nullptr == proto");

  coded_stream.SetTotalBytesLimit(kProtoReadBytesLimit, kWarningThreshold);
  return proto->ParseFromCodedStream(&coded_stream);
}

/** @ingroup domi_common
 *  @brief Read all data from binary file
 *  @param [in] file_name  File path
 *  @param [out] buffer  The address of the output memory, which needs to be released by the caller
 *  @param [out] length  Output memory size
 *  @return false fail
 *  @return true success
 */
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadBytesFromBinaryFile(const char *file_name, char **buffer,
                                                                              int &length) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_name == nullptr),
                                 REPORT_INNER_ERROR("E19999", "param file_name is nullptr, check invalid");
                                 return false, "[Check][Param] incorrect parameter. file is nullptr");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((buffer == nullptr),
                                 REPORT_INNER_ERROR("E19999", "param buffer is nullptr, check invalid");
                                 return false, "[Check][Param] incorrect parameter. buffer is nullptr");

  std::string real_path = RealPath(file_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(),
                                 REPORT_INNER_ERROR("E19999", "file path '%s' not valid, realpath failed", file_name);
                                 return false, "[Check][Param]file path '%s' not valid, realpath failed", file_name);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    REPORT_INNER_ERROR("E19999", "read file %s failed", file_name);
    GELOGE(ge::FAILED, "[Read][File] %s failed.", file_name);
    return false;
  }

  length = static_cast<int>(file.tellg());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((length <= 0), file.close(); REPORT_INNER_ERROR("E19999", "file length <= 0");
                                 return false, "[Check][Param] file length <= 0");

  file.seekg(0, std::ios::beg);

  *buffer = new(std::nothrow) char[length]();
  GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(*buffer == nullptr, false, file.close();
                                   REPORT_CALL_ERROR("E19999", "new an object failed."),
                                   "[Create][Buffer] new an object failed.");

  file.read(*buffer, length);
  file.close();
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromBinaryFile(const char *file, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || proto == nullptr),
                                 REPORT_INNER_ERROR("E19999", "param file or proto is nullptr, check invalid");
                                 return false, "[Check][Param] Input parameter file or proto is nullptr!");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(),
                                 REPORT_INNER_ERROR("E19999", "file path '%s' not valid, realpath failed", file);
                                 return false, "[Check][Param]pb file path '%s' not valid, realpath failed", file);

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "[Get][FileLength]file size not valid.");

  std::ifstream fs(real_path, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file, "ifstream is_open failed"});
    GELOGE(ge::FAILED, "[Open][RealPath][%s] failed.", file);
    return false;
  }

  google::protobuf::io::IstreamInputStream istream(&fs);
  google::protobuf::io::CodedInputStream coded_stream(&istream);

  bool ret = ReadProtoFromCodedInputStream(coded_stream, proto);

  fs.close();

  if (!ret) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19005", {"file"}, {file});
    GELOGE(ge::FAILED, "[Read][Proto] Parse file[%s] failed.", file);
    return ret;
  }

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromArray(const void *data, int size, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((proto == nullptr || data == nullptr || size == 0),
                                 REPORT_INNER_ERROR("E19999", "param proto or data is nullptr "
                                                    "or size is 0, check invalid"); return false,
                                 "[Check][Param]incorrect parameter. proto is nullptr || data is nullptr || size is 0");

  google::protobuf::io::CodedInputStream coded_stream(reinterpret_cast<uint8_t *>(const_cast<void *>(data)), size);
  return ReadProtoFromCodedInputStream(coded_stream, proto);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromText(const char *file,
                                                                        google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || message == nullptr),
                                 REPORT_INNER_ERROR("E19999", "param file or message is nullptr, check invalid");
                                 return false,
                                 "[Check][Param]incorrect parameter. nullptr == file || nullptr == message");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(),
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19000", {"path", "errmsg"},
                                                                                 {file, strerror(errno)});
                                 return false, "[Check][Param]Path[%s]'s realpath is empty, errmsg[%s]", file,
                                 strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "[Check][Param] file size not valid.");

  std::ifstream fs(real_path.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    REPORT_INNER_ERROR("E19999", "open file:%s failed", real_path.c_str());
    GELOGE(ge::FAILED, "[Open][ProtoFile] failed, real path is '%s' when orginal file path is '%s'.",
           real_path.c_str(), file);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(!ret, ErrorManager::GetInstance().ATCReportErrMessage("E19018", {"protofile"}, {file});
                  GELOGE(ret, "[Parse][File] [%s] through [google::protobuf::TextFormat::Parse] failed, "
                         "please check whether the file is a valid protobuf format file.", file));
  fs.close();

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromMem(const char *data, int size,
                                                                       google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((data == nullptr || message == nullptr),
                                 REPORT_INNER_ERROR("E19999", "param data or message is nullptr,check invalid");
                                 return false,
                                 "[Check][Param] incorrect parameter. data is nullptr || message is nullptr");
  std::string str(data, static_cast<size_t>(size));
  std::istringstream fs(str);

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(!ret, REPORT_CALL_ERROR("E19999", "parse failed, please check your text file.");
                  GELOGE(ret, "[Call][Parse] ret fail, please check your text file."));

  return ret;
}

///
/// @brief get the Original Type of FrameworkOp
/// @param [in] node
/// @param [out] type
/// @return Status
///
Status GetOriginalType(const ge::NodePtr &node, string &type) {
  GE_CHECK_NOTNULL(node);
  type = node->GetType();
  GE_IF_BOOL_EXEC(type != FRAMEWORKOP, return SUCCESS);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  bool ret = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  if (!ret) {
    REPORT_CALL_ERROR("E19999", "Get FrameWorkOp original type [%s] from node:%s failed.",
                      type.c_str(), node->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Invoke][GetStr] Get FrameWorkOp original type [%s] from node:%s failed",
           type.c_str(), node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Get FrameWorkOp original type [%s]", type.c_str());
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY bool ValidateStr(const std::string &str, const std::string &mode) {
  char ebuff[kMaxBuffSize];
  regex_t reg;
  int cflags = REG_EXTENDED | REG_NOSUB;
  int ret = regcomp(&reg, mode.c_str(), cflags);
  if (ret) {
    regerror(ret, &reg, ebuff, kMaxBuffSize);
    GELOGW("regcomp failed, reason: %s", ebuff);
    regfree(&reg);
    return true;
  }

  ret = regexec(&reg, str.c_str(), 0, nullptr, 0);
  if (ret) {
    regerror(ret, &reg, ebuff, kMaxBuffSize);
    GELOGE(ge::PARAM_INVALID, "[Invoke][RegExec] failed, reason: %s", ebuff);
    regfree(&reg);
    return false;
  }

  regfree(&reg);
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::string CurrentTimeInStr() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (ptm == nullptr) {
    GELOGE(ge::FAILED, "[Invoke][LocalTime] failed.");
    return "";
  }

  const int kTimeBufferLen = 32;
  char buffer[kTimeBufferLen + 1] = {0};
  // format: 20171122042550
  std::strftime(buffer, kTimeBufferLen, "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}
}  // namespace parser
}  // namespace ge
