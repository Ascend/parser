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
#include <fstream>
#include <regex.h>
#include <ctime>

#include "common/string_util.h"
#include "common/debug/log.h"
#include "common/op/ge_op_utils.h"
#include "ge/ge_api_types.h"
#include "graph/opsproto_manager.h"
#include "omg/parser/parser_inner_ctx.h"
#include "tbe_plugin_loader.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/register_tbe.h"
#include "framework/omg/parser/parser_types.h"
#include "common/util/error_manager/error_manager.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;
using namespace ge::parser;

namespace {
/// The maximum length of the file.
/// Based on the security coding specification and the current actual (protobuf) model size, it is determined as 2G-1
const int kMaxFileSizeLimit = INT_MAX;
const int kMaxBuffSize = 256;
const int kProtoReadBytesLimit = INT_MAX;     // Max size of 2 GB minus 1 byte.
const int kWarningThreshold = 536870912 * 2;  // 536870912 represent 512M

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
  if (ge::GetParserContext().out_top_names.empty()) {
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
  ge::GetParserContext().net_out_nodes = output_nodes_name;
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
  TBEPluginLoader::Instance().LoadPluginSo(options);

  // load and save custom op proto for prediction
  (void)LoadOpsProtoLib();
  SaveCustomCaffeProtoPath();

  auto op_registry = domi::OpRegistry::Instance();
  if (op_registry == nullptr) {
    GELOGE(FAILED, "Get OpRegistry instance failed");
    return FAILED;
  }

  std::vector<OpRegistrationData> registrationDatas = op_registry->registrationDatas;
  GELOGI("The size of registrationDatas in parser is: %zu", registrationDatas.size());
  for (OpRegistrationData &reg_data : registrationDatas) {
    (void)OpRegistrationTbe::Instance()->Finalize(reg_data, false);
    domi::OpRegistry::Instance()->Register(reg_data);
  }

  // set init status
  if (!parser_initialized) {
    // Initialize success, first time calling initialize
    parser_initialized = true;
  }

  GELOGT(TRACE_STOP, "AclParserInitialize finished");
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
    GELOGE(ge::FAILED, "Path[%s] len is too long, it must be less than %d", path, PATH_MAX);
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
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(input_file.empty(), return -1, "input_file path is null.");

  std::string real_path = RealPath(input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return -1, "input_file path '%s' not valid", input_file.c_str());
  unsigned long long file_length = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(mmGetFileSize(input_file.c_str(), &file_length) != EN_OK,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"},
                                                                                 {input_file, strerror(errno)});
                                         return -1, "Open file[%s] failed. %s", input_file.c_str(), strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_length == 0),
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19015", {"filepath"}, {input_file});
                                         return -1, "File[%s] size is 0, not valid.", input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(file_length > kMaxFileSizeLimit,
                                 ErrorManager::GetInstance().ATCReportErrMessage(
                                         "E19016", {"filepath", "filesize", "maxlen"},
                                         {input_file, std::to_string(file_length), std::to_string(kMaxFileSizeLimit)});
                                         return -1, "File[%s] size %lld is out of limit: %d.",
                                 input_file.c_str(), file_length, kMaxFileSizeLimit);
  return static_cast<long>(file_length);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY uint64_t GetCurrentTimestamp() {
  struct timeval tv{};
  int ret = gettimeofday(&tv, nullptr);
  GE_LOGE_IF(ret != 0, "Func gettimeofday may failed: ret=%d", ret);
  auto total_use_time = tv.tv_usec + tv.tv_sec * 1000000;  // 1000000: seconds to microseconds
  return static_cast<uint64_t>(total_use_time);
}

static bool ReadProtoFromCodedInputStream(CodedInputStream &coded_stream, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(proto == nullptr,
                                 return false, "incorrect parameter. nullptr == proto");

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
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_name == nullptr), return false, "incorrect parameter. file is nullptr");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((buffer == nullptr), return false, "incorrect parameter. buffer is nullptr");

  std::string real_path = RealPath(file_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return false, "file path '%s' not valid", file_name);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "Read file %s failed.", file_name);
    return false;
  }

  length = static_cast<int>(file.tellg());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((length <= 0), file.close(); return false, "file length <= 0");

  file.seekg(0, std::ios::beg);

  *buffer = new(std::nothrow) char[length]();
  GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(*buffer == nullptr, false, file.close(), "new an object failed.");

  file.read(*buffer, length);
  file.close();
  return true;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromBinaryFile(const char *file, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || proto == nullptr),
                                 return false,
                                 "Input parameter file or proto is nullptr!");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(),
                                 return false, "pb file path '%s' not valid", file);

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "file size not valid.");

  std::ifstream fs(real_path, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file, "ifstream is_open failed"});
    GELOGE(ge::FAILED, "Open real path[%s] failed.", file);
    return false;
  }

  google::protobuf::io::IstreamInputStream istream(&fs);
  google::protobuf::io::CodedInputStream coded_stream(&istream);

  bool ret = ReadProtoFromCodedInputStream(coded_stream, proto);

  fs.close();

  if (!ret) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19005", {"file"}, {file});
    GELOGE(ge::FAILED, "Parse file[%s] failed.", file);
    return ret;
  }

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromArray(const void *data, int size, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((proto == nullptr || data == nullptr || size == 0), return false,
                                 "incorrect parameter. proto is nullptr || data is nullptr || size is 0");

  google::protobuf::io::CodedInputStream coded_stream(reinterpret_cast<uint8_t *>(const_cast<void *>(data)), size);
  return ReadProtoFromCodedInputStream(coded_stream, proto);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromText(const char *file,
                                                                        google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || message == nullptr), return false,
                                 "incorrect parameter. nullptr == file || nullptr == message");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(),
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19000", {"path", "errmsg"},
                                                                                 {file, strerror(errno)});
                                         return false, "Path[%s]'s realpath is empty, errmsg[%s]", file,
                                 strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "file size not valid.");

  std::ifstream fs(real_path.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19017", {"realpth", "protofile"}, {real_path, file});
    GELOGE(ge::FAILED,
           "Fail to open proto file real path is '%s' when orginal file path is '%s'.", real_path.c_str(), file);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(!ret,
                  ErrorManager::GetInstance().ATCReportErrMessage("E19018", {"protofile"}, {file});
                          GELOGE(ret, "Parse file[%s] through [google::protobuf::TextFormat::Parse] failed, "
                                      "please check whether the file is a valid protobuf format file.", file));
  fs.close();

  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY bool ReadProtoFromMem(const char *data, int size,
                                                                       google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((data == nullptr || message == nullptr), return false,
                                 "incorrect parameter. data is nullptr || message is nullptr");
  std::string str(data, static_cast<size_t>(size));
  std::istringstream fs(str);

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(
          !ret, GELOGE(ret, "Call [google::protobuf::TextFormat::Parse] func ret fail, please check your text file."));

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
    GELOGE(INTERNAL_ERROR, "Get FrameWorkOp original type [%s]", type.c_str());
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
    GELOGE(ge::PARAM_INVALID, "regexec failed, reason: %s", ebuff);
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
    GELOGE(ge::FAILED, "Localtime failed.");
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
