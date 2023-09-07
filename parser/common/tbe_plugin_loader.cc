/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include "tbe_plugin_loader.h"

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <typeinfo>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <regex>

#include "external/ge/ge_api_types.h"
#include "common/plugin/plugin_manager.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/string_util.h"
#include "framework/common/util.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/utils/type_utils.h"
#include "parser/common/acl_graph_parser_util.h"
#include "mmpa/mmpa_api.h"
#include "common/checker.h"

namespace ge {
namespace {
const char_t *const kOppEnvName = "ASCEND_OPP_PATH";
const char_t *const kBuiltIn = "built-in";     // opp built-in directory name
const char_t *const kVendors = "vendors";      // opp vendors directory name
const char_t *const kConfig = "config.ini";    // opp vendors config file name
const size_t kVendorConfigPartsCount = 2U;
const char_t *const kLibRegisterSo = "libregister.so";
}  // namespace
std::map<string, string> TBEPluginLoader::options_ = {};

// Get Singleton Instance
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY TBEPluginLoader &TBEPluginLoader::Instance() {
  static TBEPluginLoader instance_ptr_;
  return instance_ptr_;
}

Status TBEPluginLoader::ClearHandles_() {
  Status ret = SUCCESS;
  const auto close_func = [&ret](void *handle) -> void {
    if (mmDlclose(handle) != 0) {
      ret = FAILED;
      GELOGW("Failed to close handle: %s", dlerror());
    }
  };

  for (const auto &handle : handles_vec_) {
    close_func(handle);
  }
  handles_vec_.clear();

  if (handle_reg_ != nullptr) {
    close_func(handle_reg_);
  }
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status TBEPluginLoader::Finalize() {
  Status ret = ClearHandles_();
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void TBEPluginLoader::LoadPluginSo(
    const std::map<string, string> &options) {
  vector<string> file_list;
  string caffe_parser_path;
  std::string plugin_path;

  options_ = options;
  GetCustomOpPath(plugin_path);

  // Whether there are files in the plugin so path
  GetPluginSoFileList(plugin_path, file_list, caffe_parser_path);

  //  No file
  if (file_list.empty()) {
    // Print log
    GELOGW("Can not find any plugin file in plugin_path: %s", plugin_path.c_str());
  }

  GELOGW("The shared library will not be checked. Please ensure that the source of the shared library is trusted.");

  // Load other so files except lib_caffe_parser.so in the plugin so path
  for (auto elem : file_list) {
    StringUtils::Trim(elem);

    void *handle = mmDlopen(elem.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if ((handle == nullptr) && !TryOnceAfterLoadRegisterSo(elem, &handle)) {
      GELOGW("dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), dlerror());
    } else if (find(handles_vec_.begin(), handles_vec_.end(), handle) == handles_vec_.end()) {
      // Close dl when the program exist, not close here
      GELOGI("Plugin load %s success.", elem.c_str());
      handles_vec_.push_back(handle);
    } else {
      GELOGI("Plugin so has already been loaded, no need to load again.");
    }
  }
}

Status TBEPluginLoader::GetOppPath(std::string &opp_path) {
  GELOGI("Enter get opp path schedule");
  const char *path_env = std::getenv(kOppEnvName);
  if (path_env != nullptr) {
    opp_path = path_env;
    std::string file_path = parser::RealPath(opp_path.c_str());
    if (file_path.empty()) {
      GELOGW("[Call][RealPath] File path %s is invalid.", opp_path.c_str());
    } else {
      GELOGI("Get opp path from env: %s", opp_path.c_str());
    }
    if (opp_path.back() != '/') {
      opp_path += '/';
    }
  }
  if (opp_path.empty()) {
    opp_path = GetPath();
    GELOGI("Get opp path from so path, value is %s", opp_path.c_str());
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/') + 1);
    opp_path += "ops/";
  }
  return SUCCESS;
}

bool TBEPluginLoader::IsNewOppPathStruct(const std::string &opp_path) {
  return mmIsDir((opp_path + kBuiltIn).c_str()) == EN_OK;
}

Status TBEPluginLoader::GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors) {
  GELOGI("Enter get opp plugin config file schedule, config file is '%s'", vendors_config.c_str());
  std::ifstream config(vendors_config);
  if (!config.good()) {
    GELOGI("Can not open file '%s'!", vendors_config.c_str());
    return FAILED;
  }
  std::string content;
  std::getline(config, content);
  config.close();
  GE_ASSERT_TRUE(!content.empty(), "Content of file '%s' is empty!", vendors_config.c_str());
  std::vector<std::string> v_parts = StringUtils::Split(content, '=');
  GE_ASSERT_TRUE(v_parts.size() == kVendorConfigPartsCount, "Format of file content is invalid!");
  vendors = StringUtils::Split(v_parts[1], ',');
  GE_ASSERT_TRUE(!vendors.empty(), "Format of file content is invalid!");
  (void) for_each(vendors.begin(), vendors.end(), &StringUtils::Trim);
  return SUCCESS;
}

void TBEPluginLoader::GetPluginPathFromCustomOppPath(const std::string &sub_path, std::string &plugin_path) {
  GELOGI("Start to get plugin path from ASCEND_CUSTOM_OPP_PATH schedule.");
  plugin_path = "";
  const char *const custom_opp_path_env = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (custom_opp_path_env == nullptr) {
    GELOGI("env ASCEND_CUSTOM_OPP_PATH is not defined.");
    return;
  }
  const std::string custom_opp_path = custom_opp_path_env;
  if (custom_opp_path.empty()) {
    GELOGW("env ASCEND_CUSTOM_OPP_PATH is defined but it's empty.");
    return;
  }
  GELOGI("value of env ASCEND_CUSTOM_OPP_PATH is %s.", custom_opp_path.c_str());
  std::vector<std::string> custom_paths = StringUtils::Split(custom_opp_path, ':');
  for (const auto &custom_path : custom_paths) {
    if ((!custom_path.empty()) && (mmIsDir((custom_path + "/" + sub_path).c_str()) == EN_OK)) {
      plugin_path += custom_path + "/" + sub_path + ":";
      GELOGI("custom_path '%s' is valid.", custom_path.c_str());
    } else {
      GELOGI("custom_path '%s' is invalid, which is skipped.", custom_path.c_str());
    }
  }
  GELOGI("Run GetPluginPathFromCustomOppPath finished, current plugin_path is %s.", plugin_path.c_str());
}

Status TBEPluginLoader::GetOppPluginPathOld(const std::string &opp_path,
                                            const std::string &path_fmt,
                                            std::string &plugin_path,
                                            const std::string &path_fmt_custom) {
  GELOGI("Enter get opp plugin path old schedule");
  const std::string &fmt_custom  = path_fmt_custom.empty() ? path_fmt : path_fmt_custom;
  plugin_path = (opp_path + std::regex_replace(fmt_custom, std::regex("%s"), "custom") + ":")
              + (opp_path + std::regex_replace(path_fmt, std::regex("%s"), "built-in"));
  GELOGI("plugin_path is '%s'", plugin_path.c_str());
  return SUCCESS;
}

Status TBEPluginLoader::GetOppPluginPathNew(const std::string &opp_path,
                                            const std::string &path_fmt,
                                            std::string &plugin_path,
                                            const std::string &old_custom_path,
                                            const std::string &path_fmt_custom) {
  GELOGI("Enter get opp plugin path new schedule");
  const std::string vendors_config = opp_path + kVendors + "/" + kConfig;
  std::vector<std::string> vendors;
  if (GetOppPluginVendors(vendors_config, vendors) != SUCCESS) {
    GELOGI("Can not get opp plugin vendors!");
    plugin_path += opp_path + old_custom_path + ":";
  } else {
    const std::string &fmt_custom  = path_fmt_custom.empty() ? path_fmt : path_fmt_custom;
    for (const auto &vendor : vendors) {
      plugin_path += opp_path + kVendors + "/" + std::regex_replace(fmt_custom, std::regex("%s"), vendor) + ":";
    }
  }
  plugin_path += opp_path + std::regex_replace(path_fmt, std::regex("%s"), "built-in");
  GELOGI("plugin_path is '%s'", plugin_path.c_str());
  return SUCCESS;
}

Status TBEPluginLoader::GetOpsProtoPath(std::string &opsproto_path) {
  GELOGI("Enter GetOpsProtoPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    return GetOppPluginPathOld(opp_path, "op_proto/%s/", opsproto_path);
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("op_proto/", opsproto_path);
    return GetOppPluginPathNew(opp_path, "%s/op_proto/", opsproto_path, "op_proto/custom/");
  }
}

void TBEPluginLoader::GetCustomOpPath(std::string &customop_path) {
  GELOGI("Enter get custom op path schedule");
  std::string fmk_type;
  domi::FrameworkType type = domi::TENSORFLOW;
  std::map<string, string>::const_iterator it = options_.find(FRAMEWORK_TYPE);
  if (it != options_.end()) {
    type = static_cast<domi::FrameworkType>(std::strtol(it->second.c_str(), nullptr, 10));
  }
  fmk_type = ge::TypeUtils::FmkTypeToSerialString(type);
  GELOGI("Framework type is %s.", fmk_type.c_str());

  std::string opp_path;
  Status ret = GetOppPath(opp_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get opp path.");
    return;
  }
  if (!IsNewOppPathStruct(opp_path)) {
    GELOGI("Opp plugin path structure is old version!");
    ret = GetOppPluginPathOld(opp_path, "framework/%s/" + fmk_type + "/", customop_path, "framework/%s/");
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("framework/", customop_path);
    ret = GetOppPluginPathNew(opp_path, "%s/framework/" + fmk_type + "/", customop_path, "framework/custom/",
                              "%s/framework/");
  }
  if (ret != SUCCESS) {
    GELOGW("Failed to get custom op path!");
  }
}

Status TBEPluginLoader::GetCustomCaffeProtoPath(std::string &customcaffe_path) {
  GELOGD("Enter GetCustomCaffeProtoPath schedule");
  std::string opp_path;
  GE_ASSERT_TRUE(GetOppPath(opp_path) == SUCCESS, "Failed to get opp path!");
  if (!IsNewOppPathStruct(opp_path)) {
    customcaffe_path = opp_path + "framework/custom/caffe/";
    GELOGI("Opp plugin path structure is old version! customcaffe_path is '%s'", customcaffe_path.c_str());
    return SUCCESS;
  } else {
    GELOGI("Opp plugin path structure is new version!");
    GetPluginPathFromCustomOppPath("framework/caffe/", customcaffe_path);
    const std::string vendors_config = opp_path + kVendors + "/" + kConfig;
    std::vector<std::string> vendors;
    if (GetOppPluginVendors(vendors_config, vendors) != SUCCESS) {
      GELOGI("Can not get opp plugin vendors!");
      customcaffe_path += opp_path + "framework/custom/caffe/";
    } else {
      for (const auto &vendor : vendors) {
        customcaffe_path += (customcaffe_path.empty() || (customcaffe_path.back() == ':')) ? "" : ":";
        customcaffe_path += opp_path + kVendors + "/" + vendor + "/framework/caffe/";
      }
    }
    GELOGI("customcaffe_path is '%s'", customcaffe_path.c_str());
    return SUCCESS;
  }
}

string TBEPluginLoader::GetPath() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(&TBEPluginLoader::GetPath), &dl_info) == 0) {
    GELOGW("Failed to read so path!");
    return string();
  } else {
    string so_path = dl_info.dli_fname;
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

void TBEPluginLoader::GetPluginSoFileList(const string &path, vector<string> &file_list, string &caffe_parser_path) {
  // Support to split multiple so directories by ":"
  vector<string> v_path = StringUtils::Split(path, ':');
  for (size_t i = 0; i < v_path.size(); ++i) {
    FindParserSo(v_path[i], file_list, caffe_parser_path);
    GELOGI("CustomOpLib full name = %s", v_path[i].c_str());
  }
}

void TBEPluginLoader::FindParserSo(const string &path, vector<string> &file_list, string &caffe_parser_path) {
  // Path, change to absolute path
  string real_path = ge::parser::RealPath(path.c_str());
  // Plugin path does not exist
  if (real_path.empty()) {
    GELOGW("RealPath is empty.");
    return;
  }
  struct stat stat_buf;
  if ((stat(real_path.c_str(), &stat_buf) != 0) || (!S_ISDIR(stat_buf.st_mode))) {
    GELOGW("%s is not a dir.", real_path.c_str());
    return;
  }
  struct dirent *dent(nullptr);
  DIR *dir = opendir(real_path.c_str());
  // Plugin path does not exist
  if (dir == nullptr) {
    GELOGW("Open directory %s failed.", real_path.c_str());
    return;
  }

  while ((dent = readdir(dir)) != nullptr) {
    if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) {
      continue;
    }
    string name = dent->d_name;
    string full_name = real_path + "/" + name;
    const string so_suff = ".so";
    const string caffe_parser_so_suff = "lib_caffe_parser.so";
    if (name.size() >= so_suff.size() && name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0) {
      ProcessSoFullName(file_list, caffe_parser_path, full_name, caffe_parser_so_suff);
    } else {
      FindParserSo(full_name, file_list, caffe_parser_path);
    }
  }
  closedir(dir);
}

void TBEPluginLoader::ProcessSoFullName(vector<string> &file_list, string &caffe_parser_path, string &full_name,
                                        const string &caffe_parser_so_suff) {
  if (full_name.size() >= caffe_parser_so_suff.size() &&
      full_name.compare(full_name.size() - caffe_parser_so_suff.size(), caffe_parser_so_suff.size(),
                        caffe_parser_so_suff) == 0) {
    caffe_parser_path = full_name;
  } else {
    // Save parser so path into file_list vector
    file_list.push_back(full_name);
  }
}

// for version compatibility, follow the ccb  conclusion to avoid the problem that
// custom opp so is not linked to libregister.so
// the formal solution is to modify custom opp makefile to link libregsiter.so
// after the formal solution, here can be deleted.
bool TBEPluginLoader::TryOnceAfterLoadRegisterSo(const std::string &opp_path, void **handle) {
  static std::atomic_bool flag{false};
  if (!flag && (handle != nullptr)) {
    flag = true;
    std::string tmp_path = GetModelPath();
    tmp_path.append(kLibRegisterSo);
    std::string register_path = RealPath(tmp_path.c_str());
    if (register_path.empty()) {
      GELOGW("Can not find libregister from path:%s", register_path.c_str());
      return false;
    }
    handle_reg_ = mmDlopen(register_path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (handle_reg_ == nullptr) {
      GELOGW("Load libregister failed from path:%s", register_path.c_str());
      return false;
    }
    GELOGD("Load libregister succ from path:%s, will try to load %s later.", register_path.c_str(), opp_path.c_str());

    *handle = mmDlopen(opp_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    return (*handle != nullptr);
  }
  return false;
}
}  // namespace ge
