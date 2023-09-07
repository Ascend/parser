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

#ifndef PARSER_COMMON_TBE_PLUGIN_LOADER_H_
#define PARSER_COMMON_TBE_PLUGIN_LOADER_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "external/ge/ge_api_error_codes.h"
#include "external/register/register.h"

namespace ge {
using SoHandlesVec = std::vector<void *>;
class TBEPluginLoader {
public:
  Status Finalize();

  // Get TBEPluginManager singleton instance
  static TBEPluginLoader& Instance();

  void LoadPluginSo(const std::map<string, string> &options);

  static string GetPath();

  static Status GetOpsProtoPath(std::string &opsproto_path);

  static Status GetCustomCaffeProtoPath(std::string &customcaffe_path);

private:
  TBEPluginLoader() = default;
  ~TBEPluginLoader() = default;
  Status ClearHandles_();
  static void ProcessSoFullName(vector<string> &file_list, string &caffe_parser_path, string &full_name,
                                const string &caffe_parser_so_suff);
  static Status GetOppPath(std::string &opp_path);
  static bool IsNewOppPathStruct(const std::string &opp_path);
  static Status GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors);
  static void GetPluginPathFromCustomOppPath(const std::string &sub_path, std::string &plugin_path);
  static Status GetOppPluginPathOld(const std::string &opp_path,
                                    const std::string &path_fmt,
                                    std::string &plugin_path,
                                    const std::string &path_fmt_custom = "");
  static Status GetOppPluginPathNew(const std::string &opp_path,
                                    const std::string &path_fmt,
                                    std::string &plugin_path,
                                    const std::string &old_custom_path,
                                    const std::string &path_fmt_custom = "");
  static void GetCustomOpPath(std::string &customop_path);
  static void GetPluginSoFileList(const string &path, vector<string> &file_list, string &caffe_parser_path);
  static void FindParserSo(const string &path, vector<string> &file_list, string &caffe_parser_path);
  bool TryOnceAfterLoadRegisterSo(const std::string &opp_path, void **handle);

  SoHandlesVec handles_vec_;
  void *handle_reg_ = nullptr;
  static std::map<string, string> options_;
};
}  // namespace ge

#endif // PARSER_COMMON_TBE_PLUGIN_LOADER_H_
