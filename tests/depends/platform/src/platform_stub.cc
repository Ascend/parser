/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "platform/platform_info.h"

fe::PlatformInfoManager& fe::PlatformInfoManager::Instance() {
  static fe::PlatformInfoManager pf;
  return pf;
}
fe::PlatformInfoManager& fe::PlatformInfoManager::GeInstance() {
  static fe::PlatformInfoManager pf;
  return pf;
}

uint32_t fe::PlatformInfoManager::InitializePlatformInfo() {
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetPlatformInstanceByDevice(const uint32_t &device_id,
                                                              PlatFormInfos &platform_infos) {
  return 0U;
}

uint32_t fe::PlatformInfoManager::GetPlatformInfoWithOutSocVersion(fe::PlatFormInfos &platform_info,
                                                                   fe::OptionalInfos &opti_compilation_info) {
  return 0U;
}

fe::PlatformInfoManager::PlatformInfoManager() {}
fe::PlatformInfoManager::~PlatformInfoManager() {}

uint32_t fe::PlatFormInfos::GetCoreNum() const {
  return 8U;
}

void fe::PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type) {
  return;
}

bool fe::PlatFormInfos::GetPlatformResWithLock(const std::string &label,
                                               const std::string &key, std::string &val) {
  if (label == "DtypeMKN" && key == "Default") {
    val = "16,16,16";
  }
  return true;
}

bool fe::PlatFormInfos::GetPlatformResWithLock(const std::string &label,
                                               std::map<std::string, std::string> &res) {
  if (label == "DtypeMKN") {
    res = {{"DT_UINT8", "16,32,16"},
           {"DT_INT8", "16,32,16"},
           {"DT_INT4", "16,64,16"},
           {"DT_INT2", "16,128,16"},
           {"DT_UINT2", "16,128,16"},
           {"DT_UINT1", "16,256,16"}};
  }
  return true;
}
