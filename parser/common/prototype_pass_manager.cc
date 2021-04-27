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

#include "prototype_pass_manager.h"

#include "common/util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
ProtoTypePassManager &ProtoTypePassManager::Instance() {
  static ProtoTypePassManager instance;
  return instance;
}

Status ProtoTypePassManager::Run(google::protobuf::Message *message, const domi::FrameworkType &fmk_type) {
  GE_CHECK_NOTNULL(message);
  const auto &pass_vec = ProtoTypePassRegistry::GetInstance().GetCreateFnByType(fmk_type);
  for (const auto &pass_item : pass_vec) {
    std::string pass_name = pass_item.first;
    const auto &func = pass_item.second;
    GE_CHECK_NOTNULL(func);
    std::unique_ptr<ProtoTypeBasePass> pass = std::unique_ptr<ProtoTypeBasePass>(func());
    GE_CHECK_NOTNULL(pass);
    Status ret = pass->Run(message);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Run ProtoType pass:%s failed", pass_name.c_str());
      return ret;
    }
    GELOGD("Run ProtoType pass:%s success", pass_name.c_str());
  }
  return SUCCESS;
}
}  // namespace ge