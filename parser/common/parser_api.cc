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

#include "framework/omg/parser/parser_api.h"
#include "common/debug/log.h"

#include "common/ge/tbe_plugin_manager.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/register_tbe.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "external/ge/ge_api_types.h"

namespace ge {
static bool parser_initialized = false;
// Initialize PARSER, load custom op plugin
// options will be used later for parser decoupling
Status ParserInitialize(const std::map<std::string, std::string> &options) {
  GELOGT(TRACE_INIT, "ParserInitialize start");
  // check init status
  if (parser_initialized) {
    GELOGW("ParserInitialize is called more than once");
    return SUCCESS;
  }

  // load custom op plugin
  TBEPluginManager::Instance().LoadPluginSo(options);

  std::vector<OpRegistrationData> registrationDatas = domi::OpRegistry::Instance()->registrationDatas;
  GELOGI("The size of registrationDatas in parser is: %zu", registrationDatas.size());
  for (OpRegistrationData &reg_data : registrationDatas) {
    (void)OpRegistrationTbe::Instance()->Finalize(reg_data, true);
  }

  auto iter = options.find(ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES);
  if (iter != options.end()) {
    ge::GetParserContext().enable_scope_fusion_passes = iter->second;
  }

  // set init status
  if (!parser_initialized) {
    // Initialize success, first time calling initialize
    parser_initialized = true;
  }

  GELOGT(TRACE_STOP, "ParserInitialize finished");
  return SUCCESS;
}

Status ParserFinalize() {
  GELOGT(TRACE_INIT, "ParserFinalize start");
  // check init status
  if (!parser_initialized) {
    GELOGW("ParserFinalize is called before ParserInitialize");
    return SUCCESS;
  }

  GE_CHK_STATUS(TBEPluginManager::Instance().Finalize());
  if (parser_initialized) {
    parser_initialized = false;
  }
  return SUCCESS;
}
}  // namespace ge
