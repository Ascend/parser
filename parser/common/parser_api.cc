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

#include "framework/omg/parser/parser_api.h"

#include "common/util.h"
#include "common/acl_graph_parser_util.h"
#include "tbe_plugin_loader.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_registration_tbe.h"
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
  AclGraphParserUtil acl_graph_parse_util;
  if (acl_graph_parse_util.AclParserInitialize(options, true) != domi::SUCCESS) {
    GELOGE(GRAPH_FAILED, "Parser Initialize failed.");
    return GRAPH_FAILED;
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

  GE_CHK_STATUS(TBEPluginLoader::Instance().Finalize());
  if (parser_initialized) {
    parser_initialized = false;
  }
  return SUCCESS;
}
}  // namespace ge
