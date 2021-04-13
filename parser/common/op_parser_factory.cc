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

#include "parser/common/op_parser_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY CustomParserAdapterRegistry *CustomParserAdapterRegistry::Instance() {
  static CustomParserAdapterRegistry instance;
  return &instance;
}

FMK_FUNC_HOST_VISIBILITY void CustomParserAdapterRegistry::Register(const domi::FrameworkType framework,
                                                                    CustomParserAdapterRegistry::CREATOR_FUN fun) {
  if (funcs_.find(framework) != funcs_.end()) {
    GELOGW("Framework type %s has already registed.", TypeUtils::FmkTypeToSerialString(framework).c_str());
    return;
  }
  funcs_[framework] = fun;
  GELOGI("Register %s custom parser adapter success.", TypeUtils::FmkTypeToSerialString(framework).c_str());
  return;
}
FMK_FUNC_HOST_VISIBILITY CustomParserAdapterRegistry::CREATOR_FUN
CustomParserAdapterRegistry::GetCreateFunc(const domi::FrameworkType framework) {
  if (funcs_.find(framework) == funcs_.end()) {
    GELOGW("Framework type %s has not registed.", TypeUtils::FmkTypeToSerialString(framework).c_str());
    return nullptr;
  }
  return funcs_[framework];
}

FMK_FUNC_HOST_VISIBILITY std::shared_ptr<OpParserFactory> OpParserFactory::Instance(
    const domi::FrameworkType framework) {
  // Each framework corresponds to one op parser factory,
  // If instances are static data members of opparserfactory, the order of their construction is uncertain.
  // Instances cannot be a member of a class because they may be used before initialization, resulting in a run error.
  static std::map<domi::FrameworkType, std::shared_ptr<OpParserFactory>> instances;

  auto iter = instances.find(framework);
  if (iter == instances.end()) {
    std::shared_ptr<OpParserFactory> instance(new (std::nothrow) OpParserFactory());
    if (instance == nullptr) {
      REPORT_CALL_ERROR("E19999", "create OpParserFactory failed");
      GELOGE(INTERNAL_ERROR, "[Create][OpParserFactory] failed.");
      return nullptr;
    }
    instances[framework] = instance;
    return instance;
  }

  return iter->second;
}

FMK_FUNC_HOST_VISIBILITY std::shared_ptr<OpParser> OpParserFactory::CreateOpParser(const std::string &op_type) {
  // First look for CREATOR_FUN based on OpType, then call CREATOR_FUN to create OpParser.
  auto iter = op_parser_creator_map_.find(op_type);
  if (iter != op_parser_creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERROR("E19999", "param op_type invalid, Not supported type: %s", op_type.c_str());
  GELOGE(FAILED, "[Check][Param] OpParserFactory::CreateOpParser: Not supported type: %s", op_type.c_str());
  return nullptr;
}

FMK_FUNC_HOST_VISIBILITY std::shared_ptr<OpParser> OpParserFactory::CreateFusionOpParser(const std::string &op_type) {
  // First look for CREATOR_FUN based on OpType, then call CREATOR_FUN to create OpParser.
  auto iter = fusion_op_parser_creator_map_.find(op_type);
  if (iter != fusion_op_parser_creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERROR("E19999", "param op_type invalid, Not supported fusion op type: %s", op_type.c_str());
  GELOGE(FAILED, "[Check][Param] OpParserFactory::CreateOpParser: Not supported fusion op type: %s", op_type.c_str());
  return nullptr;
}

// This function is only called within the constructor of the global opparserregisterar object,
// and does not involve concurrency, so there is no need to lock it
FMK_FUNC_HOST_VISIBILITY void OpParserFactory::RegisterCreator(const std::string &type, CREATOR_FUN fun,
                                                               bool is_fusion_op) {
  std::map<std::string, CREATOR_FUN> *op_parser_creator_map = &op_parser_creator_map_;
  if (is_fusion_op) {
    op_parser_creator_map = &fusion_op_parser_creator_map_;
  }

  GELOGD("OpParserFactory::RegisterCreator: op type:%s, is_fusion_op:%d.", type.c_str(), is_fusion_op);
  (*op_parser_creator_map)[type] = fun;
}

FMK_FUNC_HOST_VISIBILITY bool OpParserFactory::OpParserIsRegistered(const std::string &op_type, bool is_fusion_op) {
  if (is_fusion_op) {
    auto iter = fusion_op_parser_creator_map_.find(op_type);
    if (iter != fusion_op_parser_creator_map_.end()) {
      return true;
    }
  } else {
    auto iter = op_parser_creator_map_.find(op_type);
    if (iter != op_parser_creator_map_.end()) {
      return true;
    }
  }
  return false;
}
}  // namespace ge
