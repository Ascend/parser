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

#include "omg/parser/parser_factory.h"
#include "framework/common/debug/ge_log.h"

namespace domi {
FMK_FUNC_HOST_VISIBILITY WeightsParserFactory *WeightsParserFactory::Instance() {
  static WeightsParserFactory instance;
  return &instance;
}

std::shared_ptr<WeightsParser> WeightsParserFactory::CreateWeightsParser(const domi::FrameworkType type) {
  std::map<domi::FrameworkType, WEIGHTS_PARSER_CREATOR_FUN>::iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERROR("E19999", "param type invalid, Not supported Type: %d", type);
  GELOGE(FAILED, "[Check][Param]WeightsParserFactory::CreateWeightsParser: Not supported Type: %d", type);
  return nullptr;
}

FMK_FUNC_HOST_VISIBILITY void WeightsParserFactory::RegisterCreator(const domi::FrameworkType type,
                                                                    WEIGHTS_PARSER_CREATOR_FUN fun) {
  std::map<domi::FrameworkType, WEIGHTS_PARSER_CREATOR_FUN>::iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    GELOGW("WeightsParserFactory::RegisterCreator: %d creator already exist", type);
    return;
  }

  creator_map_[type] = fun;
}

WeightsParserFactory::~WeightsParserFactory() {
  creator_map_.clear();
}

FMK_FUNC_HOST_VISIBILITY ModelParserFactory *ModelParserFactory::Instance() {
  static ModelParserFactory instance;
  return &instance;
}

std::shared_ptr<ModelParser> ModelParserFactory::CreateModelParser(const domi::FrameworkType type) {
  std::map<domi::FrameworkType, MODEL_PARSER_CREATOR_FUN>::iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERROR("E19999", "param type invalid, Not supported Type: %d", type);
  GELOGE(FAILED, "[Check][Param]ModelParserFactory::CreateModelParser: Not supported Type: %d", type);
  return nullptr;
}

FMK_FUNC_HOST_VISIBILITY void ModelParserFactory::RegisterCreator(const domi::FrameworkType type,
                                                                  MODEL_PARSER_CREATOR_FUN fun) {
  std::map<domi::FrameworkType, MODEL_PARSER_CREATOR_FUN>::iterator iter = creator_map_.find(type);
  if (iter != creator_map_.end()) {
    GELOGW("ModelParserFactory::RegisterCreator: %d creator already exist", type);
    return;
  }

  creator_map_[type] = fun;
}

ModelParserFactory::~ModelParserFactory() {
  creator_map_.clear();
}
}  // namespace domi
