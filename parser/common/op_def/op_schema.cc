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

#include "common/op_def/op_schema.h"
#include <iostream>
#include <utility>
#include "framework/common/debug/ge_log.h"

namespace ge {
OpSchema::FormalParameter::FormalParameter(const std::string &name, FormalParameterOption param_option)
    : name_(name), param_option_(param_option) {}

OpSchema::FormalParameter::~FormalParameter() {}

const std::string &OpSchema::FormalParameter::Name() const { return name_; }

OpSchema::FormalParameterOption OpSchema::FormalParameter::Option() const { return param_option_; }

OpSchema::OpSchema(const std::string &name) : name_(name) {}

OpSchema::~OpSchema() {}

OpSchema &OpSchema::Input(const std::string &name, FormalParameterOption param_option) {
  inputs_.emplace_back(FormalParameter(name, param_option));
  return *this;
}

OpSchema &OpSchema::Output(const std::string &name, FormalParameterOption param_option) {
  outputs_.emplace_back(FormalParameter(name, param_option));
  return *this;
}

OpSchema &OpSchema::Attr(const Attribute &attr) {
  (void)attributes_.insert(std::make_pair(attr.name_, attr));
  return *this;
}

#if defined(CFG_BUILD_DEBUG)
#define ATTR_SETTER_WITH_SINGLE_VALUE(Type, field, attrtype)                                               \
  OpSchema &OpSchema::Attr(const std::string &name, AttributeType attr_type, const Type &default_value) {  \
    if (attrtype != attr_type) {                                                                           \
      REPORT_INNER_ERROR("E19999", "Attribute specification param_type mismatch, input attr type %u, "     \
                         "required attr type %u.", (uint32_t)attr_type, (uint32_t)attrtype);               \
      GELOGE(FAILED, "[Check][Param]Attribute specification param_type mismatch, "                         \
             "input attr type %u, required attr type %u.", (uint32_t)attr_type, (uint32_t)attrtype);       \
      return *this;                                                                                        \
    }                                                                                                      \
                                                                                                           \
    domi::AttrDef a;                                                                                       \
    a.set_##field(default_value);                                                                          \
    Attr(Attribute(name, attr_type, a));                                                                   \
    return *this;                                                                                          \
  }
#else
#define ATTR_SETTER_WITH_SINGLE_VALUE(Type, field, attrtype)                                              \
  OpSchema &OpSchema::Attr(const std::string &name, AttributeType attr_type, const Type &default_value) { \
    if (attrtype != attr_type) {                                                                          \
      return *this;                                                                                       \
    }                                                                                                     \
    domi::AttrDef a;                                                                                      \
    a.set_##field(default_value);                                                                         \
    Attr(Attribute(name, attr_type, a));                                                                  \
    return *this;                                                                                         \
  }

#endif

#if defined(CFG_BUILD_DEBUG)
#define ATTR_SETTER_WITH_LIST_VALUE(Type, field, attrtype)                                                             \
  OpSchema &OpSchema::Attr(const std::string &name, AttributeType attr_type, const std::vector<Type> &default_value) { \
    if (attrtype != attr_type) {                                                                                       \
      REPORT_INNER_ERROR("E19999", "Attribute specification vector param_type mismatch, "                              \
                         "input attr type %u, required attr type %u.", (uint32_t)attr_type, (uint32_t)attrtype);       \
      GELOGE(FAILED, "[Check][Param]Attribute specification vector param_type mismatch, "                              \
             "input attr type %u, required attr type %u.", (uint32_t)attr_type, (uint32_t)attrtype);                   \
      return *this;                                                                                                    \
    }                                                                                                                  \
    domi::AttrDef vec_a;                                                                                               \
    for (const auto &v : default_value) {                                                                              \
      vec_a.mutable_list()->add_##field(v);                                                                            \
    }                                                                                                                  \
    Attr(Attribute(name, attr_type, vec_a));                                                                           \
    return *this;                                                                                                      \
  }                                                                                                                    \
  OpSchema &OpSchema::Attr(const std::string &name, AttributeType attr_type, const Tuple<Type> &default_value) {       \
    if (attrtype != attr_type) {                                                                                       \
      REPORT_INNER_ERROR("E19999", "Attribute specification vector param_type mismatch, "                              \
                         "input attr type %u, required attr type %u.", (uint32_t)attr_type, (uint32_t)attrtype);       \
      GELOGE(FAILED, "[Check][Param]Attribute specification vector param_type mismatch, "                              \
             "input attr type %u, required attr type %u.", (uint32_t)attr_type, (uint32_t)attrtype);                   \
      return *this;                                                                                                    \
    }                                                                                                                  \
    domi::AttrDef tuple_a;                                                                                             \
    for (const auto &v : default_value) {                                                                              \
      tuple_a.mutable_list()->add_##field(v);                                                                          \
    }                                                                                                                  \
    Attr(Attribute(name, attr_type, tuple_a));                                                                         \
    return *this;                                                                                                      \
  }
#else
#define ATTR_SETTER_WITH_LIST_VALUE(Type, field, attrtype)                                                             \
  OpSchema &OpSchema::Attr(const std::string &name, AttributeType attr_type, const std::vector<Type> &default_value) { \
    if (attrtype != attr_type) {                                                                                       \
      return *this;                                                                                                    \
    }                                                                                                                  \
    domi::AttrDef vec_a;                                                                                               \
    for (const auto &v : default_value) {                                                                              \
      vec_a.mutable_list()->add_##field(v);                                                                            \
    }                                                                                                                  \
    Attr(Attribute(name, attr_type, vec_a));                                                                           \
    return *this;                                                                                                      \
  }                                                                                                                    \
  OpSchema &OpSchema::Attr(const std::string &name, AttributeType attr_type, const Tuple<Type> &default_value) {       \
    if (attrtype != attr_type) {                                                                                       \
      return *this;                                                                                                    \
    }                                                                                                                  \
    domi::AttrDef tuple_a;                                                                                             \
    for (const auto &v : default_value) {                                                                              \
      tuple_a.mutable_list()->add_##field(v);                                                                          \
    }                                                                                                                  \
    Attr(Attribute(name, attr_type, tuple_a));                                                                         \
    return *this;                                                                                                      \
  }

#endif
ATTR_SETTER_WITH_SINGLE_VALUE(uint32_t, u, AttributeType::UINT)
ATTR_SETTER_WITH_SINGLE_VALUE(int64_t, i, AttributeType::INT)
ATTR_SETTER_WITH_SINGLE_VALUE(bool, b, AttributeType::BOOL)
ATTR_SETTER_WITH_SINGLE_VALUE(float, f, AttributeType::FLOAT)
ATTR_SETTER_WITH_SINGLE_VALUE(std::string, s, AttributeType::STRING)

ATTR_SETTER_WITH_LIST_VALUE(uint32_t, u, AttributeType::UINTLIST)
ATTR_SETTER_WITH_LIST_VALUE(int64_t, i, AttributeType::INTLIST)
ATTR_SETTER_WITH_LIST_VALUE(bool, b, AttributeType::BOOLLIST)
ATTR_SETTER_WITH_LIST_VALUE(float, f, AttributeType::FLOATLIST)
ATTR_SETTER_WITH_LIST_VALUE(std::string, s, AttributeType::STRINGLIST)

OpSchema &OpSchema::AttrRequired(const std::string &name, AttributeType attr_type) {
  Attr(Attribute(name, attr_type, true));
  return *this;
}

bool OpSchema::HasDefaultAttr(const std::string &name) const {
  auto it = attributes_.find(name);
  if (it == attributes_.end()) {
    return false;
  }

  // required does not need a default value
  return !it->second.required_;
}

const domi::AttrDef &OpSchema::GetDefaultAttr(const std::string &name) const {
  auto it = attributes_.find(name);
  if (it == attributes_.end()) {
    const static domi::AttrDef attr_def;
    return attr_def;
  }
  return it->second.default_value_;
}

bool OpSchema::Verify(const ge::OpDescPtr op_def) const {
  if (op_def->GetType() != name_) {
    REPORT_INNER_ERROR("E19999", "Name not math, op schema name: %s, opdef type: %s.",
                       name_.c_str(), op_def->GetType().c_str());
    GELOGE(FAILED, "[Check][Param]Name not math, op schema name: %s, opdef type: %s.",
           name_.c_str(), op_def->GetType().c_str());
    return false;
  }

  // Required field verification
  for (const auto &pair : attributes_) {
    const auto &attr = pair.second;
    if (!attr.required_) {
      continue;
    }
    if (!op_def->HasAttr(attr.name_)) {
      REPORT_INNER_ERROR("E19999", "Required attribute: %s of op: %s is missing.",
                         attr.name_.c_str(), op_def->GetName().c_str());
      GELOGE(FAILED, "[Invoke][HasAttr]Required attribute: %s of op: %s is missing.",
             attr.name_.c_str(), op_def->GetName().c_str());
      return false;
    }
  }

  return true;
}

OpSchemaFactory &OpSchemaFactory::Instance() {
  static OpSchemaFactory instance;
  return instance;
}

const OpSchema *OpSchemaFactory::Get(const std::string &op) const {
  auto it = op_schema_map_.find(op);
  if (it == op_schema_map_.end()) {
    return nullptr;
  }
  return &it->second;
}

OpSchemaRegistry::OpSchemaRegistry(OpSchema &op_schema) {
  OpSchemaFactory &op_factory = OpSchemaFactory::Instance();

  // save op_schema to the map
  if (op_factory.op_schema_map_.count(op_schema.name_)) {
    GELOGD("Failed to register op schema: %s., reason: already exist!", op_schema.name_.c_str());
    return;
  }

  (void)op_factory.op_schema_map_.emplace(std::make_pair(op_schema.name_, op_schema));
}
}  // namespace ge
