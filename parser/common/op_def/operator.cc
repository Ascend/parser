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

#include "operator.h"
#include <utility>
#include "framework/common/fmk_types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"

using ge::BoolTuple;
using ge::FloatTuple;
using ge::IntTuple;
using ge::StringTuple;
using ge::UintTuple;

namespace ge {
ParserOperator::ParserOperator(const std::string &type) {
  type_ = type;
  op_schema_ = ge::OpSchemaFactory::Instance().Get(type);
  if (op_schema_ == nullptr) {
    GELOGW("Cannot find op schema of op type: %s", type.c_str());
  }
}

ParserOperator &ParserOperator::Input(const ParserOperator &in_op, uint32_t index) {
  if (index == 0) {
    inputs_.push_back(in_op.GetName());
  } else {
    inputs_.push_back(in_op.GetName() + ":" + std::to_string(index));
  }
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ParserOperator &ParserOperator::Name(const std::string &name) {
  name_ = name;
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ParserOperator &ParserOperator::Type(const std::string &type) {
  type_ = type;
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ParserOperator &ParserOperator::InputTensorDesc(
  const ge::GeTensorDesc &input_tensordesc) {
  input_descs_.push_back(input_tensordesc);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ParserOperator &ParserOperator::OutputTensorDesc(
  const ge::GeTensorDesc &output_tensordesc) {
  output_descs_.push_back(output_tensordesc);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ParserOperator &ParserOperator::AttrVector(
  std::string key,
  std::vector<int32_t> &value) {
  domi::AttrDef out;
  auto it = op_attrs_.find(key);
  if (it != op_attrs_.end()) {
    out = it->second.value_;
  }
  for (auto &v : value) {
    out.mutable_list()->add_i(v);
  }
  (void)op_attrs_.erase(key);
  (void)op_attrs_.insert(std::make_pair(key, OpAttribute(key, out)));
  return *this;
}
FMK_FUNC_DEV_VISIBILITY FMK_FUNC_DEV_VISIBILITY ParserOperator &ParserOperator::AttrVector(
  std::string key,
  std::vector<int64_t> &value) {
  domi::AttrDef out;
  auto it = op_attrs_.find(key);
  if (it != op_attrs_.end()) {
    out = it->second.value_;
  }
  for (auto &v : value) {
    out.mutable_list()->add_i(v);
  }
  (void)op_attrs_.erase(key);
  (void)op_attrs_.insert(std::make_pair(key, OpAttribute(key, out)));
  return *this;
}

ParserOperator &ParserOperator::Attr(const OpAttribute &attr) {
  auto it = op_attrs_.find(attr.name_);
  if (it != op_attrs_.end()) {
    (void)op_attrs_.erase(it);
  }
  (void)op_attrs_.insert(std::make_pair(attr.name_, attr));
  return *this;
}

ParserOperator &ParserOperator::Attr_bt(const std::string &name, const std::string &value) {
  domi::AttrDef a;
  a.set_bt(value);
  Attr(OpAttribute(name, a));
  return *this;
}

#define ATTR_SETTER_WITH_SINGLE_VALUE(type, field)                                   \
  ParserOperator &ParserOperator::Attr(const std::string &name, const type &value) { \
    domi::AttrDef a;                                                                 \
    a.set_##field(value);                                                            \
    Attr(OpAttribute(name, a));                                                      \
    return *this;                                                                    \
  }

#define ATTR_SETTER_WITH_LIST_VALUE(type, field)                                                  \
  ParserOperator &ParserOperator::Attr(const std::string &name, const std::vector<type> &value) { \
    domi::AttrDef a;                                                                              \
    auto attr_list = a.mutable_list();                                                            \
    for (size_t i = 0; i < value.size(); ++i) {                                                   \
      attr_list->add_##field(value[i]);                                                           \
    }                                                                                             \
    Attr(OpAttribute(name, a));                                                                   \
    return *this;                                                                                 \
  }                                                                                               \
  ParserOperator &ParserOperator::Attr(const std::string &name, const ge::Tuple<type> &value) {   \
    domi::AttrDef a;                                                                              \
    auto attr_list = a.mutable_list();                                                            \
    for (uint32_t i = 0; i < value.ndim(); ++i) {                                                 \
      attr_list->add_##field(value[i]);                                                           \
    }                                                                                             \
    Attr(OpAttribute(name, a));                                                                   \
    return *this;                                                                                 \
  }

ATTR_SETTER_WITH_SINGLE_VALUE(int64_t, i)
ATTR_SETTER_WITH_SINGLE_VALUE(bool, b)
ATTR_SETTER_WITH_SINGLE_VALUE(float, f)
ATTR_SETTER_WITH_SINGLE_VALUE(std::string, s)
ATTR_SETTER_WITH_SINGLE_VALUE(uint32_t, i)

ATTR_SETTER_WITH_LIST_VALUE(int64_t, i)
ATTR_SETTER_WITH_LIST_VALUE(bool, b)
ATTR_SETTER_WITH_LIST_VALUE(float, f)
ATTR_SETTER_WITH_LIST_VALUE(std::string, s)
ATTR_SETTER_WITH_LIST_VALUE(uint32_t, i)

#define ATTR_GET_SINGLE_VALUE(type, field, type_name)                        \
  type ParserOperator::Get##type_name##Attr(const std::string &name) const { \
    domi::AttrDef single_val;                                                \
    auto it = op_attrs_.find(name);                                          \
    if (it != op_attrs_.end()) {                                             \
      single_val = it->second.value_;                                        \
    } else {                                                                 \
      if (op_schema_ && op_schema_->HasDefaultAttr(name)) {                  \
        single_val = op_schema_->GetDefaultAttr(name);                       \
      }                                                                      \
    }                                                                        \
    return single_val.field();                                               \
  }
ATTR_GET_SINGLE_VALUE(uint32_t, i, Uint)
ATTR_GET_SINGLE_VALUE(int64_t, i, Int)
ATTR_GET_SINGLE_VALUE(float, f, Float)
ATTR_GET_SINGLE_VALUE(bool, b, Bool)
ATTR_GET_SINGLE_VALUE(std::string, s, String)

#define ATTR_GET_TUPLE_VALUE(type, field, tuple_type_name)                                    \
  tuple_type_name ParserOperator::Get##tuple_type_name##Attr(const std::string &name) const { \
    domi::AttrDef value;                                                                      \
    auto it = op_attrs_.find(name);                                                           \
    if (it != op_attrs_.end()) {                                                              \
      value = it->second.value_;                                                              \
    } else {                                                                                  \
      if (op_schema_ && op_schema_->HasDefaultAttr(name)) {                                   \
        value = op_schema_->GetDefaultAttr(name);                                             \
      }                                                                                       \
    }                                                                                         \
    const auto attr_def = value.list();                                                       \
    std::size_t n = attr_def.field##_size();                                                  \
    std::vector<type> vec(n);                                                                 \
    for (std::size_t i = 0; i < n; i++) {                                                     \
      vec[i] = attr_def.field(i);                                                             \
    }                                                                                         \
    return tuple_type_name(vec);                                                              \
  }

ATTR_GET_TUPLE_VALUE(uint32_t, i, UintTuple)
ATTR_GET_TUPLE_VALUE(int64_t, i, IntTuple)
ATTR_GET_TUPLE_VALUE(float, f, FloatTuple)
ATTR_GET_TUPLE_VALUE(bool, b, BoolTuple)
ATTR_GET_TUPLE_VALUE(std::string, s, StringTuple)
}  // namespace domi
