/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

// File:        pb2json.h
// Description: This header file for protobuf message and json interconversion

#ifndef PARSER_COMMON_CONVERT_PB2JSON_H_
#define PARSER_COMMON_CONVERT_PB2JSON_H_
#include <functional>
#include <memory>
#include <set>
#include <string>
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "nlohmann/json.hpp"

namespace ge {
using Json = nlohmann::json;
using ProtobufMsg = ::google::protobuf::Message;
using ProtobufReflection = ::google::protobuf::Reflection;
using ProtobufFieldDescriptor = ::google::protobuf::FieldDescriptor;
using ProtobufDescriptor = ::google::protobuf::Descriptor;
using ProtobufEnumValueDescriptor = ::google::protobuf::EnumValueDescriptor;

class Pb2Json {
 public:
  /**
  * @ingroup domi_omg
  * @brief Transfer protobuf object to JSON object
  * @param [out] json Converted JSON object
  * @return void success
  * @author
  */
  static void Message2Json(const ProtobufMsg &message, const std::set<std::string> &black_fields, Json &json,
                           bool enum2str = false, int depth = 0);

  static void RepeatedMessage2Json(const ProtobufMsg &message, const ProtobufFieldDescriptor *field,
                                   const ProtobufReflection *reflection, const std::set<std::string> &black_fields,
                                   Json &json, bool enum2str, int depth = 0);

  static void EnumJson2Json(Json &json);

protected:
  static void Enum2Json(const ProtobufEnumValueDescriptor *enum_value_desc, const ProtobufFieldDescriptor *field,
                        bool enum2str, Json &json);

  static void RepeatedEnum2Json(const ProtobufEnumValueDescriptor *enum_value_desc, bool enum2str, Json &json);

  static void OneField2Json(const ProtobufMsg &message, const ProtobufFieldDescriptor *field,
                            const ProtobufReflection *reflection, const std::set<std::string> &black_fields, Json &json,
                            bool enum2str, int depth);

  static std::string TypeBytes2String(std::string &field_name, std::string &type_bytes);

  static int DictInit(Json &json, std::vector<std::string> &idx2name,
                      std::vector<std::string> &idx2value, std::vector<bool> &use_string_val);

  static int AttrReplaceKV(Json &attrs, const std::vector<std::string> &idx2name,
                           const std::vector<std::string> &idx2value, const std::vector<bool> &use_string_val);
};
}  // namespace ge

#endif  // PARSER_COMMON_CONVERT_PB2JSON_H_
