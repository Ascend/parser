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
// Description: This imply file for protobuf message and json interconversion

#include "common/convert/pb2json.h"
#include <google/protobuf/text_format.h>
#include <set>
#include <string>
#include "securec.h"
#include "framework/common/fmk_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/enum_attr_utils.h"

using std::set;
using std::string;

namespace ge {
namespace {
const int kSignificantDigits = 10;
const int kMaxParseDepth = 20;
const int NO_COMPRESS = 0;
const int USE_OM_COMPRESS = 1;
}
// JSON parses non utf8 character throwing exceptions, so some fields need to be shielded through black fields
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void Pb2Json::Message2Json(const ProtobufMsg &message,
                                                                            const set<string> &black_fields, Json &json,
                                                                            bool enum2str, int depth) {
  if (depth > kMaxParseDepth) {
    REPORT_INNER_ERROR("E19999", "Message depth:%d can not exceed %d.", depth, kMaxParseDepth);
    GELOGE(FAILED, "[Check][Param]Message depth can not exceed %d.", kMaxParseDepth);
    return;
  }

  auto descriptor = message.GetDescriptor();
  auto reflection = message.GetReflection();
  if (descriptor == nullptr || reflection == nullptr) {
    return;
  }

  auto count = descriptor->field_count();

  for (auto i = 0; i < count; ++i) {
    const auto field = descriptor->field(i);
    if (field == nullptr) {
      return;
    }

    // Do not display weight data
    if (black_fields.find(field->name()) != black_fields.end()) {
      continue;
    }

    if (field->is_repeated()) {
      if (reflection->FieldSize(message, field) > 0) {
        RepeatedMessage2Json(message, field, reflection, black_fields, json[field->name()], enum2str, depth);
      }
      continue;
    }

    if (!reflection->HasField(message, field)) {
      continue;
    }

    OneField2Json(message, field, reflection, black_fields, json, enum2str, depth);
  }

  if (depth == 0) {
    EnumJson2Json(json);
  }
}

void Pb2Json::OneField2Json(const ProtobufMsg &message, const ProtobufFieldDescriptor *field,
                            const ProtobufReflection *reflection, const set<string> &black_fields, Json &json,
                            bool enum2str, int depth) {
  switch (field->type()) {
    case ProtobufFieldDescriptor::TYPE_MESSAGE: {
      const ProtobufMsg &tmp_message = reflection->GetMessage(message, field);
      if (tmp_message.ByteSizeLong() != 0UL) {
        Message2Json(tmp_message, black_fields, json[field->name()], enum2str, depth + 1);
      }
      break;
    }

    case ProtobufFieldDescriptor::TYPE_BOOL:
      json[field->name()] = reflection->GetBool(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_ENUM: {
      auto *enum_value_desc = reflection->GetEnum(message, field);
      Enum2Json(enum_value_desc, field, enum2str, json);
      break;
    }

    case ProtobufFieldDescriptor::TYPE_INT32:
    case ProtobufFieldDescriptor::TYPE_SINT32:
    case ProtobufFieldDescriptor::TYPE_SFIXED32:
      json[field->name()] = reflection->GetInt32(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_UINT32:
    case ProtobufFieldDescriptor::TYPE_FIXED32:
      json[field->name()] = reflection->GetUInt32(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_INT64:
    case ProtobufFieldDescriptor::TYPE_SINT64:
    case ProtobufFieldDescriptor::TYPE_SFIXED64:
      json[field->name()] = reflection->GetInt64(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_UINT64:
    case ProtobufFieldDescriptor::TYPE_FIXED64:
      json[field->name()] = reflection->GetUInt64(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_FLOAT:
      char str[kSignificantDigits];
      if (sprintf_s(str, kSignificantDigits, "%g", reflection->GetFloat(message, field)) != -1) {
        json[field->name()] = str;
      } else {
        json[field->name()] = reflection->GetFloat(message, field);
      }

      break;

    case ProtobufFieldDescriptor::TYPE_STRING:
      json[field->name()] = reflection->GetString(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_BYTES: {
      string field_name = field->name();
      std::string scratch;
      std::string value = reflection->GetStringReference(message, field, &scratch);
      std::string cescape_value = google::protobuf::CEscape(value);
      GELOGD("After cescape data:%s", cescape_value.c_str());
      json[field_name] = cescape_value;
      break;
    }

    default:
      break;
  }
}

string Pb2Json::TypeBytes2String(string &field_name, string &type_bytes) {
  if (field_name != "offset") {
    return type_bytes;
  }
  string result = "";
  for (char temp_value : type_bytes) {
    char str[kSignificantDigits];
    if (sprintf_s(str, kSignificantDigits, "%c", temp_value) == -1) {
      GELOGW("Convert bytes to string fail, filed name:%s", field_name.c_str());
      continue;
    }
    result += str;
  }
  return result;
}

void Pb2Json::RepeatedMessage2Json(const ProtobufMsg &message, const ProtobufFieldDescriptor *field,
                                   const ProtobufReflection *reflection, const set<string> &black_fields, Json &json,
                                   bool enum2str, int depth) {
  if ((field == nullptr) || (reflection == nullptr)) {
    Message2Json(message, black_fields, json, enum2str, depth + 1);
    return;
  }

  for (auto i = 0; i < reflection->FieldSize(message, field); ++i) {
    Json tmp_json;
    switch (field->type()) {
      case ProtobufFieldDescriptor::TYPE_MESSAGE: {
        const ProtobufMsg &tmp_message = reflection->GetRepeatedMessage(message, field, i);
        if (tmp_message.ByteSizeLong() != 0UL) {
          Message2Json(tmp_message, black_fields, tmp_json, enum2str, depth + 1);
        }
      } break;

      case ProtobufFieldDescriptor::TYPE_BOOL:
        tmp_json = reflection->GetRepeatedBool(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_ENUM: {
        auto *enum_value_desc = reflection->GetRepeatedEnum(message, field, i);
        RepeatedEnum2Json(enum_value_desc, enum2str, tmp_json);
      } break;

      case ProtobufFieldDescriptor::TYPE_INT32:
      case ProtobufFieldDescriptor::TYPE_SINT32:
      case ProtobufFieldDescriptor::TYPE_SFIXED32:
        tmp_json = reflection->GetRepeatedInt32(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_UINT32:
      case ProtobufFieldDescriptor::TYPE_FIXED32:
        tmp_json = reflection->GetRepeatedUInt32(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_INT64:
      case ProtobufFieldDescriptor::TYPE_SINT64:
      case ProtobufFieldDescriptor::TYPE_SFIXED64:
        tmp_json = reflection->GetRepeatedInt64(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_UINT64:
      case ProtobufFieldDescriptor::TYPE_FIXED64:
        tmp_json = reflection->GetRepeatedUInt64(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_FLOAT:
        tmp_json = reflection->GetRepeatedFloat(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_STRING:
      case ProtobufFieldDescriptor::TYPE_BYTES:
        tmp_json = reflection->GetRepeatedString(message, field, i);
        break;

      default:
        break;
    }
    json += tmp_json;
  }
}

void Pb2Json::Enum2Json(const ProtobufEnumValueDescriptor *enum_value_desc, const ProtobufFieldDescriptor *field,
                        bool enum2str, Json &json) {
  if (enum_value_desc != nullptr) {
    if (field == nullptr) {
      return;
    }
    if (enum2str) {
      json[field->name()] = enum_value_desc->name();
    } else {
      json[field->name()] = enum_value_desc->number();
    }
  }
}

void Pb2Json::RepeatedEnum2Json(const ProtobufEnumValueDescriptor *enum_value_desc, bool enum2str, Json &json) {
  if (enum_value_desc != nullptr) {
    if (enum2str) {
      json = enum_value_desc->name();
    } else {
      json = enum_value_desc->number();
    }
  }
}

int Pb2Json::DictInit(Json &json, std::vector<string> &idx2name, std::vector<string> &idx2value,
                      std::vector<bool> &use_string_val) {
  if (json.find("attr") == json.end()) {
    return NO_COMPRESS;
  }

  int om_compress_version = NO_COMPRESS;
  for (auto it = json["attr"].begin(); it != json["attr"].end();) {
    const auto &key = (*it)["key"];
    const auto &value = (*it)["value"];
    bool obj_del = true;
    if (key == ATTR_MODEL_OM_COMPRESS_VERSION) {
      om_compress_version = value["i"];
    } else if (key == ATTR_MODEL_ATTR_NAME_ENUM) {
      idx2name = value["list"]["s"].get<std::vector<string>>();
    } else if (key == ATTR_MODEL_ATTR_VALUE_ENUM) {
      idx2value = value["list"]["s"].get<std::vector<string>>();
    } else if (key == ATTR_MODEL_ATTRS_USE_STRING_VALUE) {
      use_string_val = value["list"]["b"].get<std::vector<bool>>();
    } else {
      obj_del = false;
    }

    if (obj_del) {
      it = json["attr"].erase(it);
    } else {
      it++;
    }
  }

  return om_compress_version;
}

int Pb2Json::AttrReplaceKV(Json &json, const std::vector<string> &idx2name,
                           const std::vector<string> &idx2value,
                           const std::vector<bool> &use_string_val) {
  if (!json.is_array() && !json.is_object()) {
    return 0;
  }

  if (json.find("key") != json.end() && json.find("value") != json.end()) {
    auto &key = json["key"];
    auto &value = json["value"];

    bool is_value_string = false;
    std::string attr_name;
    auto ret = EnumAttrUtils::GetAttrName(idx2name, use_string_val, key, attr_name, is_value_string);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERROR("E19999", "Key convert failed.");
      return -1;
    }
    key = attr_name;

    if (!is_value_string) {
      return 0;
    }

    if (value.find("i") != value.end()) { // value->i
      std::string attr_value;
      ret = EnumAttrUtils::GetAttrValue(idx2value, value["i"], attr_value);
      value["s"] = attr_value;
      value.erase("i");
    }
    if (value.find("list") != value.end()) { // value->list
      if (value["list"].find("i") != value["list"].end()) { // list->i
        std::vector<std::string> attr_values;
        ret = EnumAttrUtils::GetAttrValues(idx2value, value["list"]["i"], attr_values);
        value["list"]["s"] = attr_values;
        value["list"].erase("i");
      }
      if (value["list"].find("val_type") != value["list"].end()) { // list->val_type
        value["list"]["val_type"] = "VT_LIST_STRING";
      }
    }

    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERROR("E19999", "Value of \"%s\" convert failed.", attr_name.c_str());
      return -1;
    }
  }

  for (auto &sub_json : json) {
    if (AttrReplaceKV(sub_json, idx2name, idx2value, use_string_val) < 0) {
      GELOGE(FAILED, "EnumJson convert failed.");
      return  -1;
    }
  }
  return 0;
}

void Pb2Json::EnumJson2Json(Json &json) {
  std::vector<string> idx2name;
  std::vector<string> idx2value;
  std::vector<bool> use_string_val;
  if (DictInit(json, idx2name, idx2value, use_string_val) != USE_OM_COMPRESS) {
    return;
  }
  AttrReplaceKV(json, idx2name, idx2value, use_string_val);
}
}  //  namespace ge
