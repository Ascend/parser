/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "parser/common/proto_file_parser.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <sys/types.h>
#include <unistd.h>
#include "common/string_util.h"
#include "common/util.h"
#include "parser/common/acl_graph_parser_util.h"
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"

using std::ifstream;
using std::vector;
using std::string;

namespace {
const char kMinNum = '0';
const char kMaxNum = '9';
const int kMinLineWordSize = 3;
const int kMinMessageLineWords = 2;
const int kMaxIdentifier = 536870912; // 2^29 - 1
const int kTmpFileNameLen = 16;
const int kMinRandomNum = 0;
const int kMaxRandomNum = 9;
const int kDecimalMulti = 10;
const int kOpenRetValue = 0;
const int kMessageNameIndex = 2;
const char *const kTmpPath = "/tmp";
const char *const kMessage = "message";
const char *const kLayerParameter = "LayerParameter";
const char *const kNetParameter = "NetParameter";
const char *const kStartBrace = "{";
const char *const kCloseBrace = "}";
const char *const kOptional = "optional";
const char *const kRepeated = "repeated";
const char *const kRequired = "required";

bool GetIdentifier(const std::string &line, int &identifier) {
  int size = line.size();
  auto pos = line.find("=");
  if (pos == std::string::npos) {
    return false;
  }
  for (int i = pos + 1; i < size; i++) {
    if (line[i] == ';') {
      break;
    }
    if (line[i] >= kMinNum && line[i] <= kMaxNum) {
      identifier = identifier * kDecimalMulti + line[i] - kMinNum;
    }
    if (identifier > kMaxIdentifier || identifier < 0) {
      return false;
    }
  }
  if (identifier == 0) {
    return false;
  }
  return true;
}

void GetName(const std::string &op_info, string &op_name) {
  op_name.assign(op_info);
  auto pos = op_name.find("=");
  if (pos != string::npos) {
      op_name = op_name.substr(0, pos);
  }
}

void GetOpParamInfo(const std::string &line, std::vector<std::string> &op_param_info) {
  std::istringstream string_stream(line);
  std::string temp;
  while (std::getline(string_stream, temp, ' ')) {
    if (temp.empty()) {
      continue;
    }
    op_param_info.emplace_back(std::move(temp));
  }
}

string GetMessageName(const std::string &line) {
  std::vector<std::string> op_param_info;
  GetOpParamInfo(line, op_param_info);
  string message_name;
  if (op_param_info.size() < kMinMessageLineWords) {
    message_name = "";
    return message_name;
  }
  message_name = op_param_info[1];
  auto pos = message_name.find(kStartBrace);
  if (pos != string::npos) {
    message_name = message_name.substr(0, pos);
  }
  return message_name;
}

string CreatTmpName(int len) {
  std::uniform_int_distribution<int> u(kMinRandomNum, kMaxRandomNum);
  std::default_random_engine e;
  e.seed(time(0));
  string tmp_name = "";
  for (int i = 0; i < len; i++) {
    tmp_name += std::to_string(u(e));
  }
  return tmp_name;
}

bool SaveIdentifierOpMapInfo(const string &line,  std::map<int, std::pair<string, string>> &identifier_op_map,
                             std::map<std::string, std::pair<int, string>> &op_identifier_map) {
  std::vector<std::string> op_param_info;
  GetOpParamInfo(line, op_param_info);
  int info_size = op_param_info.size();
  if (info_size < kMinLineWordSize) {
    REPORT_INNER_ERROR("E19999", "Words size:%d of line[%s] is less than kMinLineWordSize[%d].",
                       info_size, line.c_str(), kMinLineWordSize);
    GELOGE(ge::FAILED, "[Check][Size] Words size:%d of line[%s] is less than kMinLineWordSize[%d].",
           info_size, line.c_str(), kMinLineWordSize);
    return false;
  }

  if (op_param_info[0] != kOptional && op_param_info[0] != kRepeated && op_param_info[0] != kRequired) {
    REPORT_INNER_ERROR("E19999", "Split line[%s] failed.", line.c_str());
    GELOGE(ge::FAILED, "[Check][Param] Split line[%s] failed.", line.c_str());
    return false;
  }

  // get identifier
  int identifier = 0;
  bool ret = GetIdentifier(line, identifier);
  if (!ret) {
    GELOGE(ge::FAILED, "[Get][Identifier] of line[%s] failed.", line.c_str());
    return false;
  }

  // get op_name
  string name;
  GetName(op_param_info[kMessageNameIndex], name);

  identifier_op_map[identifier] = std::make_pair(op_param_info[1], name);
  op_identifier_map[name] = std::make_pair(identifier, op_param_info[1]);
  return true;
}

bool CheckRealPath(const char *file_path) {
  string dest_path = ge::parser::RealPath(file_path);
  if (dest_path.empty()) {
    GELOGW("Path [%s] is not real existed.", file_path);
    return false;
  }
  return true;
}
} // namespace

namespace ge {
ProtoFileParser::~ProtoFileParser() {
  if (!fusion_proto_path.empty() && CheckRealPath(fusion_proto_path.c_str())) {
    (void)remove(fusion_proto_path.c_str());
  }
}

std::string ProtoFileParser::GetFusionProtoFile() {
  return fusion_proto_path;
}

Status ProtoFileParser::CreatProtoFile() {
  if (fusion_proto_path.empty()) {
    fusion_proto_path.assign(kTmpPath);
    fusion_proto_path += "/" + CreatTmpName(kTmpFileNameLen);
  }

  int fd = open(fusion_proto_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP);
  if (fd < kOpenRetValue) {
    REPORT_INNER_ERROR("E19999", "creat tmp proto file[%s] failed.", fusion_proto_path.c_str());
    GELOGE(FAILED, "[Open][File] creat tmp proto file[%s] failed.", fusion_proto_path.c_str());
    return FAILED;
  }
  close(fd);
  return SUCCESS;
}

Status ProtoFileParser::ParseProtoFile(const string &proto_file,
                                       std::map<int, std::pair<string, string>> &identifier_op_map,
                                       std::map<std::string, std::pair<int, string>> &op_identifier_map) {
  ifstream read_file;
  read_file.open(proto_file, std::ios::in);
  if (read_file.fail()) {
    REPORT_INNER_ERROR("E19999", "ifsream open proto file[%s] failed.", proto_file.c_str());
    GELOGE(FAILED, "[Open][File] ifsream open proto file[%s] failed.", proto_file.c_str());
    return FAILED;
  }

  std::string line;
  bool save_flag = false;
  while (std::getline(read_file, line)) {
    if (line.find(kMessage) != std::string::npos && line.find(kLayerParameter) != std::string::npos) {
      save_flag = true;
      continue;
    }

    if (save_flag && line.find(kCloseBrace) != std::string::npos) {
      save_flag = false;
      break;
    }

    if (save_flag) {
      if (line.find(kRepeated) == std::string::npos && line.find(kOptional) == std::string::npos &&
          line.find(kRequired) == std::string::npos) {
        continue;
      }
      bool ret = SaveIdentifierOpMapInfo(line, identifier_op_map, op_identifier_map);
      if (!ret) {
        read_file.close();
        return FAILED;
      }
    }
  }
  read_file.close();
  return SUCCESS;
}

Status ProtoFileParser::AddCustomAndConflictLayer(const char *custom_proto_file, std::ofstream &write_tmp) {
  ifstream read_custom;
  read_custom.open(custom_proto_file, std::ios::in);
  if (read_custom.fail()) {
    REPORT_INNER_ERROR("E19999", "ifsream open custom proto file[%s] failed.", custom_proto_file);
    GELOGE(FAILED, "[Open][File] ifsream open custom proto file[%s] failed.", custom_proto_file);
    return FAILED;
  }

  std::string line_custom;
  bool custom_in_layer = false;
  while (std::getline(read_custom, line_custom)) {
    if (line_custom.find(kMessage) != std::string::npos && line_custom.find(kLayerParameter) != std::string::npos) {
      custom_in_layer = true;
      continue;
    }

    if (!custom_in_layer) {
      continue;
    }

    if (line_custom.find(kCloseBrace) != std::string::npos) {
      custom_in_layer = false;
      break;
    }
    // exclude remark lines
    if (line_custom.find(kRepeated) == std::string::npos && line_custom.find(kOptional) == std::string::npos &&
          line_custom.find(kRequired) == std::string::npos) {
        continue;
    }
    // exclude repeated lines
    if (custom_repeat_line_map_.count(line_custom) == 0) {
      write_tmp << line_custom << '\n';
    }
  }
  read_custom.close();
  return SUCCESS;
}

Status ProtoFileParser::AddCustomAndConflictMessage(const char *custom_proto_file, std::ofstream &write_tmp) {
  ifstream read_custom;
  read_custom.open(custom_proto_file, std::ios::in);
  if (read_custom.fail()) {
    REPORT_INNER_ERROR("E19999", "ifsream open custom proto file[%s] failed.", custom_proto_file);
    GELOGE(FAILED, "[Open][File] ifsream open custom proto file[%s] failed.", custom_proto_file);
    return FAILED;
  }

  std::string line_custom;
  bool custom_in_message = false;
  while (std::getline(read_custom, line_custom)) {
    if (line_custom.find(kMessage) != std::string::npos) {
      std::string message_name = GetMessageName(line_custom);
      if (message_name != kLayerParameter && message_name != kNetParameter) {
        custom_in_message = true;
        write_tmp << line_custom << '\n';
      } else {
        custom_in_message = false;
      }
      continue;
    }

    // exclude repeated messages
    if (custom_in_message) {
      write_tmp << line_custom << '\n';
    }
  }
  read_custom.close();
  return SUCCESS;
}

Status ProtoFileParser::WriteCaffeProtoFile(const char *custom_proto_file,
                                            std::ifstream &read_caffe,
                                            std::ofstream &write_tmp) {
  std::string line_caffe;
  bool caffe_in_layer = false;
  bool caffe_in_unrepeated_message = true;
  string tmp_message_name;
  while (std::getline(read_caffe, line_caffe)) {
    if (line_caffe.find(kMessage) != std::string::npos) {
      tmp_message_name.assign(GetMessageName(line_caffe));
      if (custom_repeat_message_map_.count(tmp_message_name) > 0) {
        caffe_in_unrepeated_message = false;
      } else {
        caffe_in_unrepeated_message = true;
        if (tmp_message_name == kLayerParameter) {
          caffe_in_layer = true;
        }
      }
    }
    if (!caffe_in_unrepeated_message) {
      continue;
    }
    if (caffe_in_layer && line_caffe.find(kCloseBrace) != std::string::npos) {
      if (AddCustomAndConflictLayer(custom_proto_file, write_tmp) != SUCCESS) {
        GELOGE(FAILED, "[Invoke][AddCustomAndConflictLayer] Add conflict and new layer line "
               "from custom proto to dest proto failed, protofile:%s.", custom_proto_file);
        return FAILED;
      }
      caffe_in_layer = false;
    }

    // exclude conflict lines
    if (caffe_in_layer && caffe_conflict_line_map_.count(line_caffe) > 0) {
      GELOGD("pass line: %s", line_caffe.c_str());
      continue;
    }
    write_tmp << line_caffe << '\n';
  }
  return SUCCESS;
}

Status ProtoFileParser::WriteProtoFile(const char *caffe_proto_file,
                                       const char *custom_proto_file) {
  std::ifstream read_caffe;
  std::ofstream write_tmp;
  read_caffe.open(caffe_proto_file, std::ios::in);
  if (read_caffe.fail()) {
    REPORT_INNER_ERROR("E19999", "ifsream open proto file[%s] failed.", caffe_proto_file);
    GELOGE(FAILED, "[Open][File] ifsream open proto file[%s] failed.", caffe_proto_file);
    return FAILED;
  }
  write_tmp.open(fusion_proto_path, std::ios::out);
  if (write_tmp.fail()) {
    REPORT_INNER_ERROR("E19999", "ofstream open proto file[%s] failed.", fusion_proto_path.c_str());
    GELOGE(FAILED, "[Open][File] ofstream open proto file[%s] failed.", fusion_proto_path.c_str());
    read_caffe.close();
    return FAILED;
  }

  if (WriteCaffeProtoFile(custom_proto_file, read_caffe, write_tmp) != SUCCESS) {
    read_caffe.close();
    write_tmp.close();
    return FAILED;
  }

  if (AddCustomAndConflictMessage(custom_proto_file, write_tmp) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][AddCustomAndConflictMessage] Add conflict and new message from custom proto "
           "to dest proto failed, proto file:%s.", custom_proto_file);
    read_caffe.close();
    write_tmp.close();
    return FAILED;
  }

  read_caffe.close();
  write_tmp.close();
  return SUCCESS;
}

Status ProtoFileParser::FindConflictLine(const char *proto_file, int identifier,
                                         std::string &dest_line) {
  ifstream read_file;
  read_file.open(proto_file, std::ios::in);
  if (read_file.fail()) {
    REPORT_INNER_ERROR("E19999", "open file[%s] failed.", proto_file);
    GELOGE(FAILED, "[Open][File] [%s] failed.", proto_file);
    return FAILED;
  }

  std::string line;
  bool save_flag = false;
  while (std::getline(read_file, line)) {
    if (line.find(kMessage) != std::string::npos && line.find(kLayerParameter) != std::string::npos) {
      save_flag = true;
      continue;
    }

    if (save_flag && line.find(kCloseBrace) != std::string::npos) {
      save_flag = false;
      break;
    }

    int tmp_identifier = 0;
    if (save_flag && GetIdentifier(line, tmp_identifier) && tmp_identifier == identifier) {
      dest_line.assign(line);
      read_file.close();
      return SUCCESS;
    }
  }
  read_file.close();
  REPORT_INNER_ERROR("E19999", "find line according to identifier[%d] failed.", identifier);
  GELOGE(FAILED, "[Find][Line] according to identifier[%d] failed.", identifier);
  return FAILED;
}

void ProtoFileParser::CheckConflictOp(const char *caffe_proto_file, const char *custom_proto_file,
                                      std::map<std::string, std::pair<int, string>> &caffe_op_identifier_map,
                                      std::map<std::string, std::pair<int, string>> &custom_op_identifier_map) {
  for (auto iter = custom_op_identifier_map.begin(); iter != custom_op_identifier_map.end(); ++iter) {
    if (caffe_op_identifier_map.count(iter->first) > 0) {
      string message_name = iter->first;
      auto caffe_pair = caffe_op_identifier_map[iter->first];
      auto custom_pair = custom_op_identifier_map[iter->first];
      if (caffe_pair.first != custom_pair.first || caffe_pair.second != custom_pair.second) {
        // consider conflict op and name and type;
        GELOGD("Find conflict op: caffe_identifier[%d], custom_identifier[%d], op_name[%s].",
               caffe_pair.first, custom_pair.first, message_name.c_str());
        std::string caffe_conflict_line;
        (void)FindConflictLine(caffe_proto_file, caffe_pair.first, caffe_conflict_line);
        GELOGD("conflict: %s", caffe_conflict_line.c_str());
        caffe_conflict_line_map_[caffe_conflict_line]++;
      } else {
        // consider repeat op and name and type; could be removed
        std::string custom_repeat_line;
        (void)FindConflictLine(custom_proto_file, caffe_pair.first, custom_repeat_line);
        custom_repeat_line_map_[custom_repeat_line]++;
        GELOGD("repeat: %s", custom_repeat_line.c_str());
      }
    }
  }
}

void ProtoFileParser::CheckConflictIdentifier(const char *caffe_proto_file, const char *custom_proto_file,
                                              std::map<int, std::pair<string, string>> caffe_identifier_op_map,
                                              std::map<int, std::pair<string, string>> custom_identifier_op_map) {
  for (auto iter = custom_identifier_op_map.begin(); iter != custom_identifier_op_map.end(); ++iter) {
    if (caffe_identifier_op_map.count(iter->first) > 0) {
      int identifier = iter->first;
      auto caffe_pair = caffe_identifier_op_map[iter->first];
      auto custom_pair = custom_identifier_op_map[iter->first];
      if (caffe_pair.first != custom_pair.first || caffe_pair.second != custom_pair.second) {
        // consider conflict op and name and type;
        GELOGD("Find conflict op: caffe_op[%s], custom_op[%s], identifier[%d].",
               caffe_pair.first.c_str(), custom_pair.first.c_str(),
               identifier);
        std::string caffe_conflict_line;
        (void)FindConflictLine(caffe_proto_file, identifier, caffe_conflict_line);
        GELOGD("conflict: %s", caffe_conflict_line.c_str());
        caffe_conflict_line_map_[caffe_conflict_line]++;
      } else {
        // consider repeat op and name and type;
        std::string custom_repeat_line;
        (void)FindConflictLine(custom_proto_file, identifier, custom_repeat_line);
        custom_repeat_line_map_[custom_repeat_line]++;
        GELOGD("repeat: %s", custom_repeat_line.c_str());
      }
    }
  }
}

Status ProtoFileParser::RecordProtoMessage(const string &proto_file) {
  ifstream read_file;
  read_file.open(proto_file, std::ios::in);
  if (read_file.fail()) {
    REPORT_INNER_ERROR("E19999", "ifsream open proto file[%s] failed.", proto_file.c_str());
    GELOGE(FAILED, "[Open][File] ifsream open proto file[%s] failed.", proto_file.c_str());
    return FAILED;
  }

  std::string line;
  while (std::getline(read_file, line)) {
    if (line.find(kMessage) != std::string::npos) {
      std::string message_name = GetMessageName(line);
      if (message_name != kLayerParameter && message_name != kNetParameter) {
        custom_repeat_message_map_[message_name]++;
      }
    }
  }
  read_file.close();
  return SUCCESS;
}

Status ProtoFileParser::CombineProtoFile(const char *caffe_proto_file, const char *custom_proto_file,
                                         std::string &dest_proto_file) {
  GE_CHECK_NOTNULL(caffe_proto_file);
  GE_CHECK_NOTNULL(custom_proto_file);

  if (!CheckRealPath(caffe_proto_file) || !CheckRealPath(custom_proto_file)) {
    REPORT_CALL_ERROR("E19999", "caffe proto[%s] or custom proto[%s] is not existed.",
                      caffe_proto_file, custom_proto_file);
    GELOGE(FAILED, "[Check][Param] caffe proto[%s] or custom proto[%s] is not existed.",
           caffe_proto_file, custom_proto_file);
    return FAILED;
  }

  GELOGI("Start fusion custom and caffe proto to file.");
  std::map<int, std::pair<string, string>> caffe_identifier_op_map;
  std::map<int, std::pair<string, string>> custom_identifier_op_map;
  std::map<std::string, std::pair<int, string>> caffe_op_identifier_map;
  std::map<std::string, std::pair<int, string>> custom_op_identifier_map;

  (void)ParseProtoFile(caffe_proto_file, caffe_identifier_op_map, caffe_op_identifier_map);
  (void)ParseProtoFile(custom_proto_file, custom_identifier_op_map, custom_op_identifier_map);
  (void)RecordProtoMessage(custom_proto_file);

  // check identifier or op_type is same
  CheckConflictIdentifier(caffe_proto_file, custom_proto_file,
                          caffe_identifier_op_map, custom_identifier_op_map);
  CheckConflictOp(caffe_proto_file, custom_proto_file,
                  caffe_op_identifier_map, custom_op_identifier_map);

  if (CreatProtoFile() != SUCCESS) {
    return FAILED;
  }

  if (WriteProtoFile(caffe_proto_file, custom_proto_file) != SUCCESS) {
    GELOGE(FAILED, "[Write][ProtoFile] Combine caffe proto and custom proto to dest proto file failed.");
    return FAILED;
  }
  dest_proto_file.assign(fusion_proto_path);
  GELOGI("Fusion custom and caffe proto to file[%s] success.", dest_proto_file.c_str());
  return SUCCESS;
}
} // namespace ge