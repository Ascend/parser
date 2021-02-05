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

#ifndef PROTO_FILE_PARSE_UTIL_
#define PROTO_FILE_PARSE_UTIL_

#include <map>
#include <string>
#include "ge/ge_api_types.h"

namespace ge {
class ProtoFileParser {
public:
  ProtoFileParser(){};
  ProtoFileParser(const char *dest_path){
    fusion_proto_path = dest_path;
  }
  ~ProtoFileParser();
  Status CombineProtoFile(const char *caffe_proto_file, const char *custom_proto_file,
                          std::string &dest_proto_file);
  std::string GetFusionProtoFile();
private:
  Status CreatProtoFile();
  Status ParseProtoFile(const std::string &proto_file,
                        std::map<int, std::pair<std::string, std::string> > &identifier_op_map,
                        std::map<std::string, std::pair<int, std::string> > &op_identifier_map);
  Status WriteCaffeProtoFile(const char *custom_proto_file,
                             std::ifstream &read_caffe,
                             std::ofstream &write_tmp);
  Status WriteProtoFile(const char *caffe_proto_file, const char *custom_proto_file);
  Status FindConflictLine(const char *proto_file, int identifier,
                          std::string &dest_line);
  Status AddCustomAndConflictLayer(const char *custom_proto_file, std::ofstream &write_tmp);
  Status AddCustomAndConflictMessage(const char *custom_proto_file, std::ofstream &write_tmp);
  void CheckConflictOp(const char *caffe_proto_file, const char *custom_proto_file,
                       std::map<std::string, std::pair<int, std::string>> &caffe_op_identifier_map,
                       std::map<std::string, std::pair<int, std::string>> &custom_op_identifier_map);
  void CheckConflictIdentifier(const char *caffe_proto_file, const char *custom_proto_file,
                               std::map<int, std::pair<std::string, std::string>> caffe_identifier_op_map,
                               std::map<int, std::pair<std::string, std::string>> custom_identifier_op_map);
  Status RecordProtoMessage(const std::string &proto_file);
  std::map<std::string, int> caffe_conflict_line_map_;
  std::map<std::string, int> custom_repeat_line_map_;
  std::map<std::string, int> custom_repeat_message_map_;
  std::string fusion_proto_path;
};
}  // namespace ge

#endif  // PROTO_FILE_PARSE_UTIL_