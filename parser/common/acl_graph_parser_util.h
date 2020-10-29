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

#ifndef ACL_GRAPH_PARSE_UTIL_
#define ACL_GRAPH_PARSE_UTIL_

#include <map>
#include <string>
#include "common/types.h"
#include "graph/utils/graph_utils.h"

namespace ge {
class AclGrphParseUtil {
 public:
  AclGrphParseUtil() {}
  virtual ~AclGrphParseUtil() {}
  domi::Status LoadOpsProtoLib();
  void SaveCustomCaffeProtoPath();
  domi::Status AclParserInitialize(const std::map<std::string, std::string> &options);
  domi::Status SetDefaultOutputNode(ge::Graph &graph);

 private:
  bool parser_initialized = false;
  domi::Status GetOutputLeaf(NodePtr node, std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info);
  void GetOutputNodesNameAndIndex(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                  std::vector<std::string> &output_nodes_name);
};
}  // namespace ge

#endif  // ACL_GRAPH_PARSE_UTIL_