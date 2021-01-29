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

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_FUNCTIONDEF_H
#define GE_GRAPH_OPTIMIZE_GRAPH_FUNCTIONDEF_H

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/anchor.h"
#include "graph/ge_attr_value.h"
#include "graph/graph.h"
#include "proto/tensorflow/graph.pb.h"
#include "register/register_error_codes.h"

using domi::tensorflow::AttrValue;
using domi::tensorflow::AttrValue_ListValue;
using domi::tensorflow::DataType;
using domi::tensorflow::DT_INVALID;
using domi::tensorflow::FunctionDef;
using domi::tensorflow::FunctionDefLibrary;
using domi::tensorflow::NodeDef;
using std::string;
using std::to_string;
using std::vector;

namespace ge {
class GraphToFunctionDef {
 public:
  static domi::Status RecordArg(ge::ComputeGraphPtr graph,
                          const vector<ge::InDataAnchorPtr> &in_anchor);

  static domi::Status RecordResult(ge::ComputeGraphPtr graph,
                             const vector<ge::OutDataAnchorPtr> &out_anchor);

  static domi::Status DavGraphToFunctionDef(ge::ComputeGraphPtr graph,
                                      const string &name, FunctionDef *fdef);

  static domi::Status BuildFunctionDef(ge::ComputeGraphPtr &graph,
                                 const string &nme_in,
                                 FunctionDefLibrary *library,
                                 NodeDef *call_node_def,
                                 vector<ge::InDataAnchorPtr> &in_anchor,
                                 vector<ge::OutDataAnchorPtr> &out_anchor);

  static bool FindAttrValue(const domi::tensorflow::NodeDef *nodeDef,
                            const string attr_name,
                            domi::tensorflow::AttrValue &attr_value);

  static void AddNodeAttr(const string &attr_name,
                          const domi::tensorflow::AttrValue &value,
                          domi::tensorflow::NodeDef *node_def);
};

class NameMapHelper {
 public:
  NameMapHelper() = default;

  ~NameMapHelper() {}

  string UniqueInputOrOutputName(const string &name);

  string UniqueNodeName(const string &name);

  string Renormalize(const string &name) const;

 private:
  string GetUniqueName(const string &name);

  std::set<string> used_names_;
  std::map<string, string> name_mapping_;
};
}  // namespace ge

#endif  // GE_GRAPH_OPTIMIZE_GRAPH_FUNCTIONDEF_H
