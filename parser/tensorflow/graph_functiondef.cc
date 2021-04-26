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

#include "graph_functiondef.h"
#include <iostream>
#include "common/fmk_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/omg/parser/parser_types.h"
#include "parser/common/acl_graph_parser_util.h"
#include "common/types_map.h"
#include "common/util.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/ge_inner_error_codes.h"

namespace {
constexpr char UNKNOWN[] = "unknown";
constexpr char UNDERLINE = '_';
}  // namespace
namespace ge {
using AttrValueMap = ::google::protobuf::Map<string, domi::tensorflow::AttrValue>;
vector<domi::tensorflow::DataType> arg_datetypes_;
vector<domi::tensorflow::DataType> result_datetypes_;

string NameMapHelper::GetUniqueName(const string &name) {
  if (used_names_.insert(name).second) {
    return name;
  }
  int i = 0;
  while (true) {
    const string candidate = name + "_" + to_string(i);
    if (used_names_.insert(candidate).second) {
      return candidate;
    }
    ++i;
  }
}

string NameMapHelper::UniqueInputOrOutputName(const string &name) {
  // Normalize first
  string normalized = name;
  if (name.empty()) {
    normalized = UNKNOWN;
  }
  for (auto ch : normalized) {
    if (!isalnum(ch)) {
      ch = UNDERLINE;
    } else if (isupper(ch)) {
      ch = tolower(ch);
    }
  }
  // uniquify
  const string unique_name = GetUniqueName(normalized);
  name_mapping_[name] = unique_name;
  return unique_name;
}

string NameMapHelper::UniqueNodeName(const string &name) {
  // uniquify
  const string unique_name = GetUniqueName(name);
  name_mapping_[name] = unique_name;
  return unique_name;
}

string NameMapHelper::Renormalize(const string &name) const {
  const auto iter = name_mapping_.find(name);
  if (iter == name_mapping_.end()) return string();
  return iter->second;
}

domi::Status ComputeArgRange(const domi::tensorflow::NodeDef &node_def, const domi::tensorflow::OpDef::ArgDef &arg_def,
                             const domi::tensorflow::OpDef &op_def, int *num) {
  GE_CHECK_NOTNULL(num);
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "num" times.
    domi::tensorflow::AttrValue attr_value;
    // Get attribute number_att, if the attribute does not exist, return failure
    GE_IF_BOOL_EXEC(
      !GraphToFunctionDef::FindAttrValue(&node_def, arg_def.number_attr(), attr_value),
      GELOGE(domi::INTERNAL_ERROR, "In NodeDef %s Attr number_attr is not exist.", node_def.name().c_str());
      REPORT_INNER_ERROR("E19999", "Attr:number_attr not exist in node:%s, check invalid", node_def.name().c_str());
      return domi::INTERNAL_ERROR);
    *num = attr_value.i();
  } else if (!arg_def.type_list_attr().empty()) {
    domi::tensorflow::AttrValue attr_value;
    /// Get the attribute type_list_attr, if the attribute does not exist, return
    /// failure
    GE_IF_BOOL_EXEC(
      !GraphToFunctionDef::FindAttrValue(&node_def, arg_def.type_list_attr(), attr_value),
      GELOGE(domi::INTERNAL_ERROR, "In NodeDef %s Attr type_list_attr is not exist.", node_def.name().c_str());
      REPORT_INNER_ERROR("E19999", "Attr:type_list_attr not exist in node:%s, check invalid", node_def.name().c_str());
      return domi::INTERNAL_ERROR);
    *num = attr_value.list().type_size();
  } else if ((!arg_def.type_attr().empty()) || (arg_def.type() != DT_INVALID)) {
    *num = 1;
  } else {
    GELOGE(domi::INTERNAL_ERROR, "In NodeDef %s Attr type_list_attr is not exist.", node_def.name().c_str());
    REPORT_INNER_ERROR("E19999", "arg_def for node:%s is invalid, number_attr type_list_attr type_attr all empty",
                       node_def.name().c_str());
    return domi::INTERNAL_ERROR;
  }
  return SUCCESS;
}

using NameRangeMap = std::map<string, std::pair<int, int>>;

domi::Status NameRangesHelper(const domi::tensorflow::NodeDef &node_def,
                              const google::protobuf::RepeatedPtrField<domi::tensorflow::OpDef_ArgDef> &args,
                              const domi::tensorflow::OpDef &op_def, NameRangeMap *result) {
  GE_CHECK_NOTNULL(result);
  int start = 0;
  int num = 0;
  for (const auto &arg : args) {
    GE_RETURN_IF_ERROR(ComputeArgRange(node_def, arg, op_def, &num));
    (*result)[arg.name()] = std::make_pair(start, start + num);
    start += num;
  }
  return SUCCESS;
}

domi::Status NameRangesForNode(const domi::tensorflow::NodeDef &node_def, const domi::tensorflow::OpDef &op_def,
                               NameRangeMap *outputs) {
  GE_IF_BOOL_EXEC(outputs == nullptr, return FAILED);

  return NameRangesHelper(node_def, op_def.output_arg(), op_def, outputs);
}

domi::Status RemapFunctionDef(FunctionDef *fdef, const string &name, NameMapHelper &node_names,
                              std::map<string, string> &tensor_renaming,
                              std::map<string, string> &return_values) {
  GE_CHECK_NOTNULL(fdef);
  // Detect missing function inputs..
  for (int i = 0; i < fdef->signature().input_arg_size(); ++i) {
    const string &input_name = fdef->signature().input_arg(i).name();
    GE_IF_BOOL_EXEC(input_name.empty(),
                    REPORT_INNER_ERROR("E19999", "In fdef %s, index:%d input_name is empty, check invalid",
                                       fdef->signature().name().c_str(), i);
                    GELOGE(domi::INTERNAL_ERROR, "In fdef %s  input_name null .", fdef->signature().name().c_str());
                    return domi::INTERNAL_ERROR);
  }

  /// Remap input names.  We do this as a second pass to allow the nodes to be in
  /// any order.
  for (int n_index = 0; n_index < fdef->node_def_size(); ++n_index) {
    NodeDef *node_def = fdef->mutable_node_def(n_index);
    for (int i = 0; i < node_def->input_size(); ++i) {
      if (node_def->input(i).find("^") != string::npos) {
        // Control input
        const string normalized = node_names.Renormalize(node_def->input(i).substr(1));

        GE_IF_BOOL_EXEC(normalized.empty(),
                        REPORT_INNER_ERROR("E19999", "Could not remap control input %s of node %s in function %s",
                                           node_def->input(i).c_str(), node_def->name().c_str(), name.c_str());
                        GELOGE(domi::INTERNAL_ERROR, "Could not remap control input %s of node %s in function %s .",
                               node_def->input(i).c_str(), node_def->name().c_str(), name.c_str());
                        return domi::INTERNAL_ERROR);

        *node_def->mutable_input(i) = "^" + normalized;
      } else {
        const auto iter = tensor_renaming.find(node_def->input(i));

        GE_IF_BOOL_EXEC(iter == tensor_renaming.end(),
                        REPORT_INNER_ERROR("E19999", "Could not remap input %s of node %s in function %s",
                                           node_def->input(i).c_str(), node_def->name().c_str(), name.c_str());
                        GELOGE(domi::INTERNAL_ERROR, "Could not remap input %s of node %s in function %s .",
                               node_def->input(i).c_str(), node_def->name().c_str(), name.c_str());
                        return domi::INTERNAL_ERROR);

        *node_def->mutable_input(i) = iter->second;
      }
    }
  }

  // Remap return values.
  for (int r = 0; r < fdef->signature().output_arg_size(); ++r) {
    const string &ret_name = fdef->signature().output_arg(r).name();

    GE_IF_BOOL_EXEC(ret_name.empty(),
                    REPORT_INNER_ERROR("E19999", "Missing output %d to function %s", r, name.c_str());
                    GELOGE(domi::INTERNAL_ERROR, "Missing output %d to function %s .", r, name.c_str());
                    return domi::INTERNAL_ERROR);

    const string &return_value = return_values[ret_name];

    GE_IF_BOOL_EXEC(return_value.empty(),
                    REPORT_INNER_ERROR("E19999", "Could not remap return value %d ,%s of %s in function %s", r,
                                       ret_name.c_str(), return_value.c_str(), name.c_str());
                    GELOGE(domi::INTERNAL_ERROR, "Could not remap return value %d ,%s of %s in function %s .", r,
                           ret_name.c_str(), return_value.c_str(), name.c_str());
                    return domi::INTERNAL_ERROR);

    const auto iter = tensor_renaming.find(return_value);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(iter == tensor_renaming.end(),
                                   REPORT_INNER_ERROR("E19999", "can not find value[%s] in tensor_renaming map",
                                                      return_value.c_str());
                                   return domi::INTERNAL_ERROR,
                                   "can not find value[%s] in tensor_renaming map.", return_value.c_str());

    (*fdef->mutable_ret())[ret_name] = iter->second;
  }

  return SUCCESS;
}

// Add output operator for graph before converting func
domi::Status GraphToFunctionDef::RecordResult(ge::ComputeGraphPtr graph,
                                              const vector<ge::OutDataAnchorPtr> &out_anchor) {
  GE_CHECK_NOTNULL(graph);
  int32_t index = 0;
  result_datetypes_.clear();
  for (const auto &anchor : out_anchor) {
    GE_CHECK_NOTNULL(anchor);
    GE_CHECK_NOTNULL(anchor->GetOwnerNode()->GetOpDesc());
    int32_t type = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(anchor->GetIdx()).GetDataType();
    auto iter = GE_TENSORFLOW_DATA_TYPE_MAP.find(type);
    GE_IF_BOOL_EXEC(iter == GE_TENSORFLOW_DATA_TYPE_MAP.end(),
                    REPORT_INNER_ERROR("E19999", "datatype:%d of output:%d in node:%s:%s is not supported",
                                       type, anchor->GetIdx(), anchor->GetOwnerNode()->GetName().c_str(),
                                       anchor->GetOwnerNode()->GetName().c_str());
                    GELOGE(PARAM_INVALID, "data_type %d not supported", type);
                    return PARAM_INVALID);
    int32_t dtype = iter->second;

    string op_name = anchor->GetOwnerNode()->GetName() + "_" + to_string(anchor->GetIdx()) + "_retval";
    ge::OpDescPtr op = nullptr;
    GE_MAKE_SHARED(op = std::make_shared<ge::OpDesc>(op_name, ge::parser::NETOUTPUT), return FAILED);
    graphStatus status = op->AddInputDesc(ge::GeTensorDesc());
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed", op->GetName().c_str(), op->GetType().c_str());
      GELOGE(FAILED, "Add input desc for op:%s failed.", op->GetName().c_str());
      return FAILED;
    }
    status = op->AddOutputDesc(ge::GeTensorDesc());
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed", op->GetName().c_str(), op->GetType().c_str());
      GELOGE(FAILED, "Add output desc for op:%s failed.", op->GetName().c_str());
      return FAILED;
    }
    (void)ge::AttrUtils::SetInt(op, "T", static_cast<int32_t >(dtype));
    (void)ge::AttrUtils::SetInt(op, "ret_index", static_cast<int32_t >(index));
    ge::NodePtr res_node = graph->AddNode(op);
    GE_CHECK_NOTNULL(res_node);
    bool node_exists = false;
    for (const ge::NodePtr &node : graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node);
      if (node->GetName() == anchor->GetOwnerNode()->GetName()) {
        ge::OutDataAnchorPtr out_archor_ptr = node->GetOutDataAnchor(anchor->GetIdx());
        GE_CHECK_NOTNULL(out_archor_ptr);
        ge::InDataAnchorPtr in_archor_ptr = res_node->GetInDataAnchor(0);
        GE_CHECK_NOTNULL(in_archor_ptr);
        ge::graphStatus ret = ge::GraphUtils::AddEdge(out_archor_ptr, in_archor_ptr);
        if (ret != ge::GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                            out_archor_ptr->GetOwnerNode()->GetName().c_str(),
                            out_archor_ptr->GetOwnerNode()->GetType().c_str(), out_archor_ptr->GetIdx(),
                            in_archor_ptr->GetOwnerNode()->GetName().c_str(),
                            in_archor_ptr->GetOwnerNode()->GetType().c_str(), in_archor_ptr->GetIdx());
          GELOGE(domi::INTERNAL_ERROR, "Add edge failed,src op:%s,dst op:%s", node->GetName().c_str(),
                 res_node->GetName().c_str());
          return FAILED;
        }
        node_exists = true;
      }
    }
    GE_IF_BOOL_EXEC(!node_exists,
                    GELOGE(FAILED, "node not exists!");
                    REPORT_CALL_ERROR("E19999", "Node:%s(%s) not found in graph:%s, check invalid",
                                      anchor->GetOwnerNode()->GetName().c_str(),
                                      anchor->GetOwnerNode()->GetType().c_str(),
                                      graph->GetName().c_str());
                    return FAILED);
    result_datetypes_.emplace_back(domi::tensorflow::DataType(dtype));

    index++;
  }
  return SUCCESS;
}

/// Add input operator for graph before converting function.
/// Input operator will generate input parameters during function conversion
domi::Status GraphToFunctionDef::RecordArg(ge::ComputeGraphPtr graph, const vector<ge::InDataAnchorPtr> &in_anchor) {
  GE_CHECK_NOTNULL(graph);
  int32_t index = 0;
  arg_datetypes_.clear();
  for (const auto &anchor : in_anchor) {
    GE_CHECK_NOTNULL(anchor);
    GE_CHECK_NOTNULL(anchor->GetOwnerNode()->GetOpDesc());
    auto tensor_desc_ptr = anchor->GetOwnerNode()->GetOpDesc()->GetInputDescPtr(anchor->GetIdx());
    GE_CHECK_NOTNULL_EXEC(tensor_desc_ptr, return domi::FAILED);

    int32_t type = tensor_desc_ptr->GetDataType();
    auto iter = GE_TENSORFLOW_DATA_TYPE_MAP.find(type);
    GE_IF_BOOL_EXEC(iter == GE_TENSORFLOW_DATA_TYPE_MAP.end(),
                    REPORT_INNER_ERROR("E19999", "datatype:%d of input:%d in node:%s:%s is not supported",
                                       type, anchor->GetIdx(), anchor->GetOwnerNode()->GetName().c_str(),
                                       anchor->GetOwnerNode()->GetName().c_str());
                    GELOGE(PARAM_INVALID, "data_type %d not supported", type);
                    return PARAM_INVALID);
    int32_t dtype = iter->second;

    GE_CHECK_NOTNULL(anchor->GetPeerOutAnchor());
    string op_name = anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName() + "_" +
                     to_string(anchor->GetPeerOutAnchor()->GetIdx()) + "_arg";
    ge::OpDescPtr op = nullptr;
    GE_MAKE_SHARED(op = std::make_shared<ge::OpDesc>(op_name, ge::parser::DATA), return FAILED);
    graphStatus status = op->AddOutputDesc(ge::GeTensorDesc());
    if (status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed", op->GetName().c_str(), op->GetType().c_str());
      GELOGE(FAILED, "Add output desc for op:%s failed.", op->GetName().c_str());
      return FAILED;
    }

    (void)ge::AttrUtils::SetInt(op, "T", (int32_t)dtype);
    (void)ge::AttrUtils::SetInt(op, "arg_index", (int32_t)index);
    ge::NodePtr arg_node = graph->AddNode(op);
    GE_CHECK_NOTNULL(arg_node);
    bool node_exists = false;
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetName() == anchor->GetOwnerNode()->GetName()) {
        ge::OutDataAnchorPtr out_archor_ptr = arg_node->GetOutDataAnchor(0);
        GE_CHECK_NOTNULL(out_archor_ptr);
        ge::InDataAnchorPtr in_archor_ptr = node->GetInDataAnchor(anchor->GetPeerOutAnchor()->GetIdx());
        GE_CHECK_NOTNULL(in_archor_ptr);
        (void)ge::GraphUtils::RemoveEdge(in_archor_ptr->GetPeerOutAnchor(), in_archor_ptr);
        ge::graphStatus ret = ge::GraphUtils::AddEdge(out_archor_ptr, in_archor_ptr);
        if (ret != ge::GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                            out_archor_ptr->GetOwnerNode()->GetName().c_str(),
                            out_archor_ptr->GetOwnerNode()->GetType().c_str(), out_archor_ptr->GetIdx(),
                            in_archor_ptr->GetOwnerNode()->GetName().c_str(),
                            in_archor_ptr->GetOwnerNode()->GetType().c_str(), in_archor_ptr->GetIdx());
          GELOGE(domi::INTERNAL_ERROR, "Add edge failed,src op:%s,dst op:%s", arg_node->GetName().c_str(),
                 node->GetName().c_str());
          return FAILED;
        }
        node_exists = true;
      }
    }
    GE_IF_BOOL_EXEC(!node_exists,
                    REPORT_CALL_ERROR("E19999", "Node:%s(%s) not found in graph:%s, check invalid",
                                      anchor->GetOwnerNode()->GetName().c_str(),
                                      anchor->GetOwnerNode()->GetType().c_str(),
                                      graph->GetName().c_str());
                    GELOGE(FAILED, "node not exists!"); return FAILED);
    arg_datetypes_.emplace_back(domi::tensorflow::DataType(dtype));
    index++;
  }
  return SUCCESS;
}

// Convert Davinci's graph to tensorflow's functiondef
domi::Status GraphToFunctionDef::DavGraphToFunctionDef(ge::ComputeGraphPtr graph, const string &name,
                                                       FunctionDef *fdef) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(fdef);
  fdef->mutable_signature()->set_name(name);

  std::map<string, string> tensor_renaming;
  std::map<string, string> return_values;
  NameMapHelper node_names;

  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetOpDesc()->GetType() == ge::parser::DATA) {
      int64_t index = 0;

      int64_t type = 1;
      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(node->GetOpDesc(), "T", type), PARAM_INVALID,
                             "Get type attr failed");

      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(node->GetOpDesc(), "arg_index", index), PARAM_INVALID,
                             "Get arg_index attr failed");

      while (fdef->signature().input_arg_size() <= index) {
        fdef->mutable_signature()->add_input_arg();
      }
      domi::tensorflow::OpDef::ArgDef *argdef = fdef->mutable_signature()->mutable_input_arg(index);
      argdef->set_type(domi::tensorflow::DataType(type));
      const string normalized = node_names.UniqueInputOrOutputName(node->GetName());
      argdef->set_name(normalized);
      tensor_renaming[node->GetName() + ":0"] = normalized;
      continue;
    }

    if (node->GetOpDesc()->GetType() == ge::parser::NETOUTPUT) {
      int64_t index = 0;
      int64_t type = 1;

      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(node->GetOpDesc(), "T", type), PARAM_INVALID,
                             "Get type attr failed");

      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(node->GetOpDesc(), "ret_index", index), PARAM_INVALID,
                             "Get arg_index attr failed");

      while (fdef->signature().output_arg_size() <= index) {
        fdef->mutable_signature()->add_output_arg();
      }

      domi::tensorflow::OpDef::ArgDef *argdef = fdef->mutable_signature()->mutable_output_arg(index);
      argdef->set_type(domi::tensorflow::DataType(type));
      const string normalized = node_names.UniqueInputOrOutputName(node->GetName());
      argdef->set_name(normalized);

      ge::OutDataAnchorPtr o_anchor = node->GetAllInDataAnchors().at(0)->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(o_anchor);
      string n_name = o_anchor->GetOwnerNode()->GetName() + ":" + to_string(o_anchor->GetIdx());
      return_values[normalized] = n_name;
      continue;
    }

    // Analysis of nodedef of original tensorflow
    ge::GeAttrValue::BYTES nodedef_bytes;
    GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetBytes(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_NODE_DEF, nodedef_bytes),
                           PARAM_INVALID, "Get type attr nodedef failed.");
    domi::tensorflow::NodeDef node_def_;
    GE_CHK_BOOL_RET_STATUS(node_def_.ParseFromArray(nodedef_bytes.GetData(), nodedef_bytes.GetSize()), PARAM_INVALID,
                           "parse nodedef failed.");

    // Analysis of opdef of original tensorflow
    string opdef_string;
    GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_OP_DEF, opdef_string),
                           PARAM_INVALID, "Get type attr op_def failed.");

    domi::tensorflow::OpDef op_def;
    GE_CHK_BOOL_RET_STATUS(op_def.ParseFromString(opdef_string), PARAM_INVALID, "parse op_def failed.");

    // add nodedef
    NodeDef *node_def = fdef->add_node_def();
    *node_def = node_def_;

    node_def->mutable_attr()->erase(ge::ATTR_NAME_FRAMEWORK_OP_DEF);
    node_def->mutable_attr()->erase(ge::ATTR_NAME_OUTPUT_TENSOR_DESC);
    node_def->mutable_attr()->erase(ge::ATTR_NAME_INPUT_TENSOR_DESC);
    // No device information required for framework
    node_def->clear_device();

    node_def->set_name(node_names.UniqueNodeName(node->GetName()));

    // Reset input names based on graph rather than the NodeDef.
    node_def->clear_input();

    // Edges, indexed by dst_input.
    vector<ge::InDataAnchorPtr> in_anchors;
    ge::InControlAnchorPtr in_control_anchor;

    for (const auto &anchor : node->GetAllInDataAnchors()) {
      if (static_cast<int>(in_anchors.size()) <= anchor->GetIdx()) {
        in_anchors.resize(anchor->GetIdx() + 1);
      }
      in_anchors[anchor->GetIdx()] = anchor;
    }

    // Add regular inputs
    for (auto anchor : in_anchors) {
      GE_IF_BOOL_EXEC(anchor == nullptr,
                      REPORT_INNER_ERROR("E19999", "Nonconsecutive input edges; missing input edge for node %s",
                                        node_def_.name().c_str());
                      GELOGE(domi::INTERNAL_ERROR, "Nonconsecutive input edges; missing input edge , for node %s .",
                             node_def_.name().c_str());
                      return domi::INTERNAL_ERROR);

      if (anchor->GetPeerOutAnchor() != nullptr) {
        string t_name =
          anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName() + ":" + to_string(anchor->GetPeerOutAnchor()->GetIdx());
        node_def->add_input(t_name);
      }
    }

    // Add control inputs
    GE_CHECK_NOTNULL(node->GetInControlAnchor());
    for (const auto &anchor : node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      node_def->add_input("^" + anchor->GetOwnerNode()->GetName());
    }

    // Populate tensor_renaming.
    NameRangeMap output_ranges;
    GE_RETURN_IF_ERROR(NameRangesForNode(node_def_, op_def, &output_ranges));

    for (const auto &output : output_ranges) {
      for (int i = output.second.first; i < output.second.second; ++i) {
        const string tensor_name = node_def->name() + ":" + output.first + ":" + to_string(i - output.second.first);
        tensor_renaming[(node->GetName() + ":" + to_string(i))] = tensor_name;
      }
    }
  }

  // Remap FunctionDef
  GE_RETURN_IF_ERROR(RemapFunctionDef(fdef, name, node_names, tensor_renaming, return_values));

  return SUCCESS;
}

void SetInputOut(NodeDef *call_node_def, vector<ge::InDataAnchorPtr> &in_anchor) {
  GE_CHK_BOOL_EXEC(call_node_def != nullptr, return, "call_node_def is null.");
  for (const auto &anchor : in_anchor) {
    if ((anchor != nullptr) && (anchor->GetPeerOutAnchor() != nullptr)) {
      call_node_def->add_input(anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName() + "_" +
                               to_string(anchor->GetPeerOutAnchor()->GetIdx()));
    }
  }
}

domi::Status GraphToFunctionDef::BuildFunctionDef(ge::ComputeGraphPtr &graph, const string &name_in,
                                                  FunctionDefLibrary *library, NodeDef *call_node_def,
                                                  vector<ge::InDataAnchorPtr> &in_anchor,
                                                  vector<ge::OutDataAnchorPtr> &out_anchor) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(library);
  GE_CHECK_NOTNULL(call_node_def);
  // Current date / time base on the current system
  string now_time = ge::parser::CurrentTimeInStr();
  static int i = 0;
  const string name = name_in + now_time + to_string(i);
  i++;
  // set node_def
  call_node_def->set_op(name);
  call_node_def->set_name(name);

  // Add func property
  domi::tensorflow::AttrValue value;
  domi::tensorflow::NameAttrList *function = value.mutable_func();
  function->set_name(name);
  *function->mutable_attr() = call_node_def->attr();
  GraphToFunctionDef::AddNodeAttr("function", value, call_node_def);

  // Add input for nodedef
  SetInputOut(call_node_def, in_anchor);

  // Add input and output nodes to the graph
  GE_RETURN_IF_ERROR(GraphToFunctionDef::RecordArg(graph, in_anchor));
  GE_RETURN_IF_ERROR(GraphToFunctionDef::RecordResult(graph, out_anchor));

  domi::tensorflow::AttrValue tin_value;
  domi::tensorflow::AttrValue tout_value;
  // Add tin tout attribute
  domi::tensorflow::AttrValue::ListValue list;
  for (auto type : arg_datetypes_) {
    tin_value.mutable_list()->clear_type();
    tin_value.mutable_list()->add_type(type);
  }
  if (!arg_datetypes_.empty()) {
    GraphToFunctionDef::AddNodeAttr("Tin", tin_value, call_node_def);
  }
  for (auto type : result_datetypes_) {
    tout_value.mutable_list()->clear_type();
    tout_value.mutable_list()->add_type(type);
  }
  if (!result_datetypes_.empty()) {
    GraphToFunctionDef::AddNodeAttr("Tout", tout_value, call_node_def);
  }
  // Convert DaVinci graph to functiondef
  FunctionDef *fdef = library->add_function();
  GE_RETURN_IF_ERROR(GraphToFunctionDef::DavGraphToFunctionDef(graph, name, fdef));

  return SUCCESS;
}

bool GraphToFunctionDef::FindAttrValue(const domi::tensorflow::NodeDef *node_def, const string attr_name,
                                       domi::tensorflow::AttrValue &attr_value) {
  if (node_def == nullptr) {
    GELOGE(PARAM_INVALID, "Input param node is nullptr.");
    return false;
  }
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue> &attr = node_def->attr();

  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    attr_value = it->second;
    return true;
  }

  return false;
}

void GraphToFunctionDef::AddNodeAttr(const string &attr_name, const domi::tensorflow::AttrValue &value,
                                     domi::tensorflow::NodeDef *node_def) {
  GE_CHK_BOOL_TRUE_EXEC_INFO(node_def == nullptr, return, "input parameter is null.");
  node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_name, value));
}
}  // namespace ge
