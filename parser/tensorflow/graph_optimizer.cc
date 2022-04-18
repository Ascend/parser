/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "graph_optimizer.h"
#include "common/op_types.h"
#include "common/types_map.h"
#include "common/util.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "graph_functiondef.h"
#include "parser/common/acl_graph_parser_util.h"
#include "register/op_registry.h"

namespace ge {
REGISTER_OPTYPE_DEFINE(TF_MAXIMUM_GRAD, "MaximumGrad");
REGISTER_OPTYPE_DEFINE(TF_MATMUL, "Matmul");
REGISTER_OPTYPE_DEFINE(TFRELU6, "Relu6");
REGISTER_OPTYPE_DEFINE(TF_BATCH_MATMUL, "BatchMatmul");
}  // namespace ge

namespace ge {
namespace {
const char RRTVAL_NODE_NAME_SUFFIX[] = "_RetVal";
const char *const kShapeNodeType = "Shape";
const char *const kShapeNodeNamePrefix = "getnext_shape_";
const char *const kIteratorType = "Iterator";
const char *const kIteratorV2Type = "IteratorV2";
const char *const kGetNextType = "IteratorGetNext";
const char *const kDynGetNextType = "DynamicGetNext";
}  // namespace

Status ParserGraphOptimizer::FusionFmkop() {
  GELOGI("graph_optimizer.cpp && FustionFmkop()");
  GE_CHECK_NOTNULL(graph_);
  std::unordered_map<string, std::vector<NodePtr>> node_cluser_Map;
  GE_CHK_STATUS_RET(MarkForFusion(node_cluser_Map), "find framework node to be fused fail.");
  GE_IF_BOOL_EXEC(node_cluser_Map.empty(), return SUCCESS);

  for (auto it = node_cluser_Map.begin(); it != node_cluser_Map.end(); ++it) {
    GE_CHK_STATUS_RET(UpdateGraph(it->second), "fusion framework nodes failed. node：%s", (it->first).c_str());
  }
  // fuse all fmkop and then delete node
  for (auto it = node_cluser_Map.begin(); it != node_cluser_Map.end(); ++it) {
    for (auto node : it->second) {
      GE_CHK_STATUS_RET(GraphUtils::IsolateNode(node, {}), "Isolate removed node: %s, type: %s failed",
                        node->GetName().c_str(), node->GetType().c_str());
      GE_CHK_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph_, node),
                        "Remove node: %s, type: %s without relink failed", node->GetName().c_str(),
                        node->GetType().c_str());
    }
  }

  return SUCCESS;
}

Status ParserGraphOptimizer::MarkForFusion(unordered_map<string, vector<NodePtr>> &node_cluster_map) {
  GE_CHECK_NOTNULL(graph_);
  bool has_get_next = false;
  bool has_dyn_get_next = false;
  for (auto node : graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == kDynGetNextType) {
      has_dyn_get_next = true;
      break;
    }
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != ge::parser::FRAMEWORK_OP_TYPE, continue);
    string type;
    GE_CHK_STATUS_RET(ge::parser::GetOriginalType(node, type));
    if (type == kGetNextType) {
      has_get_next = true;
      break;
    }
  }
  return GetFusionCluster(has_get_next, has_dyn_get_next, node_cluster_map);
}

Status ParserGraphOptimizer::GetFusionCluster(const bool has_get_next, const bool has_dyn_get_next,
                                              unordered_map<string, vector<NodePtr>> &node_cluster_map) {
  GE_CHECK_NOTNULL(graph_);
  for (auto node : graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != ge::parser::FRAMEWORK_OP_TYPE, continue)
    string type;
    GE_CHK_STATUS_RET(ge::parser::GetOriginalType(node, type));
    if (type == kGetNextType) {
      vector<NodePtr> temp_node_cluser;
      for (auto in_anchor : node->GetAllInDataAnchors()) {
        OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
        GE_CHECK_NOTNULL(peer_out_anchor);
        NodePtr src_node = peer_out_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(src_node);
        temp_node_cluser.push_back(src_node);
      }
      temp_node_cluser.push_back(node);
      for (auto out_anchor : node->GetAllOutDataAnchors()) {
        GE_CHECK_NOTNULL(out_anchor);
        for (auto in_anchor : out_anchor->GetPeerInDataAnchors()) {
          GE_CHECK_NOTNULL(in_anchor);
          NodePtr dst_node = in_anchor->GetOwnerNode();
          GE_CHECK_NOTNULL(dst_node);
          GE_CHECK_NOTNULL(dst_node->GetOpDesc());
          if ((dst_node->GetName().find(kShapeNodeNamePrefix) != std::string::npos) &&
              (dst_node->GetOpDesc()->GetType() == kShapeNodeType)) {
            temp_node_cluser.emplace_back(dst_node);
          }
        }
      }
      if (temp_node_cluser.size() > 1) {
        vector<NodePtr> node_cluser;
        node_cluser.assign(temp_node_cluser.begin(), temp_node_cluser.end());
        node_cluster_map[temp_node_cluser[0]->GetName()] = node_cluser;
      }
      temp_node_cluser.clear();
      GELOGI("MarkForFusion, IteratorGetNext graph mark success.");
    }

    const bool dataset_init = (!has_get_next) && (!has_dyn_get_next) &&
                              ((type == kIteratorType) || (type == kIteratorV2Type));
    if (dataset_init) {
      GE_CHK_STATUS_RET(FindFmkNodeCluser(node_cluster_map), "find framework node to be fused fail.");
      GELOGI("MarkForFusion, Iterator init graph mark success.");
    }
  }
  return SUCCESS;
}

// find frameworkOP
Status ParserGraphOptimizer::FindFmkNodeCluser(unordered_map<string, vector<NodePtr>> &node_cluser_Map) const {
  vector<NodePtr> temp_node_cluser;

  for (auto node : graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr temp_node_desc_ptr = node->GetOpDesc();
    GE_CHECK_NOTNULL(temp_node_desc_ptr);
    GE_IF_BOOL_EXEC(temp_node_desc_ptr->GetType() == ge::parser::DATA_TYPE, continue);

    if (temp_node_desc_ptr->GetType() == ge::parser::FRAMEWORK_OP_TYPE &&
        (temp_node_desc_ptr->GetName().find(RRTVAL_NODE_NAME_SUFFIX) == string::npos)) {
      temp_node_cluser.push_back(node);
    } else {
      if (temp_node_cluser.size() > 1) {
        vector<NodePtr> node_cluser;
        node_cluser.assign(temp_node_cluser.begin(), temp_node_cluser.end());
        node_cluser_Map[temp_node_cluser[0]->GetName()] = node_cluser;
      }
      temp_node_cluser.clear();
    }
  }
  if (temp_node_cluser.size() > 1) {
    vector<NodePtr> node_cluser;
    node_cluser.assign(temp_node_cluser.begin(), temp_node_cluser.end());
    node_cluser_Map[temp_node_cluser[0]->GetName()] = node_cluser;
  }
  return SUCCESS;
}

Status CollectNodeFuncs(vector<ge::NodePtr> &nodes, FunctionDefLibrary *library) {
  for (auto node : nodes) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr opDef = node->GetOpDesc();
    string funcdefStr;
    ge::Buffer funcDefBytes;

    GE_IF_BOOL_EXEC(
        AttrUtils::GetBytes(opDef, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF, funcDefBytes), FunctionDefLibrary funcLib;
        GE_CHECK_NOTNULL(funcDefBytes.GetData());
        string str(PtrToPtr<uint8_t, char_t>(funcDefBytes.GetData()), funcDefBytes.GetSize());
        GELOGI("FUNCDEF: Get function -> %s.", str.c_str()); GE_IF_BOOL_EXEC(
            funcLib.ParseFromArray(funcDefBytes.GetData(), funcDefBytes.GetSize()), library->MergeFrom(funcLib)));
  }
  return SUCCESS;
}

Status ParserGraphOptimizer::UpdateGraph(vector<NodePtr> &nodes) {
  ComputeGraphPtr sub_graph = nullptr;
  GE_MAKE_SHARED(sub_graph = std::make_shared<ComputeGraph>("subGraph"), sub_graph = nullptr; return PARAM_INVALID);

  unordered_map<string, NodePtr> node_map;
  vector<InDataAnchorPtr> input_anchors;
  vector<OutDataAnchorPtr> output_anchors;
  map<OutDataAnchorPtr, vector<InDataAnchorPtr>> output_in_map;
  vector<InControlAnchorPtr> input_control_anchors;
  vector<OutControlAnchorPtr> output_control_anchors;

  GE_CHK_STATUS_RET(InsertNode(sub_graph, nodes, input_anchors, output_anchors, output_in_map, input_control_anchors,
                               output_control_anchors, node_map),
                    "insert node to sub_graph failed.");
  GE_CHK_STATUS_RET(LinkInnerAnchor(node_map), "Link inner anchor failed.");

  std::unique_ptr<NodeDef> node_def(new (std::nothrow) NodeDef());  // tensorflow NodeDef
  GE_CHECK_NOTNULL(node_def);
  std::unique_ptr<FunctionDefLibrary> func_def_lib(new (std::nothrow) FunctionDefLibrary());
  GE_CHECK_NOTNULL(func_def_lib);
  // convert graph to FunctionDef
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(nodes.size() == 0,
                                 REPORT_INNER_ERROR("E19999", "Param nodes size must greater than 0");
                                 return PARAM_INVALID, "node size must greater than 0 .");
  GE_CHK_STATUS_RET(CollectNodeFuncs(nodes, func_def_lib.get()), "Collect functionDef in nodes failed.");
  GE_CHK_STATUS_RET(GraphToFunctionDef::BuildFunctionDef(sub_graph, nodes[0]->GetName(), func_def_lib.get(),
                                                         node_def.get(), input_anchors, output_anchors),
                    "Build functiondef failed.");
  string nodefStr;
  string funcdefStr;

  GE_IF_BOOL_EXEC(!node_def->SerializeToString(&nodefStr),
                  REPORT_CALL_ERROR("E19999", "Serialize nodedef to string failed");
                  GELOGE(PARAM_INVALID, "Serialize nodedef to string failed.");
                  return PARAM_INVALID);

  GE_IF_BOOL_EXEC(!func_def_lib->SerializeToString(&funcdefStr),
                  REPORT_CALL_ERROR("E19999", "Serialize func_def to string failed, ");
                  GELOGE(PARAM_INVALID, "Serialize func_def to string failed.");
                  return PARAM_INVALID);

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(nodes.size() == 0, return PARAM_INVALID, "nodes is empty.");

  std::string fusion_op_name;
  for (auto node : nodes) {
    fusion_op_name += node->GetName();
  }

  const uint32_t kFusionOpNameMaxLen = 1024;
  if (fusion_op_name.size() > kFusionOpNameMaxLen) {
    fusion_op_name = nodes[0]->GetName();
  }

  OpDescPtr fusion_node_opdef = nullptr;
  GE_MAKE_SHARED(
      fusion_node_opdef = std::make_shared<OpDesc>(fusion_op_name, nodes[0]->GetOpDesc()->GetType()),
      fusion_node_opdef = nullptr;
      return FAILED);

  std::string type = "";
  GE_CHK_STATUS_RET(ge::parser::GetOriginalType(nodes[0], type));
  (void)AttrUtils::SetStr(fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);

  (void)AttrUtils::SetZeroCopyBytes(
      fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF,
      Buffer::CopyFrom(PtrToPtr<const char_t, const uint8_t>(funcdefStr.data()), funcdefStr.length()));
  (void)AttrUtils::SetZeroCopyBytes(
      fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
      Buffer::CopyFrom(PtrToPtr<const char_t, const uint8_t>(nodefStr.data()), nodefStr.length()));

  (void)AttrUtils::SetInt(fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, ge::GetParserContext().type);

  // reconstruct fusion_node and edges
  GE_CHK_STATUS_RET(RebuildOutputAnchors(output_anchors, fusion_node_opdef),
                    "rebuild output edges to fusion node failed.");
  GE_CHK_STATUS_RET(RebuildInputAnchors(input_anchors, fusion_node_opdef),
                    "rebuild input edges to fusion node failed.");
  NodePtr fusion_node = graph_->AddNode(fusion_node_opdef);

  // add Anchors
  GE_CHK_STATUS_RET(RebuildFusionNode(input_anchors, output_anchors, output_in_map, input_control_anchors,
                                      output_control_anchors, fusion_node),
                    "rebuild node failed!");

  return SUCCESS;
}

Status ParserGraphOptimizer::InsertNode(ge::ComputeGraphPtr sub_graph, vector<ge::NodePtr> &nodes,
                                        vector<ge::InDataAnchorPtr> &input_anchors,
                                        vector<ge::OutDataAnchorPtr> &output_anchors,
                                        map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                                        vector<ge::InControlAnchorPtr> &input_control_anchors,
                                        vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                        unordered_map<string, ge::NodePtr> &node_map) {
  GE_CHECK_NOTNULL(sub_graph);
  for (NodePtr node : nodes) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr op_def = node->GetOpDesc();
    NodePtr new_node = sub_graph->AddNode(op_def);
    GE_CHECK_NOTNULL(new_node);
    node_map[node->GetName()] = new_node;

    // Input
    for (auto in_anchor : node->GetAllInDataAnchors()) {  // data
      OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_out_anchor->GetOwnerNode());
      GE_IF_BOOL_EXEC(iter == nodes.end(), input_anchors.emplace_back(in_anchor));
    }
    // Output
    for (auto out_anchor : node->GetAllOutDataAnchors()) {
      bool hasOutNode = false;
      // data anchor
      for (auto peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_in_anchor->GetOwnerNode());
        GE_IF_BOOL_EXEC(iter == nodes.end(), output_in_map[out_anchor].emplace_back(peer_in_anchor); hasOutNode = true);
      }
      GE_IF_BOOL_EXEC(hasOutNode, output_anchors.emplace_back(out_anchor));
    }

    InControlAnchorPtr node_in_control = node->GetInControlAnchor();
    GE_IF_BOOL_EXEC(
        node_in_control != nullptr, for (auto peer_out_anchor
                                         : node_in_control->GetPeerOutControlAnchors()) {
          vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_out_anchor->GetOwnerNode());
          GE_IF_BOOL_EXEC(iter == nodes.end(), input_control_anchors.emplace_back(node_in_control));
        });
    OutControlAnchorPtr node_out_control = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(
        node_out_control != nullptr, for (auto peer_in_control_anchor
                                          : node_out_control->GetPeerInControlAnchors()) {
          vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_in_control_anchor->GetOwnerNode());
          GE_IF_BOOL_EXEC(iter == nodes.end(), output_control_anchors.emplace_back(node_out_control));
        });
  }
  return SUCCESS;
}

Status ParserGraphOptimizer::LinkInnerAnchor(unordered_map<string, ge::NodePtr> &node_map) const {
  for (auto node : graph_->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node_map.count(node->GetName()) == 0, continue);
    NodePtr dst = node_map[node->GetName()];
    for (auto in_anchor : node->GetAllInDataAnchors()) {
      OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      GE_IF_BOOL_EXEC(node_map.count(peer_out_anchor->GetOwnerNode()->GetName()) == 0, continue);
      NodePtr src = node_map[peer_out_anchor->GetOwnerNode()->GetName()];

      GE_IF_BOOL_EXEC(ge::GraphUtils::AddEdge(src->GetOutDataAnchor(peer_out_anchor->GetIdx()),
                                              dst->GetInDataAnchor(in_anchor->GetIdx())) != GRAPH_SUCCESS,
                      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                        src->GetName().c_str(), src->GetType().c_str(), peer_out_anchor->GetIdx(),
                                        dst->GetName().c_str(), dst->GetType().c_str(), in_anchor->GetIdx());
                      GELOGE(FAILED,
                             "LinkInnerAnchor Link data anchor failed, src node: %s, "
                             "dst node: %s.",
                             src->GetName().c_str(), dst->GetName().c_str());
                      return FAILED);
    }

    InControlAnchorPtr node_in_control = node->GetInControlAnchor();
    GE_IF_BOOL_EXEC(
        node_in_control != nullptr, for (auto peer_out_ctl_anchor
                                         : node_in_control->GetPeerOutControlAnchors()) {
          GE_IF_BOOL_EXEC(node_map.count(peer_out_ctl_anchor->GetOwnerNode()->GetName()) == 0, continue);
          NodePtr src_ctrl = node_map[peer_out_ctl_anchor->GetOwnerNode()->GetName()];
          GE_IF_BOOL_EXEC(
              ge::GraphUtils::AddEdge(src_ctrl->GetOutControlAnchor(), dst->GetInControlAnchor()) != GRAPH_SUCCESS,
              REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                                src_ctrl->GetName().c_str(), src_ctrl->GetType().c_str(),
                                dst->GetName().c_str(), dst->GetType().c_str());
              GELOGE(FAILED,
                     "LinkInnerAnchor Link control anchor failed, src node: "
                     "%s, dst node: %s.",
                     src_ctrl->GetName().c_str(), dst->GetName().c_str());
              return FAILED);

        });
  }
  return SUCCESS;
}

// rebuild output anchor
Status ParserGraphOptimizer::RebuildOutputAnchors(vector<ge::OutDataAnchorPtr> &output_anchors,
                                                  ge::OpDescPtr fusion_op_desc) {
  std::vector<int64_t> output_list;
  GE_CHECK_NOTNULL(fusion_op_desc);

  // create input desc
  for (auto out_anchor : output_anchors) {
    NodePtr src_node = out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);

    GeTensorDesc src_out_desc = src_node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
    GE_CHK_BOOL_EXEC(fusion_op_desc->AddOutputDesc(src_out_desc) == ge::GRAPH_SUCCESS, return FAILED);

    ge::DataType data_type = src_out_desc.GetDataType();
    const std::map<int32_t, int32_t>::const_iterator iter = GE_TENSORFLOW_DATA_TYPE_MAP.find((int32_t)data_type);
    GE_IF_BOOL_EXEC(
        iter == GE_TENSORFLOW_DATA_TYPE_MAP.end(),
        REPORT_INNER_ERROR("E19999", "datatype:%d of output:%d in node:%s:%s is not supported",
                           data_type, out_anchor->GetIdx(), src_node->GetName().c_str(), src_node->GetName().c_str());
        GELOGE(PARAM_INVALID, "data_type %s not supported", ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
        return PARAM_INVALID);

    int32_t dtype = iter->second;
    output_list.push_back((int64_t)dtype);
    GELOGI("FUNCDEF: output_list push_back  %d.", dtype);
  }
  GE_IF_BOOL_EXEC(!output_list.empty(), (void)AttrUtils::SetListInt(fusion_op_desc, ge::T_OUT_DATATYPE, output_list));

  return SUCCESS;
}
// rebuild input desc
Status ParserGraphOptimizer::RebuildInputAnchors(vector<ge::InDataAnchorPtr> &input_anchors,
                                                 ge::OpDescPtr fusion_op_desc) {
  std::vector<int64_t> input_list;
  GE_CHECK_NOTNULL(fusion_op_desc);
  // add input desc
  for (auto in_anchor : input_anchors) {
    NodePtr dst_node = in_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(dst_node);

    auto tensorDescPtr = dst_node->GetOpDesc()->GetInputDescPtr(in_anchor->GetIdx());
    GE_CHECK_NOTNULL_EXEC(tensorDescPtr, return domi::FAILED);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((fusion_op_desc->AddInputDesc(*tensorDescPtr)) != GRAPH_SUCCESS,
                                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                                     fusion_op_desc->GetName().c_str(),
                                                     fusion_op_desc->GetType().c_str());
                                   return FAILED,
                                   "Add fusion_op_desc AddInputDesc failed");
    ge::DataType data_type = tensorDescPtr->GetDataType();
    const std::map<int32_t, int32_t>::const_iterator iter = GE_TENSORFLOW_DATA_TYPE_MAP.find((int32_t)data_type);
    GE_IF_BOOL_EXEC(
        iter == GE_TENSORFLOW_DATA_TYPE_MAP.end(),
        REPORT_INNER_ERROR("E19999", "datatype:%d of input:%d in node:%s:%s is not supported",
                           data_type, in_anchor->GetIdx(), dst_node->GetName().c_str(), dst_node->GetName().c_str());
        GELOGE(PARAM_INVALID, "data_type %s not supported", ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
        return PARAM_INVALID);

    int32_t dtype = iter->second;
    input_list.push_back((int64_t)dtype);
    GELOGI("FUNCDEF: input_list push_back  %d.", dtype);
  }
  GE_IF_BOOL_EXEC(!input_list.empty(), (void)AttrUtils::SetListInt(fusion_op_desc, ge::T_IN_DATATYPE, input_list));

  return SUCCESS;
}

Status ParserGraphOptimizer::RebuildFusionNode(vector<ge::InDataAnchorPtr> &input_anchors,
                                               vector<ge::OutDataAnchorPtr> &output_anchors,
                                               map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                                               vector<ge::InControlAnchorPtr> &input_control_anchors,
                                               vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                               ge::NodePtr fusion_node) {
  GE_CHECK_NOTNULL(fusion_node);
  int32_t src_index = 0;

  for (auto out_anchor : output_anchors) {
    for (auto in_anchor : output_in_map[out_anchor]) {
      (void)in_anchor->Unlink(out_anchor);
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(fusion_node->GetOutDataAnchor(src_index), in_anchor),
                                  "Add anchor between fusion node and in anchor node!");
    }
    src_index++;
  }
  src_index = 0;
  for (auto in_anchor : input_anchors) {
    OutDataAnchorPtr out_anchor = in_anchor->GetPeerOutAnchor();
    out_anchor->Unlink(in_anchor);
    GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(out_anchor, fusion_node->GetInDataAnchor(src_index)),
                                "Add anchor between out anchor node and fusion node!");
    src_index++;
  }

  for (auto out_control_anchor : output_control_anchors) {
    for (auto in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      in_control_anchor->Unlink(out_control_anchor);
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(fusion_node->GetOutControlAnchor(), in_control_anchor),
                                  "Add anchor between fusion node and in control anchor node!");
    }
  }
  for (auto in_control_anchor : input_control_anchors) {
    for (auto out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      out_control_anchor->Unlink(in_control_anchor);
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(out_control_anchor, fusion_node->GetInControlAnchor()),
                                  "Add anchor between out control anchor node and fusion node!");
    }
  }
  return SUCCESS;
}
}  // namespace ge
