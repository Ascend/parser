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

#include "onnx_parser.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include "common/convert/message2operator.h"
#include "common/convert/pb2json.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "external/graph/operator_factory.h"
#include "external/register/register_error_codes.h"
#include "external/parser/onnx_parser.h"
#include "external/ge/ge_api_types.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "framework/omg/parser/parser_types.h"
#include "omg/parser/parser_factory.h"
#include "onnx_op_parser.h"
#include "onnx_util.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/pre_checker.h"
#include "parser/common/acl_graph_parser_util.h"
#include "parser/common/model_saver.h"
#include "parser/common/parser_utils.h"
#include "parser/common/prototype_pass_manager.h"
#include "parser/onnx/onnx_custom_parser_adapter.h"
#include "parser/onnx/onnx_util.h"
#include "register/op_registry.h"
#include "register/register_fmk_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "subgraph_adapter/subgraph_adapter_factory.h"

namespace ge {
graphStatus PrepareBeforeParse(AclGrphParseUtil &acl_graph_parse_util,
                               const std::map<AscendString, AscendString> &parser_params,
                               ge::Graph &graph, std::shared_ptr<domi::ModelParser> &model_parser) {
  GetParserContext().type = domi::ONNX;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::ONNX)));

  if (acl_graph_parse_util.AclParserInitialize(options) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AclParserInitialize failed.");
    GELOGE(ge::FAILED, "[Init][AclParser] failed.");
    return ge::FAILED;
  }

  string output_name;
  if (acl_graph_parse_util.ParseParamsBeforeGraph(parser_params, output_name) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "ParseParamsBeforeGraph failed.");
    GELOGE(ge::FAILED, "[Parser][Params] before graph failed.");
    return ge::FAILED;
  }
  // Create an empty computegraph
  string graph_name = output_name.empty() ? "tmpGraph" : output_name;
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(compute_graph);

  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::ONNX);
  GE_CHECK_NOTNULL(model_parser);
  return ge::SUCCESS;
}

graphStatus HandleAfterParse(AclGrphParseUtil &acl_graph_parse_util,
                             const std::map<AscendString, AscendString> &parser_params,
                             ge::Graph &graph) {
  if (acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "ParseParamsAfterGraph failed.");
    GELOGE(ge::FAILED, "[Parser][Params] after graph failed.");
    return ge::FAILED;
  }

  if (acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Set graph default output node failed.");
    GELOGE(ge::FAILED, "[Update][NodeInfo] Set graph %s default output node failed.", graph.GetName().c_str());
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

graphStatus aclgrphParseONNX(const char *model_file,
                             const std::map<AscendString, AscendString> &parser_params, ge::Graph &graph) {
  GE_CHECK_NOTNULL(model_file);
  // load custom plugin so and proto
  AclGrphParseUtil acl_graph_parse_util;
  std::shared_ptr<domi::ModelParser> model_parser;

  if (PrepareBeforeParse(acl_graph_parse_util, parser_params, graph, model_parser) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Invoke][PrepareBeforeParse] failed.");
    return ge::FAILED;
  }

  GE_CHECK_NOTNULL(model_parser);
  // parse onnx model_file to GE graph
  ge::graphStatus ret = model_parser->Parse(model_file, graph);
  if (ret != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "parse modelfile %s failed, graph:%s", model_file, graph.GetName().c_str());
    GELOGE(ret, "[Parse][ModelFile] %s failed, graph %s.", model_file, graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("Parser graph %s success.", graph.GetName().c_str());

  if (HandleAfterParse(acl_graph_parse_util, parser_params, graph) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Invoke][HandleAfterParse] failed.");
    return ge::FAILED;
  }

  GELOGI("AclgrphParse graph %s success.", graph.GetName().c_str());
  return ge::SUCCESS;
}

graphStatus aclgrphParseONNXFromMem(const char *buffer, size_t size,
                                    const std::map<AscendString, AscendString> &parser_params, ge::Graph &graph) {
  GE_CHECK_NOTNULL(buffer);
  // load custom plugin so and proto
  AclGrphParseUtil acl_graph_parse_util;
  std::shared_ptr<domi::ModelParser> model_parser;

  if (PrepareBeforeParse(acl_graph_parse_util, parser_params, graph, model_parser)  != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Invoke][PrepareBeforeParse] failed.");
    return ge::FAILED;
  }

  // parse caffe model_file to GE graph
  ge::graphStatus ret = model_parser->ParseFromMemory(buffer, (uint32_t)size, graph);
  if (ret != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "ParseFromMemory failed");
    GELOGE(ret, "[Parser][Graph] %s failed.", graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("Parser graph %s success.", graph.GetName().c_str());

  if (HandleAfterParse(acl_graph_parse_util, parser_params, graph)  != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Invoke][HandleAfterParse] failed.");
    return ge::FAILED;
  }
    GELOGI("AclgrphParse graph %s success.", graph.GetName().c_str());
    return ge::SUCCESS;
}
} // namespace ge

namespace ge {
namespace {
const std::map<std::string, std::string> kOnnxOpMap = {
    {ge::kOpTypeInput, ge::parser::DATA},
    {ge::kOpTypeConstant, ge::parser::CONSTANT}
};
const int64_t kDimValue = 1;

struct ParseArg {
  ge::onnx::GraphProto *onnx_graph;
  ge::NodePtr parent_node;
  std::string graph_name;
  uint32_t subgraph_index;
};

Status GenSubgraphParseTasks(const ge::ComputeGraphPtr &parent_graph, std::deque<ParseArg> &args) {
  GELOGI("Gen subgraph parse tasks start");
  for (auto &node : parent_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto subgraph_name_to_index : op_desc->GetSubgraphNameIndexes()) {
      auto i = subgraph_name_to_index.second;
      auto subgraph_iname = subgraph_name_to_index.first;
      if (subgraph_iname.empty()) {
        GELOGW("The subgraph index %u of node %s is empty", i, node->GetName().c_str());
        continue;
      }

      // change the graph name to ensure it is unique in GE
      std::string unique_subgraph_name;
      OnnxUtil::GenUniqueSubgraphName(i, subgraph_iname, node->GetName(), unique_subgraph_name);

      GELOGD("Add subgraph parse task to the queue, node %s, index %u, subgraph instance name %s",
             node->GetName().c_str(), i, unique_subgraph_name.c_str());
      args.push_back({nullptr, node, unique_subgraph_name, i});
    }
  }
  GELOGI("Gen subgraph parse tasks end");
  return SUCCESS;
}

Status BuildLinkForChildAndParentGraph(const ge::ComputeGraphPtr &sub_graph, const ParseArg &arg) {
  if (arg.parent_node == nullptr) {
    return SUCCESS;
  }
  auto parent_node = arg.parent_node;
  auto index = arg.subgraph_index;
  auto ret = ge::NodeUtils::SetSubgraph(*parent_node, index, sub_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Subgraph] Failed to set subgraph %s to node %s index %u", sub_graph->GetName().c_str(),
           parent_node->GetName().c_str(), index);
    REPORT_CALL_ERROR("E19999", "Failed to set subgraph %s to node %s index %u", sub_graph->GetName().c_str(),
                      parent_node->GetName().c_str(), index);
    return ret;
  }
  return SUCCESS;
}

Status PostOpProcessForSubgraph(const ParseArg &arg, ge::ComputeGraphPtr sub_graph) {
  if (arg.parent_node == nullptr) {
    return SUCCESS;
  }
  std::string op_type = arg.parent_node->GetType();
  std::string op_name = arg.parent_node->GetName();
  domi::ParseSubgraphFuncV2 parse_func_v2 = nullptr;
  auto post_func =
      domi::OpRegistry::Instance()->GetParseSubgraphPostFunc(op_type);
  if (post_func == nullptr) {
    GELOGW("The subgraph post func for node %s type %s is null", op_name.c_str(), op_type.c_str());
    if (domi::OpRegistry::Instance()->GetParseSubgraphPostFunc(op_type, parse_func_v2) != SUCCESS || parse_func_v2 == nullptr) {
      GELOGW("The subgraph post func v2 for node %s type %s is null", op_name.c_str(), op_type.c_str());
      return SUCCESS;
    }
  }

  GELOGD("Post process for subgraph %s node %s type %s", arg.graph_name.c_str(), arg.parent_node->GetName().c_str(),
         arg.parent_node->GetType().c_str());

  // Refresh node_name in subgraph
  for (const ge::NodePtr &node : sub_graph->GetDirectNode()) {
    if (node->GetOpDesc() == nullptr) {
      continue;
    }
    node->GetOpDesc()->SetName(sub_graph->GetName() + "/" + node->GetName());
  }

  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(sub_graph);
  Status ret = FAILED;
  if (post_func != nullptr) {
    ret = post_func(arg.graph_name, graph);
  } else if (parse_func_v2 != nullptr) {
    ret = parse_func_v2(arg.graph_name.c_str(), graph);
  }
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[PostProcess][Subgraph]Failed to post-process subgraph %s on node %s type %s",
           arg.graph_name.c_str(), arg.parent_node->GetName().c_str(), arg.parent_node->GetType().c_str());
    REPORT_CALL_ERROR("E19999", "Failed to post-process subgraph %s on node %s type %s",
                      arg.graph_name.c_str(), arg.parent_node->GetName().c_str(), arg.parent_node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}
}

Status OnnxModelParser::ParseOutput(ge::onnx::GraphProto &onnx_graph) {
  if (onnx_graph.output_size() == 0) {
    GELOGE(FAILED, "[Parse][Output] Onnx graph:%s has zero output", onnx_graph.name().c_str());
    REPORT_INPUT_ERROR("E16001", std::vector<std::string>({"value"}), std::vector<std::string>({"output"}));
    return FAILED;
  }

  // get output value info map
  for (int i = 0; i < onnx_graph.output_size(); i++) {
    ge::onnx::ValueInfoProto value_info = onnx_graph.output(i);
    GELOGI("The index of %d output name : %s.", i, value_info.name().c_str());
    output_node_names_.emplace_back(value_info.name());
  }
  return SUCCESS;
}

Status OnnxModelParser::ParseInput(const std::map<std::string, ge::onnx::TensorProto> &initializer_name_tensor,
                                   bool is_subgraph, ge::onnx::GraphProto &onnx_graph) {
  if (!is_subgraph && onnx_graph.input_size() == 0) {
    GELOGE(FAILED, "[Parse][Input] Root onnx graph:%s has zero input", onnx_graph.name().c_str());
    REPORT_INPUT_ERROR("E16001", std::vector<std::string>({"value"}), std::vector<std::string>({"input"}));
    return FAILED;
  }

  // get input value info map
  int64_t data_index = 0;
  for (int i = 0; i < onnx_graph.input_size(); i++) {
    ge::onnx::ValueInfoProto value_info = onnx_graph.input(i);
    GELOGI("The index of %d input name : %s.", i, value_info.name().c_str());

    /// if the input is initialized by a default value found in ‘initializer’,
    /// it will be considered as a const node.
    auto initializer_iter = initializer_name_tensor.find(value_info.name());
    if (initializer_iter != initializer_name_tensor.end()) {
      continue;
    }

    ge::onnx::TensorProto tensor_tmp;
    if (value_info.has_type()) {
      const ge::onnx::TypeProto type = value_info.type();
      if (type.has_tensor_type()) {
        const ge::onnx::TypeProto_Tensor type_proto_tensor = type.tensor_type();
        int32_t elem_type = type_proto_tensor.elem_type();
        tensor_tmp.set_data_type(elem_type);
        if (type_proto_tensor.has_shape()) {
          const ge::onnx::TensorShapeProto tensor_shape = type_proto_tensor.shape();
          for (int j = 0; j < tensor_shape.dim_size(); j++) {
            const ge::onnx::TensorShapeProto_Dimension dimension = tensor_shape.dim(j);
            int64_t dim_value = -1;
            if (dimension.value_case() == kDimValue) {
              dim_value = dimension.dim_value();
            }
            tensor_tmp.add_dims(dim_value);
            GELOGI("elem_type: %d, dim_value: %ld", elem_type, dim_value);
          }
        }
      }
    }
    // Construct node for input
    ge::onnx::NodeProto *input_node = onnx_graph.add_node();
    input_node->set_name(value_info.name());
    input_node->set_op_type(ge::kOpTypeInput);
    input_node->add_output(value_info.name());
    // add tensor
    ge::onnx::AttributeProto *attribute = input_node->add_attribute();
    attribute->set_name(ge::kAttrNameInput);
    ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
    *attribute_tensor = tensor_tmp;
    // add index
    ge::onnx::AttributeProto *attribute_index = input_node->add_attribute();
    attribute_index->set_name(ge::kAttrNameIndex);
    attribute_index->set_i(data_index++);
    // add subgraph attr
    if (is_subgraph) {
      attribute = input_node->add_attribute();
      attribute->set_name(ge::kAttrNameIsSubgraphOp);
    }
    input_node_names_.emplace_back(value_info.name());
  }
  return SUCCESS;
}

Status OnnxModelParser::ParseInitializer(ge::onnx::GraphProto &onnx_graph,
                                         std::map<std::string, ge::onnx::TensorProto> &initializer_name_tensor) {
  // Construct const node for weight
  int index = 0;
  for (auto it : initializer_name_tensor) {
    ge::onnx::NodeProto *const_node = onnx_graph.add_node();
    std::string output_name = it.first + "_" + to_string(index++);
    const_node->set_name(output_name);
    const_node->set_op_type(ge::kOpTypeConstant);
    const_node->add_output(it.first);
    ge::onnx::AttributeProto *attribute = const_node->add_attribute();
    attribute->set_name(ge::kAttrNameValue);
    ge::onnx::TensorProto *attribute_t = attribute->mutable_t();
    *attribute_t = it.second;
  }

  return SUCCESS;
}

Status OnnxModelParser::UpdateAllNodeName(ge::onnx::GraphProto &onnx_graph) {
  int index = 0;
  for (int i = 0; i < onnx_graph.node_size(); i++) {
    ge::onnx::NodeProto *node = onnx_graph.mutable_node(i);
    if (node->name().empty()) {
      std::string node_name = node->op_type() + "_" + to_string(index++);
      node->set_name(node_name);
    }
  }

  return SUCCESS;
}

Status OnnxModelParser::ConstructOriType(const ge::onnx::NodeProto *node_proto, std::string &ori_type) {
  GE_CHECK_NOTNULL(node_proto);

  ori_type = node_proto->op_type();
  if (kOnnxOpMap.find(ori_type) != kOnnxOpMap.end()) {
    return SUCCESS;
  }

  std::string domain = node_proto->domain();
  int64_t version = 0;
  if (!domain.empty()) {
    auto it = domain_verseion_.find(domain);
    if (it != domain_verseion_.end()) {
      version = it->second;
    } else {
      REPORT_INNER_ERROR("E19999", "The opset of domain[%s] has no responding version.", domain.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param]The opset of domain[%s] has no responding version.", domain.c_str());
      return PARAM_INVALID;
    }
  } else {
    size_t domain_version_size = domain_verseion_.size();
    if (domain_version_size == 1) {
      domain = domain_verseion_.begin()->first;
      version = domain_verseion_.begin()->second;
    } else {
      GELOGE(PARAM_INVALID, "[Check][Param]The size of domain_version[%zu] should be equal to one.",
             domain_version_size);
      ErrorManager::GetInstance().ATCReportErrMessage("E16005", {"domain_version_size"},
                                                      {to_string(domain_version_size)});
      return PARAM_INVALID;
    }
  }

  if (domain.empty()) {
    domain = "ai.onnx";
  }

  ori_type = domain + "::" + to_string(version) + "::" + ori_type;
  return SUCCESS;
}

Status OnnxModelParser::AdapterOpType(const ge::onnx::NodeProto *node_proto, std::string &ori_type,
                                      std::string &op_type) {
  GE_CHECK_NOTNULL(node_proto);
  ori_type = node_proto->op_type();

  auto map_it = kOnnxOpMap.find(ori_type);
  if (map_it != kOnnxOpMap.end()) {
    op_type = map_it->second;
    ori_to_om_type_[ori_type] = op_type;
    return SUCCESS;
  }

  Status ret = ConstructOriType(node_proto, ori_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Construct][OriType] for [%s] failed.", ori_type.c_str());
    return ret;
  }

  if (!domi::OpRegistry::Instance()->GetOmTypeByOriOpType(ori_type, op_type)) {
    ErrorManager::GetInstance().ATCReportErrMessage("E16002", {"optype"}, {ori_type});
    GELOGE(PARAM_INVALID, "[Get][OmType] according ori_type : %s failed.", ori_type.c_str());
    return PARAM_INVALID;
  }

  ori_to_om_type_[ori_type] = op_type;
  return SUCCESS;
}

Status OnnxModelParser::TransNodeToOperator(const ge::onnx::NodeProto *node_proto, ge::Operator &op,
                                            const string &op_type) {
  GE_CHECK_NOTNULL(node_proto);
  string node_name = node_proto->name();
  op = ge::OperatorFactory::CreateOperator(node_name, op_type);
  if (op.GetName() != node_name) {
    REPORT_INPUT_ERROR("E10501", std::vector<std::string>({"opname", "optype"}),
                       std::vector<std::string>({node_name, op_type}));
    GELOGE(INTERNAL_ERROR, "[Creat][Op] IR for op[%s] optype[%s] is not registered.",
           node_name.c_str(), op_type.c_str());
    return INTERNAL_ERROR;
  }

  GELOGI("After create operator, op[%s]: type[%s] have input size: %zu, output size: %zu", op.GetName().c_str(),
         op.GetOpType().c_str(), op.GetInputsSize(), op.GetOutputsSize());
  return SUCCESS;
}

Status OnnxModelParser::ConstructInputOutputContext(const ge::onnx::NodeProto *node_proto) {
  GE_CHECK_NOTNULL(node_proto);

  std::string node_name = node_proto->name();
  for (int i = 0; i < node_proto->input_size(); i++) {
    std::string input_name = node_proto->input(i);
    inputs_map_[input_name].emplace_back(node_name, i);
  }

  for (int i = 0; i < node_proto->output_size(); i++) {
    std::string output_name = node_proto->output(i);
    outputs_map_[output_name].emplace_back(node_name, i);
  }

  return SUCCESS;
}

Status OnnxModelParser::SetOperatorInputs() {
  for (auto in_iter = inputs_map_.begin(); in_iter != inputs_map_.end(); in_iter++) {
    auto out_iter = outputs_map_.find(in_iter->first);
    if (out_iter == outputs_map_.end()) {
      GELOGW("Unknown input: %s:%d for node: %s, which maybe option input.",
             in_iter->first.c_str(),
             in_iter->second[0].second,
             in_iter->second[0].first.c_str());
      continue;
    }

    std::vector<std::pair<std::string, int>> &input_node_indexs = in_iter->second;
    std::vector<std::pair<std::string, int>> &output_node_indexs = out_iter->second;
    for (auto input_node_index : input_node_indexs) {
      for (auto out_node_index : output_node_indexs) {
        auto input_op_iter = name_operator_.find(input_node_index.first);
        if (input_op_iter == name_operator_.end()) {
          REPORT_INNER_ERROR("E19999", "Node: %s can not find in name_operator map.", input_node_index.first.c_str());
          GELOGE(INTERNAL_ERROR, "[Check][Param] Node: %s can not find in name_operator map.",
                 input_node_index.first.c_str());
          return INTERNAL_ERROR;
        }
        auto output_op_iter = name_operator_.find(out_node_index.first);
        if (output_op_iter == name_operator_.end()) {
          REPORT_INNER_ERROR("E19999", "Node: %s can not find in name_operator map.", out_node_index.first.c_str());
          GELOGE(INTERNAL_ERROR, "[Check][Param] Node: %s can not find in name_operator map.",
                 out_node_index.first.c_str());
          return INTERNAL_ERROR;
        }

        auto dst_op = input_op_iter->second;
        auto src_op = output_op_iter->second;
        int dst_index = input_node_index.second;
        int src_index = out_node_index.second;
        GELOGI("Start add output:%d of op:%s as input:%d of op:%s.", src_index, src_op.GetName().c_str(), dst_index,
               dst_op.GetName().c_str());
        auto dst_op_desc = ge::OpDescUtils::GetOpDescFromOperator(dst_op);
        GE_CHECK_NOTNULL(dst_op_desc);
        auto src_op_desc = ge::OpDescUtils::GetOpDescFromOperator(src_op);
        GE_CHECK_NOTNULL(src_op_desc);
        dst_op.SetInput(dst_op_desc->GetInputNameByIndex(dst_index), src_op,
                        src_op_desc->GetOutputNameByIndex(src_index));
      }
    }
  }
  return SUCCESS;
}

Status OnnxModelParser::Prechecker(ge::onnx::GraphProto &onnx_graph) {
  ge::PreChecker::Instance().Clear();
  for (int i = 0; i < onnx_graph.node_size(); i++) {
    ge::onnx::NodeProto *node = onnx_graph.mutable_node(i);
    std::string ori_type;
    Status ret = ConstructOriType(node, ori_type);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Construct][OriType] for [%s] failed.", ori_type.c_str());
      return ret;
    }
    GELOGI("Construct ori type : %s ", ori_type.c_str());
    if (ge::PreChecker::Instance().AddOp(node, node->name(), ori_type) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddOp failed, node:%s", node->name().c_str());
      GELOGE(FAILED, "[Add][NodeDef] to PreChecker failed, node name: %s.", node->name().c_str());
      return FAILED;
    }
    if (ge::PreChecker::Instance().CheckName(node) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "CheckName failed for node:%s", node->name().c_str());
      GELOGE(FAILED, "[Check][Name] failed, node name: %s.", node->name().c_str());
      return FAILED;
    }
    if (kOnnxOpMap.find(ori_type) == kOnnxOpMap.end()) {
      if (ge::PreChecker::Instance().CheckType(node) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "CheckType failed for node:%s", node->name().c_str());
        GELOGE(FAILED, "[Check][Type] failed, node name: %s.", node->name().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status OnnxModelParser::ParseOpParam(const ge::onnx::NodeProto *node_proto, ge::Operator &op,
                                     std::shared_ptr<OpParser> &op_parser) {
  GE_CHECK_NOTNULL(node_proto);
  GE_CHECK_NOTNULL(op_parser);
  std::string op_type = node_proto->op_type();

  Status status = FAILED;
  domi::ParseParamByOpFunc parse_param_func = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(op_type);
  if (parse_param_func == nullptr) {
    status = op_parser->ParseParams(node_proto, op);
  } else {
    ge::Operator op_src(node_proto->name(), op_type);
    status = Message2Operator::ParseOperatorAttrs(node_proto, 1, op_src);
    if (status != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Auto mapping node:%s(%s) to operator failed",
                        node_proto->name().c_str(), op_type.c_str());
      GELOGE(status, "Node[%s] auto mapping failed.", node_proto->name().c_str());
      return status;
    }
    std::shared_ptr<ge::OnnxCustomParserAdapter> onnx_custom_op_parser =
            std::dynamic_pointer_cast<ge::OnnxCustomParserAdapter>(op_parser);
    status = onnx_custom_op_parser->ParseParams(op_src, op);
    op_src.BreakConnect();
  }

  if (status != SUCCESS) {
    ErrorManager::GetInstance().ATCReportErrMessage("E11010", {"opname", "optype"}, {node_proto->name(), op_type});
    GELOGE(status, "[Parse][Params] for op [%s] fail, optype [%s]", node_proto->name().c_str(), op_type.c_str());
    return status;
  }

  return SUCCESS;
}

Status OnnxModelParser::ParseAllNodeProto(ge::onnx::GraphProto &onnx_graph, ge::Graph &graph) {
  for (int i = 0; i < onnx_graph.node_size(); i++) {
    ge::onnx::NodeProto *node_proto = onnx_graph.mutable_node(i);
    std::string node_name = node_proto->name();
    std::string ori_type = node_proto->op_type();
    GELOGI("Start parse node which name is %s, type is %s", node_name.c_str(), ori_type.c_str());

    std::string op_type;
    Status status = AdapterOpType(node_proto, ori_type, op_type);
    if (status != SUCCESS) {
      GELOGE(status, "[Adapt][OpType] Adapter op type for ori type %s failed.", ori_type.c_str());
      REPORT_CALL_ERROR("E19999", "Adapter op type for ori type %s failed.", ori_type.c_str());
      return status;
    }
    node_proto->set_op_type(ori_type);

    GELOGI("Trans original type:%s to op type:%s", ori_type.c_str(), op_type.c_str());

    ge::Operator op;
    status = TransNodeToOperator(node_proto, op, op_type);
    if (status != SUCCESS) {
      GELOGE(status, "[Trans][Node] Trans node to operator for %s:%s failed.", node_name.c_str(), op_type.c_str());
      REPORT_CALL_ERROR("E19999", "Trans node to operator for %s:%s failed.", node_name.c_str(), op_type.c_str());
      return status;
    }

    // 7. op parser
    std::shared_ptr<ge::OpParserFactory> factory = ge::OpParserFactory::Instance(domi::ONNX);
    GE_CHECK_NOTNULL(factory);
    std::shared_ptr<ge::OpParser> op_parser = factory->CreateOpParser(op_type);
    GE_CHECK_NOTNULL(op_parser);
    status = ParseOpParam(node_proto, op, op_parser);
    if (status != SUCCESS) {
      GELOGE(status, "[Parse][Params] for %s:%s failed ret:%d.", node_name.c_str(), op_type.c_str(), status);
      return status;
    }

    GELOGI("After ParseParams, op[%s]: type[%s] have input size: %zu, output size: %zu", op.GetName().c_str(),
           op.GetOpType().c_str(), op.GetInputsSize(), op.GetOutputsSize());

    ge::graphStatus graph_status = graph.AddOp(op);
    if (graph_status != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[Add][Op] Add op:%s to graph failed.", op.GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Add op:%s to graph failed.", op.GetName().c_str());
      return FAILED;
    }
    name_operator_[op.GetName()] = op;

    // 8. Construct input output relation of every node
    status = ConstructInputOutputContext(node_proto);
    if (status != SUCCESS) {
      REPORT_INNER_ERROR("E19999", "ConstructInputOutputContext failed.");
      GELOGE(status, "[Construct][RelationMap] to input and output failed.");
      return status;
    }
  }
  GELOGI("Parse all node proto success.");
  return SUCCESS;
}

Status OnnxModelParser::GetGraphInputs(ge::onnx::GraphProto &onnx_graph, std::vector<ge::Operator> &input_ops) {
  if (input_node_names_.empty()) {
    // subgraph might not have input, we use constant nodes as the start nodes of graph
    for (int i = 0; i < onnx_graph.node_size(); i++) {
      ge::onnx::NodeProto *node = onnx_graph.mutable_node(i);
      if (node->op_type() == kOpTypeConstant) {
        input_node_names_.emplace_back(node->name());
      }
    }
  }
  for (auto in_name : input_node_names_) {
    auto in_op = name_operator_.find(in_name);
    if (in_op == name_operator_.end()) {
      GELOGE(PARAM_INVALID, "[Get][Inputs] Model assigned input node name: %s can not find in graph.",
             in_name.c_str());
      REPORT_INNER_ERROR("E19999", "Model assigned input node name: %s can not find in graph.",
                         in_name.c_str());
      return PARAM_INVALID;
    }
    input_ops.emplace_back(in_op->second);
    GELOGI("Model assigned input node name: %s", in_op->second.GetName().c_str());
  }
    return SUCCESS;
}

Status OnnxModelParser::GetGraphOutputs(std::vector<std::pair<Operator, std::vector<size_t>>> &output_ops) {
  for (auto output_name : output_node_names_) {
    auto itr = outputs_map_.find(output_name);
    if (itr == outputs_map_.end()) {
      GELOGE(PARAM_INVALID, "[Get][Outputs] Can not find output:%s in graph.", output_name.c_str());
      REPORT_INNER_ERROR( "E19999", "[Get][Outputs] Can not find output:%s in graph.", output_name.c_str());
      return PARAM_INVALID;
    }

    std::vector<std::pair<std::string, int>> node_names_indexes = itr->second;
    for (const auto &node_name_index : node_names_indexes) {
      auto node_name = node_name_index.first;
      auto out_op_itr = name_operator_.find(node_name);
      if (out_op_itr == name_operator_.end()) {
        GELOGE(PARAM_INVALID, "[Get][Operator] Can not find operator: %s in graph.", node_name.c_str());
        REPORT_INNER_ERROR("E19999", "Can not find operator: %s in graph.", node_name.c_str());
        return PARAM_INVALID;
      }
      int index = node_name_index.second;
      output_ops.emplace_back(out_op_itr->second, vector<size_t>{static_cast<size_t>(index)});
      GELOGI("out node index %d, node:%s", index, node_name.c_str());
    }
  }
  return SUCCESS;
}

Status OnnxModelParser::GetModelFromFile(const char *file, ge::onnx::ModelProto &onnx_model) {
  GE_CHECK_NOTNULL(file);
  GELOGI("File path is %s.", file);

  // 1. Get graph from onnx model file.
  if (!ge::parser::ReadProtoFromBinaryFile(file, &onnx_model)) {
    REPORT_CALL_ERROR("E19999", "Read onnx model file:%s failed.", file);
    GELOGE(PARAM_INVALID, "[Read][ModeFile] failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OnnxModelParser::GetModelFromMemory(const char *data, uint32_t size, ge::onnx::ModelProto &onnx_model) {
  GE_CHECK_NOTNULL(data);

  // 1. Get graph from onnx model file.
  if (!ge::parser::ReadProtoFromArray(data, size, &onnx_model)) {
    REPORT_CALL_ERROR("E19999", "Read onnx model from memory failed.");
    GELOGE(PARAM_INVALID, "[Read][OnnxModel] from memory failed.");
    return FAILED;
  }
  return SUCCESS;
}

void OnnxModelParser::ClearMembers() {
  name_operator_.clear();
  input_node_names_.clear();
  output_node_names_.clear();
  inputs_map_.clear();
  outputs_map_.clear();
}

Status OnnxModelParser::AdaptAndFindAllOnnxGraph(ge::onnx::GraphProto &root_onnx_graph,
                                                 std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph) {
  std::queue<ge::onnx::GraphProto *> onnx_graph_tasks;
  int index = 0;
  onnx_graph_tasks.push(&root_onnx_graph);

  while (!onnx_graph_tasks.empty()) {
    ge::onnx::GraphProto *onnx_graph = onnx_graph_tasks.front();
    onnx_graph_tasks.pop();
    for (int i = 0; i < onnx_graph->node_size(); i++) {
      ge::onnx::NodeProto *node_proto = onnx_graph->mutable_node(i);
      if (node_proto->name().empty()) {
        std::string node_name = node_proto->op_type() + "_" + to_string(index++);
        node_proto->set_name(node_name);
      }
      GELOGD("adapt op name:%s, op type:%s", node_proto->name().c_str(), node_proto->op_type().c_str());

      SubgraphAdapterFactory *factory = SubgraphAdapterFactory::Instance();
      GE_CHECK_NOTNULL(factory);
      std::shared_ptr<SubgraphAdapter> subgraph_adapter = factory->CreateSubgraphAdapter(node_proto->op_type());
      if(subgraph_adapter == nullptr) {
        GELOGD("Do not need adapt subgraph, op type:%s", node_proto->op_type().c_str());
        continue;
      }
      std::vector<ge::onnx::GraphProto *> onnx_graphs;
      std::map<std::string, ge::onnx::GraphProto *> name_to_onnx_subgraph;
      if (subgraph_adapter->AdaptAndFindAllSubgraphs(node_proto, onnx_graphs, name_to_onnx_subgraph) != SUCCESS) {
        GELOGE(FAILED, "[Adapt][Subgraph] adapt subgraph of node:%s failed.", node_proto->name().c_str());
        REPORT_INNER_ERROR("E19999", "adapt subgraph of node:%s failed.", node_proto->name().c_str());
        return FAILED;
      }

      for (const auto &onnx_graph : onnx_graphs) {
        onnx_graph_tasks.push(onnx_graph);
      }
      for (const auto &itr : name_to_onnx_subgraph) {
        name_to_onnx_graph.emplace(itr.first, itr.second);
      }
    }
  }
  return SUCCESS;
}

Status OnnxModelParser::ModelParseToGraph(const ge::onnx::ModelProto &onnx_model, ge::Graph &root_graph) {
  if (!onnx_model.has_graph()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E16004");
    GELOGE(PARAM_INVALID, "Onnx model do not has graph.");
    return FAILED;
  }
  std::map<std::string, ge::onnx::GraphProto *> name_to_onnx_graph;
  std::deque<ParseArg> tasks;
  ge::onnx::GraphProto root_onnx_graph = onnx_model.graph();

  auto ret = AdaptAndFindAllOnnxGraph(root_onnx_graph, name_to_onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[AdaptAndFind][OnnxGraph]adapt and find all onnx graph failed, root graph:%s.",
           root_onnx_graph.name().c_str());
    return FAILED;
  }

  auto opset_import = onnx_model.opset_import();
  for (auto it : opset_import) {
    domain_verseion_[it.domain()] = it.version();
    GELOGI("Domain: %s, Version: %ld ", it.domain().c_str(), it.version());
  }
  std::string root_graph_name = root_graph.GetName().empty() ? "default_graph" : root_graph.GetName();
  tasks.push_back({&root_onnx_graph, nullptr, root_graph_name, 0});

  while (!tasks.empty()) {
    ParseArg arg = tasks.front();
    tasks.pop_front();
    bool is_subgraph = (arg.parent_node != nullptr) ? true : false;

    if (arg.onnx_graph == nullptr) {
      auto itr = name_to_onnx_graph.find(arg.graph_name);
      if (itr == name_to_onnx_graph.end()) {
        GELOGE(FAILED, "[Find][OnnxGraph] Can not find onnx graph, graph:%s.", arg.graph_name.c_str());
        REPORT_INNER_ERROR("E19999", "Can not find onnx graph, graph:%s.", arg.graph_name.c_str());
        return FAILED;
      }
      arg.onnx_graph = itr->second;
    }

    ge::onnx::GraphProto *onnx_graph = arg.onnx_graph;
    ge::Graph tmp_graph(arg.graph_name);
    ret = ModelParseToGraphImpl(is_subgraph, *onnx_graph, tmp_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Parse][Model] Model parse to graph failed, graph name:%s.", arg.graph_name.c_str());
      REPORT_INNER_ERROR("E19999", "Model parse to graph failed, graph name:%s.", arg.graph_name.c_str());
      return ret;
    }
    // To get the result for root graph
    if (!is_subgraph) {
      root_graph = tmp_graph;
    }

    ge::ComputeGraphPtr cur_compute_graph = ge::GraphUtils::GetComputeGraph(tmp_graph);
    GE_CHECK_NOTNULL(cur_compute_graph);

    ret = PostOpProcessForSubgraph(arg, cur_compute_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "[PostProcess][Subgraph]Post Op for subgraph:%s failed.", cur_compute_graph->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Post Op for subgraph:%s failed.", cur_compute_graph->GetName().c_str());
      return ret;
    }

    ret = BuildLinkForChildAndParentGraph(cur_compute_graph, arg);
    if (ret != SUCCESS) {
      GELOGE(ret, "[BuildLink][Graph] Build link for child graph:%s and parent graph failed.",
             cur_compute_graph->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Build link for child graph:%s and parent graph failed.",
                        cur_compute_graph->GetName().c_str());
      return ret;
    }

    ret = GenSubgraphParseTasks(cur_compute_graph, tasks);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Generate][Task] Failed to gen tasks on graph %s for next iteration",
             cur_compute_graph->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "Failed to gen tasks on graph %s for next iteration",
                        cur_compute_graph->GetName().c_str());
      return ret;
    }

  }
  UpdateDataFormat(root_graph);
  return SUCCESS;
}

Status OnnxModelParser::ModelParseToGraphImpl(bool is_subgraph, ge::onnx::GraphProto &onnx_graph, ge::Graph &graph) {

  ClearMembers();

  GE_RETURN_WITH_LOG_IF_ERROR(ProtoTypePassManager::Instance().Run(&onnx_graph, domi::ONNX),
                              "Run ProtoType Pass Failed");
  // 2. Get all inializer.
  std::map<std::string, ge::onnx::TensorProto> initializer_name_tensor;
  for (int i = 0; i < onnx_graph.initializer_size(); i++) {
    ge::onnx::TensorProto initializer_tensor = onnx_graph.initializer(i);
    if (!initializer_tensor.name().empty()) {
      initializer_name_tensor[initializer_tensor.name()] = initializer_tensor;
      GELOGI("Initializer name: %s .", initializer_tensor.name().c_str());
    }
  }

  // 3. Parse Input from graph.
  GELOGI("The size of initializer_name_tensor is %zu ", initializer_name_tensor.size());

  Status ret = ParseInput(initializer_name_tensor, is_subgraph, onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Input] for onnx failed.");
    return ret;
  }
  GELOGI("The size of initializer_name_tensor is %zu after ParseInput", initializer_name_tensor.size());

  // 4. Parse Constant from graph.
  ret = ParseInitializer(onnx_graph, initializer_name_tensor);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Initializer] for onnx failed.");
    return ret;
  }

  ret = ParseOutput(onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Output] Parse output for onnx failed.");
    return ret;
  }

  // 5. Update node name for node do not has name.
  ret = UpdateAllNodeName(onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][Name] of all node for onnx failed.");
    return ret;
  }

  // 6 Precheck.
  ret = Prechecker(onnx_graph);
  bool is_precheck_failed = (ret != SUCCESS) || (ge::PreChecker::Instance().HasError());
  if (is_precheck_failed) {
    GELOGE(FAILED, "[Invoke][Prechecker] failed.");
    return FAILED;
  }

  if (ge::GetParserContext().run_mode == ge::ONLY_PRE_CHECK) {
    GELOGI("Only prechecker.");
    return SUCCESS;
  }

  // 7. Construct all operator and input output tensor relation.
  ret = ParseAllNodeProto(onnx_graph, graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][AllNodeProto] failed.");
    return ret;
  }

  std::vector<string> op_names;
  graph.GetAllOpName(op_names);
  GELOGI("After trans node to operator, graph has the size of operator is %zu.", op_names.size());

  // 8. Set all operator input.
  ret = SetOperatorInputs();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][OperatorInputs] failed.");
    return ret;
  }

  // 9. Construct graph.
  std::vector<ge::Operator> input_ops;
  ret = GetGraphInputs(onnx_graph, input_ops);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][GraphInputs] failed.");
    return ret;
  }
  graph.SetInputs(input_ops);

  // root graph needn't set outputs.
  if(is_subgraph) {
    std::vector<std::pair<Operator, std::vector<size_t>>> output_ops;
    ret = GetGraphOutputs(output_ops);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][Outputs] failed.");
      return ret;
    }
    graph.SetOutputs(output_ops);
  }

  GE_RETURN_IF_ERROR(ParserUtils::ExpandOneToManyGraph(graph));

  GELOGI("Onnx model parser success.");
  return SUCCESS;
}

Status OnnxModelParser::Parse(const char *file, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  ge::onnx::ModelProto onnx_model;
  Status ret = GetModelFromFile(file, onnx_model);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][Model] From File:%s failed.", file);
    return FAILED;
  }
  ret = ModelParseToGraph(onnx_model, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Parse][Model] To Graph failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OnnxModelParser::ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  ge::onnx::ModelProto onnx_model;
  Status ret = GetModelFromMemory(data, size, onnx_model);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][Model] From Memory failed.");
    return FAILED;
  }
  ret = ModelParseToGraph(onnx_model, graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Parse][Model] To Graph failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OnnxModelParser::ToJson(const char *model_file, const char *json_file) {
  if (model_file == nullptr) {
    REPORT_INNER_ERROR("E19999", "param model file is nullprt, check invalid.");
    GELOGE(FAILED, "[Check][Param] Model file is nullptr.");
    return FAILED;
  }
  if (json_file == nullptr) {
    REPORT_INNER_ERROR("E19999", "param json file is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param]Json file is nullptr.");
    return FAILED;
  }

  ge::onnx::ModelProto onnx_model;
  GE_RETURN_WITH_LOG_IF_FALSE(ge::parser::ReadProtoFromBinaryFile(model_file, &onnx_model),
                              "[Invoke][ReadProtoFromBinaryFile] failed, file:%s.", model_file);
  ge::onnx::GraphProto graph_proto = onnx_model.graph();
  nlohmann::json j;
  ge::Pb2Json::Message2Json(graph_proto, std::set<std::string>(), j, true);
  return ge::parser::ModelSaver::SaveJsonToFile(json_file, j);
}

ge::DataType OnnxModelParser::ConvertToGeDataType(const uint32_t type) {
  return ge::OnnxUtil::ConvertOnnxDataType(type);
}

void OnnxModelParser::UpdateDataFormat(ge::Graph &graph) {
  for (GNode &gn : graph.GetDirectNode()) {
    AscendString type;
    (void)gn.GetType(type);
    if (type != parser::DATA) {
      continue;
    }
    TensorDesc in_desc;
    gn.GetInputDesc(0, in_desc);
    ge::Format ge_format = TypeUtils::DomiFormatToFormat(GetParserContext().format);
    in_desc.SetOriginFormat(ge_format);
    in_desc.SetFormat(ge_format);
    gn.UpdateInputDesc(0, in_desc);

    TensorDesc out_desc;
    gn.GetOutputDesc(0, out_desc);
    out_desc.SetOriginFormat(ge_format);
    out_desc.SetFormat(ge_format);
    gn.UpdateOutputDesc(0, out_desc);
  }
  GELOGD("Update data format success.");
  return;
}

}  // namespace domi

namespace domi {
  REGISTER_MODEL_PARSER_CREATOR(ONNX, ge::OnnxModelParser);
  REGISTER_WEIGHTS_PARSER_CREATOR(ONNX, ge::OnnxWeightsParser);
}
