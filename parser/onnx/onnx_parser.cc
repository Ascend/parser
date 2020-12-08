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
#include "common/convert/pb2json.h"
#include "common/model_saver.h"
#include "common/util.h"
#include "external/graph/operator_factory.h"
#include "external/register/register_error_codes.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "omg/parser/parser_factory.h"
#include "onnx_op_parser.h"
#include "onnx_util.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/pre_checker.h"
#include "parser/common/parser_utils.h"
#include "parser/onnx/onnx_util.h"
#include "register/op_registry.h"

namespace ge {
namespace {
std::map<std::string, std::string> kOnnxOpMap = {
    {ge::kOpTypeInput, ge::DATA}, {ge::kOpTypeConstant, ge::CONSTANT},
};
}

Status OnnxModelParser::ParseInput(ge::onnx::GraphProto &onnx_graph,
                                   std::map<std::string, ge::onnx::TensorProto> &initializer_name_tensor) {
  if (onnx_graph.input_size() == 0) {
    GELOGE(FAILED, "Onnx graph has zero input");
    return FAILED;
  }

  // get input value info map
  std::map<std::string, ge::onnx::TensorProto> input_name_tensor;
  for (int i = 0; i < onnx_graph.input_size(); i++) {
    ge::onnx::ValueInfoProto value_info = onnx_graph.input(i);
    GELOGI("The index of %d input name : %s.", i, value_info.name().c_str());

    // The input are possibly initialized by a default value found in ‘initializer.’
    auto initializer_iter = initializer_name_tensor.find(value_info.name());
    if (initializer_iter != initializer_name_tensor.end()) {
      input_name_tensor[value_info.name()] = initializer_iter->second;
      initializer_name_tensor.erase(initializer_iter);
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
            int64_t dim_value = dimension.dim_value();
            tensor_tmp.add_dims(dim_value);
            GELOGI("elem_type: %d, dim_value: %ld", elem_type, dim_value);
          }
        }
      }
    }
    input_name_tensor[value_info.name()] = tensor_tmp;
  }

  // Construct node for input
  int64_t index = 0;
  for (auto it : input_name_tensor) {
    ge::onnx::NodeProto *input_node = onnx_graph.add_node();
    input_node->set_name(it.first);
    input_node->set_op_type(ge::kOpTypeInput);
    input_node->add_output(it.first);
    // add tensor
    ge::onnx::AttributeProto *attribute = input_node->add_attribute();
    attribute->set_name(ge::kAttrNameInput);
    ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
    *attribute_tensor = it.second;
    // add index
    ge::onnx::AttributeProto *attribute_index = input_node->add_attribute();
    attribute_index->set_name(ge::kAttrNameIndex);
    attribute_index->set_i(index++);

    input_node_names_.emplace_back(it.first);
  }
  return SUCCESS;
}

Status OnnxModelParser::ParseOutput(const ge::onnx::GraphProto &onnx_graph) {
  if (onnx_graph.output_size() == 0) {
    GELOGE(FAILED, "Onnx graph has zero output");
    return FAILED;
  }

  for (int i = 0; i < onnx_graph.output_size(); i++) {
    ge::onnx::ValueInfoProto value_info = onnx_graph.output(i);
    GELOGI("The index of %d output name : %s.", i, value_info.name().c_str());

    auto it = outputs_map_.find(value_info.name());
    if (it != outputs_map_.end()) {
      std::string node_name = it->second[0].first;
      output_node_names_.emplace_back(node_name);
      GELOGI("Output node name: %s", node_name.c_str());
    }
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
      GELOGE(PARAM_INVALID, "The opset of domain[%s] has no responding version.", domain.c_str());
      return PARAM_INVALID;
    }
  } else {
    if (domain_verseion_.size() == 1){
      domain = domain_verseion_.begin()->first;
      version = domain_verseion_.begin()->second;
    } else {
      GELOGE(PARAM_INVALID, "The opset size %zu is bigger than one.", domain_verseion_.size());
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
    GELOGE(ret, "Construct ori type for [%s] failed.", ori_type.c_str());
    return ret;
  }

  if (!domi::OpRegistry::Instance()->GetOmTypeByOriOpType(ori_type, op_type)) {
    GELOGE(PARAM_INVALID, "Get omType according ori_type : %s failed.", ori_type.c_str());
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
    GELOGE(INTERNAL_ERROR, "IR for op[%s] optype[%s] is not registered.", node_name.c_str(), op_type.c_str());
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
      GELOGE(INTERNAL_ERROR, "Unknown input: %s:%d in node: %s", in_iter->first.c_str(), in_iter->second[0].second,
             in_iter->second[0].first.c_str());
      return INTERNAL_ERROR;
    }

    std::vector<std::pair<std::string, int>> &input_node_indexs = in_iter->second;
    std::vector<std::pair<std::string, int>> &output_node_indexs = out_iter->second;
    for (auto input_node_index : input_node_indexs) {
      for (auto out_node_index : output_node_indexs) {
        auto input_op_iter = name_operator_.find(input_node_index.first);
        if (input_op_iter == name_operator_.end()) {
          GELOGE(INTERNAL_ERROR, "Node: %s can not find in name_operator map.", input_node_index.first.c_str());
          return INTERNAL_ERROR;
        }
        auto output_op_iter = name_operator_.find(out_node_index.first);
        if (output_op_iter == name_operator_.end()) {
          GELOGE(INTERNAL_ERROR, "Node: %s can not find in name_operator map.", out_node_index.first.c_str());
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
      GELOGE(ret, "Construct ori type for [%s] failed.", ori_type.c_str());
      return ret;
    }
    GELOGI("Construct ori type : %s ", ori_type.c_str());
    if (ge::PreChecker::Instance().AddOp(node, node->name(), ori_type) != SUCCESS) {
      GELOGE(FAILED, "Add node_def to PreChecker failed, node name: %s.", node->name().c_str());
      return FAILED;
    }
    if (ge::PreChecker::Instance().CheckName(node) != SUCCESS) {
      GELOGE(FAILED, "Check node_def name failed, node name: %s.", node->name().c_str());
      return FAILED;
    }
    if (kOnnxOpMap.find(ori_type) == kOnnxOpMap.end()) {
      if (ge::PreChecker::Instance().CheckType(node) != SUCCESS) {
        GELOGE(FAILED, "Check node_def type failed, node name: %s.", node->name().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

void OnnxModelParser::UpdateFormat(ge::Graph &graph) {
  std::vector<string> vec_op_name;
  graph.GetAllOpName(vec_op_name);
  ge::Format format = ge::FORMAT_NCHW;
  for (string name: vec_op_name) {
    ge::Operator op;
    graph.FindOpByName(name, op);
    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_size = op_dsc->GetAllInputsSize();
    for (size_t i = 0; i < input_size; i++) {
      auto input = op_dsc->MutableInputDesc(static_cast<uint32_t>(i));
      if (input == nullptr) {
        continue;
      }
      input->SetFormat(format);
      input->SetOriginFormat(format);
    }

    auto output_size = op_dsc->GetOutputsSize();
    for (size_t i = 0; i < output_size; i++) {
      auto output = op_dsc->GetOutputDesc(i);
      output.SetFormat(format);
      output.SetOriginFormat(format);
      op_dsc->UpdateOutputDesc(i, output);
    }
  }
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
      GELOGE(status, "Adapter op type for ori type %s failed.", ori_type.c_str());
      return status;
    }
    node_proto->set_op_type(ori_type);

    GELOGI("Trans original type:%s to op type:%s", ori_type.c_str(), op_type.c_str());

    ge::Operator op;
    status = TransNodeToOperator(node_proto, op, op_type);
    if (status != SUCCESS) {
      GELOGE(status, "Trans node to operator for %s:%s failed.", node_name.c_str(), op_type.c_str());
      return status;
    }

    // 7. op parser
    std::shared_ptr<ge::OpParserFactory> factory = ge::OpParserFactory::Instance(domi::ONNX);
    GE_CHECK_NOTNULL(factory);
    std::shared_ptr<ge::OpParser> op_parser = factory->CreateOpParser(op_type);
    GE_CHECK_NOTNULL(op_parser);
    std::shared_ptr<ge::OnnxOpParser> onnx_op_parser = std::static_pointer_cast<ge::OnnxOpParser>(op_parser);
    GE_CHECK_NOTNULL(onnx_op_parser);
    status = onnx_op_parser->ParseParams(node_proto, op);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for %s:%s failed.", node_name.c_str(), op_type.c_str());
      return status;
    }

    ge::graphStatus graph_status = graph.AddOp(op);
    if (graph_status != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "Add op:%s to graph failed.", op.GetName().c_str());
      return FAILED;
    }
    name_operator_[op.GetName()] = op;

    // 8. Construct input output relation of every node
    status = ConstructInputOutputContext(node_proto);
    if (status != SUCCESS) {
      GELOGE(status, "Construct input output relation map failed.");
      return status;
    }
  }
  GELOGI("Parse all node proto success.");
  return SUCCESS;
}

Status OnnxModelParser::GetGraphInputsOutputs(std::vector<ge::Operator> &input_ops,
                                              std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs) {
  for (auto in_name : input_node_names_) {
    auto in_op = name_operator_.find(in_name);
    if (in_op == name_operator_.end()) {
      GELOGE(PARAM_INVALID, "Model assigned output node name: %s can not find in graph.",
             in_name.c_str());
      return PARAM_INVALID;
    }
    input_ops.emplace_back(in_op->second);
    GELOGI("Model assigned input node name: %s", in_op->second.GetName().c_str());
  }

  for (auto it : output_node_names_) {
    auto out_op = name_operator_.find(it);
    if (out_op == name_operator_.end()) {
      GELOGE(PARAM_INVALID, "Model assigned output node name: %s can not find in graph.",
             it.c_str());
      return PARAM_INVALID;
    }
    output_indexs.emplace_back(out_op->second, std::vector<size_t>{});
    GELOGI("Model assigned output node name: %s", out_op->second.GetName().c_str());
  }
  return SUCCESS;
}

Status OnnxModelParser::Parse(const char *file, ge::Graph &graph) {
  GE_CHECK_NOTNULL(file);
  GELOGI("File path is %s.", file);

  // 1. Get graph from onnx model file.
  ge::onnx::ModelProto onnx_model;
  if (!ge::ReadProtoFromBinaryFile(file, &onnx_model)) {
    GELOGE(PARAM_INVALID, "Read onnx model file failed.");
    return FAILED;
  }
  if (!onnx_model.has_graph()) {
    GELOGE(PARAM_INVALID, "Onnx model do not has graph.");
    return FAILED;
  }
  ge::onnx::GraphProto onnx_graph = onnx_model.graph();

  auto opset_import = onnx_model.opset_import();
  for (auto it : opset_import) {
    domain_verseion_[it.domain()] = it.version();
    GELOGI("Domain: %s, Version: %ld ", it.domain().c_str(), it.version());
  }

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
  Status ret = ParseInput(onnx_graph, initializer_name_tensor);
  if (ret != SUCCESS) {
    GELOGE(ret, "Parse input for onnx failed.");
    return ret;
  }
  GELOGI("The size of initializer_name_tensor is %zu after ParseInput", initializer_name_tensor.size());

  // 4. Parse Constant from graph.
  ret = ParseInitializer(onnx_graph, initializer_name_tensor);
  if (ret != SUCCESS) {
    GELOGE(ret, "Parse initializer for onnx failed.");
    return ret;
  }

  // 5. Update node name for node do not has name.
  ret = UpdateAllNodeName(onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Update all node name for onnx failed.");
    return ret;
  }

  // 6 Precheck.
  ret = Prechecker(onnx_graph);
  bool is_precheck_failed = (ret != SUCCESS) || (ge::PreChecker::Instance().HasError());
  if (is_precheck_failed) {
    GELOGE(FAILED, "Prechecker failed.");
    return FAILED;
  }

  if (ge::GetParserContext().run_mode == ge::ONLY_PRE_CHECK) {
    GELOGI("Only prechecker.");
    return SUCCESS;
  }

  // 7. Construct all operator and input output tensor relation.
  ret = ParseAllNodeProto(onnx_graph, graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Parse all node proto failed.");
    return ret;
  }

  // 8. Parse output from graph.
  ret = ParseOutput(onnx_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Parse output failed.");
    return ret;
  }

  // 9. Set all operator input.
  ret = SetOperatorInputs();
  if (ret != SUCCESS) {
    GELOGE(ret, "Set operator input failed.");
    return ret;
  }

  std::vector<string> op_names;
  graph.GetAllOpName(op_names);
  GELOGI("After trans node to operator, graph has the size of operator is %zu.", op_names.size());

  // 10. Construct graph.
  std::vector<ge::Operator> input_ops;
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  ret = GetGraphInputsOutputs(input_ops, output_indexs);
  if (ret != SUCCESS) {
    GELOGE(ret, "Get graph inputs and outputs failed.");
    return ret;
  }
  graph.SetInputs(input_ops).SetOutputs(output_indexs);

  GE_RETURN_IF_ERROR(ParserUtils::ExpandOneToManyGraph(graph));

  UpdateFormat(graph);

  GELOGI("Onnx model parser success.");
  return SUCCESS;
}

Status OnnxModelParser::ToJson(const char *model_file, const char *json_file) {
  if (model_file == nullptr) {
    GELOGE(FAILED, "Model file is nullptr.");
    return FAILED;
  }
  if (json_file == nullptr) {
    GELOGE(FAILED, "Json file is nullptr.");
    return FAILED;
  }

  ge::onnx::ModelProto onnx_model;
  GE_RETURN_WITH_LOG_IF_FALSE(ge::ReadProtoFromBinaryFile(model_file, &onnx_model),
                              "ReadProtoFromBinaryFile failed, file:%s.", model_file);
  ge::onnx::GraphProto graph_proto = onnx_model.graph();
  nlohmann::json j;
  ge::Pb2Json::Message2Json(graph_proto, std::set<std::string>(), j, true);
  return ge::ModelSaver::SaveJsonToFile(json_file, j);
}

ge::DataType OnnxModelParser::ConvertToGeDataType(const uint32_t type) {
  return ge::OnnxUtil::ConvertOnnxDataType(type);
}

}  // namespace domi

namespace domi {
  REGISTER_MODEL_PARSER_CREATOR(ONNX, ge::OnnxModelParser);
  REGISTER_WEIGHTS_PARSER_CREATOR(ONNX, ge::OnnxWeightsParser);
}