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

#ifndef PARSER_ONNX_ONNX_PARSER_H_
#define PARSER_ONNX_ONNX_PARSER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <string>
#include <vector>
#include "external/register/register_error_codes.h"
#include "omg/parser/model_parser.h"
#include "omg/parser/op_parser.h"
#include "omg/parser/weights_parser.h"
#include "proto/onnx/ge_onnx.pb.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxModelParser : public domi::ModelParser {
 public:
  OnnxModelParser() {}
  virtual ~OnnxModelParser() {}

  Status Parse(const char *file, ge::Graph &graph) override;

  Status ToJson(const char *model_file, const char *json_file) override;

  ge::DataType ConvertToGeDataType(const uint32_t type) override;

  Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) override { return domi::SUCCESS; }

  Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph) override;

  Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) override {
    return domi::SUCCESS;
  }

  Status ParseProtoWithSubgraph(const google::protobuf::Message *root_proto, domi::GetGraphCallback callback,
                                ge::ComputeGraphPtr &graph) override {
    return domi::SUCCESS;
  }

  Status ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) override {
    return domi::SUCCESS;
  }

 private:
  Status ParseAllNodeProto(ge::onnx::GraphProto &onnx_graph, ge::Graph &graph);

  Status ParseInput(const std::map<std::string, ge::onnx::TensorProto> &initializer_name_tensor,
                    bool is_subgraph, ge::onnx::GraphProto &onnx_graph);

  Status ParseOutput(ge::onnx::GraphProto &onnx_graph);

  Status ParseInitializer(ge::onnx::GraphProto &onnx_graph,
                          std::map<std::string, ge::onnx::TensorProto> &initializer_name_tensor);

  Status UpdateAllNodeName(ge::onnx::GraphProto &onnx_graph);

  Status ConstructOriType(const ge::onnx::NodeProto *node_proto, std::string &ori_type);

  Status AdapterOpType(const ge::onnx::NodeProto *node_proto, std::string &ori_type, std::string &om_type);

  Status TransNodeToOperator(const ge::onnx::NodeProto *node_proto, ge::Operator &op, const string &op_type);

  Status ConstructInputOutputContext(const ge::onnx::NodeProto *node_proto);

  Status SetOperatorInputs();

  Status GetGraphInputs(ge::onnx::GraphProto &onnx_graph, std::vector<ge::Operator> &input_ops);

  Status GetGraphOutputs(std::vector<std::pair<Operator, std::vector<size_t>>> &outputs);

  Status Prechecker(ge::onnx::GraphProto &onnx_graph);
  
  Status GetModelFromFile(const char *file, ge::onnx::ModelProto &onnx_model);

  Status GetModelFromMemory(const char *data, uint32_t size, ge::onnx::ModelProto &onnx_model);

  Status ModelParseToGraph(const ge::onnx::ModelProto &onnx_model, ge::Graph &graph);

  Status ModelParseToGraphImpl(bool is_subgraph, ge::onnx::GraphProto &onnx_graph, ge::Graph &graph);

  void UpdateDataFormat(ge::Graph &graph);

  void ClearMembers();

  Status ParseOpParam(const ge::onnx::NodeProto *node_proto, ge::Operator &op, std::shared_ptr<OpParser> &op_parser);

  Status AdaptAndFindAllOnnxGraph(ge::onnx::GraphProto &root_onnx_graph,
                                  std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph);

  std::map<std::string, std::string> ori_to_om_type_;

  std::map<std::string, int64_t> domain_verseion_;

  std::map<std::string, ge::Operator> name_operator_;

  std::vector<std::string> input_node_names_;

  std::vector<std::string> output_node_names_;

  std::map<std::string, std::vector<std::pair<std::string, int>>> inputs_map_;

  std::map<std::string, std::vector<std::pair<std::string, int>>> outputs_map_;
};

class PARSER_FUNC_VISIBILITY OnnxWeightsParser : public domi::WeightsParser {
 public:
  Status Parse(const char *file, ge::Graph &graph) override { return domi::SUCCESS; }

  Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) override { return domi::SUCCESS; }
};
}  // namespace domi
#endif  // PARSER_ONNX_ONNX_PARSER_H_
