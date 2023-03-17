/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include <gtest/gtest.h>
#include <iostream>
#include "parser/common/op_parser_factory.h"
#include "graph/operator_reg.h"
#include "graph/utils/graph_utils_ex.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"
#include "external/parser/onnx_parser.h"
#include "st/parser_st_utils.h"
#include "external/ge/ge_api_types.h"
#include "tests/depends/ops_stub/ops_stub.h"
#include "framework/omg/parser/parser_factory.h"
#include "parser/onnx/onnx_util.h"
#define private public
#include "parser/onnx/onnx_parser.h"
#include "parser/onnx/onnx_file_constant_parser.h"
#undef private

namespace ge {
class STestOnnxParser : public testing::Test {
 protected:
  void SetUp() {
    ParerSTestsUtils::ClearParserInnerCtx();
    RegisterCustomOp();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
};

static Status ParseParams(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

static Status ParseParamByOpFunc(const ge::Operator &op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

Status ParseSubgraphPostFnIf(const std::string& subgraph_name, const ge::Graph& graph) {
  domi::AutoMappingSubgraphIOIndexFunc auto_mapping_subgraph_index_func =
      domi::FrameworkRegistry::Instance().GetAutoMappingSubgraphIOIndexFunc(domi::ONNX);
  if (auto_mapping_subgraph_index_func == nullptr) {
    std::cout<<"auto mapping if subgraph func is nullptr!"<<std::endl;
    return FAILED;
  }
  return auto_mapping_subgraph_index_func(graph,
                                          [&](int data_index, int &parent_index) -> Status {
                                            parent_index = data_index + 1;
                                            return SUCCESS;
                                          },
                                          [&](int output_index, int &parent_index) -> Status {
                                            parent_index = output_index;
                                            return SUCCESS;
                                          });
}

void STestOnnxParser::RegisterCustomOp() {
  REGISTER_CUSTOM_OP("Conv2D")
  .FrameworkType(domi::ONNX)
  .OriginOpType("ai.onnx::11::Conv")
  .ParseParamsFn(ParseParams);

  // register if op info to GE
  REGISTER_CUSTOM_OP("If")
  .FrameworkType(domi::ONNX)
  .OriginOpType({"ai.onnx::9::If",
                 "ai.onnx::10::If",
                 "ai.onnx::11::If",
                 "ai.onnx::12::If",
                 "ai.onnx::13::If"})
  .ParseParamsFn(ParseParams)
  .ParseParamsByOperatorFn(ParseParamByOpFunc)
  .ParseSubgraphPostFn(ParseSubgraphPostFnIf);

  REGISTER_CUSTOM_OP("Add")
  .FrameworkType(domi::ONNX)
      .OriginOpType("ai.onnx::11::Add")
      .ParseParamsFn(ParseParams);

  REGISTER_CUSTOM_OP("Identity")
  .FrameworkType(domi::ONNX)
      .OriginOpType("ai.onnx::11::Identity")
      .ParseParamsFn(ParseParams);

  std::vector<OpRegistrationData> reg_datas = domi::OpRegistry::Instance()->registrationDatas;
  for (auto reg_data : reg_datas) {
    domi::OpRegTbeParserFactory::Instance()->Finalize(reg_data);
    domi::OpRegistry::Instance()->Register(reg_data);
  }
  domi::OpRegistry::Instance()->registrationDatas.clear();
}

ge::onnx::GraphProto CreateOnnxGraph() {
  ge::onnx::GraphProto onnx_graph;
  (void)onnx_graph.add_input();
  (void)onnx_graph.add_output();
  ::ge::onnx::NodeProto* node_const1 = onnx_graph.add_node();
  ::ge::onnx::NodeProto* node_const2 = onnx_graph.add_node();
  ::ge::onnx::NodeProto* node_add = onnx_graph.add_node();
  node_const1->set_op_type(kOpTypeConstant);
  node_const2->set_op_type(kOpTypeConstant);
  node_add->set_op_type("Add");

  ::ge::onnx::AttributeProto* attr = node_const1->add_attribute();
  attr->set_name(ge::kAttrNameValue);
  ::ge::onnx::TensorProto* tensor_proto = attr->mutable_t();
  tensor_proto->set_data_location(ge::onnx::TensorProto_DataLocation_EXTERNAL);
  attr = node_const1->add_attribute();

  attr = node_const2->add_attribute();
  attr->set_name(ge::kAttrNameValue);
  tensor_proto = attr->mutable_t();
  tensor_proto->set_data_location(ge::onnx::TensorProto_DataLocation_DEFAULT);

  return onnx_graph;
}

TEST_F(STestOnnxParser, onnx_parser_user_output_with_default) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "Conv_0");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "Conv_0:0:y");
}

TEST_F(STestOnnxParser, onnx_parser_precheck) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  ge::GetParserContext().run_mode = ge::ONLY_PRE_CHECK;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(STestOnnxParser, onnx_parser_if_node) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_if.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  // has circle struct, topo sort failed
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestOnnxParser, onnx_parser_expand_one_to_many) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_clip_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  MemBuffer *buffer = ParerSTestsUtils::MemBufferFromFile(model_file.c_str());
  ret = ge::aclgrphParseONNXFromMem(reinterpret_cast<char *>(buffer->data), buffer->size, parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_parser_to_json) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_clip_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  OnnxModelParser onnx_parser;

  const char *json_file = "tmp.json";
  auto ret = onnx_parser.ToJson(model_file.c_str(), json_file);
  EXPECT_EQ(ret, SUCCESS);

  const char *json_null = nullptr;
  ret = onnx_parser.ToJson(model_file.c_str(), json_null);
  EXPECT_EQ(ret, FAILED);
  const char *model_null = nullptr;
  ret = onnx_parser.ToJson(model_null, json_null);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestOnnxParser, onnx_parser_const_data_type) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_const_type.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_parser_if_node_with_const_input) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_if_const_intput.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_test_ModelParseToGraph)
{
  OnnxModelParser modelParser;
  ge::onnx::ModelProto model_proto;
  auto onnx_graph = model_proto.mutable_graph();
  *onnx_graph = CreateOnnxGraph();
  ge::Graph graph;

  Status ret = modelParser.ModelParseToGraph(model_proto, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestOnnxParser, FileConstantParseParam)
{
  OnnxFileConstantParser parser;
  ge::onnx::NodeProto input_node;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);

  ge::onnx::TensorProto tensor_proto;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("value");
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  *attribute_tensor = tensor_proto;
  attribute_tensor->set_data_type(OnnxDataType::UINT16);
  attribute_tensor->add_dims(4);

  ge::onnx::StringStringEntryProto *string_proto1 = attribute_tensor->add_external_data();
  string_proto1->set_key("location");
  string_proto1->set_value("/tmp/weight");
  ge::onnx::StringStringEntryProto *string_proto2 = attribute_tensor->add_external_data();
  string_proto2->set_key("offset");
  string_proto2->set_value("4");
  ge::onnx::StringStringEntryProto *string_proto3 = attribute_tensor->add_external_data();
  string_proto3->set_key("length");
  string_proto3->set_value("16");
  Status ret = parser.ParseParams(reinterpret_cast<Message *>(&input_node), op);
  EXPECT_EQ(ret, SUCCESS);
}
} // namespace ge
