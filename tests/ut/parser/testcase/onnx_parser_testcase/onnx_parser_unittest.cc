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
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"
#include "external/parser/onnx_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "external/ge/ge_api_types.h"
#include "framework/omg/parser/parser_factory.h"
#include "tests/depends/ops_stub/ops_stub.h"

#define protected public
#define private public
#include "parser/onnx/onnx_constant_parser.h"
#include "parser/onnx/onnx_file_constant_parser.h"
#include "parser/onnx/onnx_util.h"
#include "parser/onnx/onnx_parser.h"
#undef protected
#undef private

namespace ge {
class UtestOnnxParser : public testing::Test {
 protected:
  void SetUp() {
    ParerUTestsUtils::ClearParserInnerCtx();
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

void UtestOnnxParser::RegisterCustomOp() {
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

TEST_F(UtestOnnxParser, onnx_parser_if_node) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/if.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOnnxParser, onnx_parser_user_output_with_name_and_index) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  parser_params.insert({AscendString(ge::ir_option::OUT_NODES), AscendString("Conv_0:0")});
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
  EXPECT_EQ(net_out_name.at(0), "Conv_0:0");
}

TEST_F(UtestOnnxParser, onnx_parser_user_output_with_tensor) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  parser_params.insert({AscendString(ge::ir_option::OUT_NODES), AscendString("y")});
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

TEST_F(UtestOnnxParser, onnx_parser_user_output_with_default) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/conv2d.onnx";
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

TEST_F(UtestOnnxParser, onnx_parser_user_output_with_tensor_failed) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  parser_params.insert({AscendString(ge::ir_option::OUT_NODES), AscendString("not_exist_output")});
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, onnx_parser_expand_one_to_many) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/onnx_clip_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  MemBuffer *buffer = ParerUTestsUtils::MemBufferFromFile(model_file.c_str());
  ret = ge::aclgrphParseONNXFromMem(reinterpret_cast<char *>(buffer->data), buffer->size, parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOnnxParser, onnx_parser_to_json) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/onnx_clip_v9.onnx";
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

  char *data = nullptr;
  uint32_t size = 0;
  ge::ComputeGraphPtr graph;
  ret = onnx_parser.ParseFromMemory(data, size, graph);
  EXPECT_EQ(ret, SUCCESS);

  google::protobuf::Message *proto = nullptr;
  ret = onnx_parser.ParseProto(proto, graph);
  EXPECT_EQ(ret, SUCCESS);

  domi::GetGraphCallback callback;
  ret = onnx_parser.ParseProtoWithSubgraph(proto, callback, graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = onnx_parser.ParseAllGraph(proto, graph);
  EXPECT_EQ(ret, SUCCESS);

  string file = "./";
  ret = onnx_parser.Save(file);
  EXPECT_NE(ret, SUCCESS);

  bool ret1 = onnx_parser.HasError();
  EXPECT_EQ(ret1, SUCCESS);
  onnx_parser.Clear();

  OnnxWeightsParser onnx_weight_parser;
  char *file1 = nullptr;
  ge::Graph graph1;
  ret =  onnx_weight_parser.Parse(file1, graph1);
  EXPECT_EQ(ret, SUCCESS);

  ret =  onnx_weight_parser.ParseFromMemory(data, size, graph);
  EXPECT_EQ(ret, SUCCESS);

  ret1 = onnx_weight_parser.HasError();
  EXPECT_EQ(ret1, SUCCESS);

  ret = onnx_weight_parser.Save(file);
  EXPECT_NE(ret, SUCCESS);
  onnx_weight_parser.Clear();
}

TEST_F(UtestOnnxParser, onnx_parser_const_data_type) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/onnx_model/onnx_const_type.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestOnnxParser, OnnxModelParser_ConvertToGeDataType_test)
{
  OnnxModelParser model_parser;
  uint32_t type = OnnxDataType::FLOAT;

  Status ret = model_parser.ConvertToGeDataType(type);
  EXPECT_EQ(ret, SUCCESS);

  type = 20;
  ret = model_parser.ConvertToGeDataType(type);
  EXPECT_EQ(ret, ge::DataType::DT_UNDEFINED);
}


TEST_F(UtestOnnxParser, OnnxModelParser_ParseConvertData_test)
{
  OnnxConstantParser constant_parser;
  ge::onnx::TensorProto tensor_proto;
  tensor_proto.set_data_type(ge::DataType::DT_UNDEFINED);

  ge::Tensor tensor;
  int count = 1;

  Status ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, FAILED);

  tensor_proto.set_data_type(OnnxDataType::INT32);
  count = 0;
  ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, SUCCESS);

  tensor_proto.set_data_type(OnnxDataType::BFLOAT16);
  count = 1;
  ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, FAILED);

  tensor_proto.set_data_type(OnnxDataType::STRING);
  ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, FAILED);

  tensor_proto.set_raw_data("Test");
  ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, SUCCESS);

  tensor_proto.set_data_type(OnnxDataType::FLOAT);
  ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOnnxParser, OnnxModelParser_ParseConvertData_test_bool)
{
  OnnxConstantParser constant_parser;
  ge::onnx::TensorProto tensor_proto;
  tensor_proto.set_data_type(OnnxDataType::INT32);
  ge::Tensor tensor ;
  TensorDesc tensor_desc = tensor.GetTensorDesc();
  tensor_desc.SetDataType(ge::DataType::DT_BOOL);
  tensor.SetTensorDesc(tensor_desc);
  int count = 1;
  tensor_proto.set_raw_data("Test");
  Status ret = constant_parser.ParseConvertData(tensor_proto, tensor, count);
  EXPECT_EQ(ret, SUCCESS);

}


TEST_F(UtestOnnxParser, OnnxConstantParser_ParseConvertTensor_test)
{
  OnnxConstantParser constant_parser;
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  attribute->set_f(1.0);
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);

  ge::Tensor tensor;
  Status ret = constant_parser.ParseConvertTensor(*attribute_tensor, tensor);
  EXPECT_EQ(ret, FAILED);

  attribute_tensor->add_dims(-1);
  ret = constant_parser.ParseConvertTensor(*attribute_tensor, tensor);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, OnnxConstantParser_ParseConvertDataType_test)
{
  OnnxConstantParser constant_parser;
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  attribute->set_f(1.0);
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(OnnxDataType::BFLOAT16);
  attribute_tensor->add_dims(4);

  ge::Tensor tensor;
  Status ret = constant_parser.ParseConvertDataType(*attribute_tensor, tensor);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, FileConstantGetTensorProto)
{
  OnnxFileConstantParser parser;
  ge::onnx::NodeProto input_node;
  ge::onnx::TensorProto tensor_proto;
  Status ret = parser.GetTensorProto(input_node, tensor_proto);
  EXPECT_EQ(ret, FAILED);

  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute = input_node.add_attribute();
  attribute->set_name("value");

  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  *attribute_tensor = tensor_proto;
  ret = parser.GetTensorProto(input_node, tensor_proto);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOnnxParser, FileConstantParseShape)
{
  OnnxFileConstantParser parser;
  ge::onnx::TensorProto tensor_proto;
  tensor_proto.add_dims(4);
  tensor_proto.add_dims(2);
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);

  parser.ParseShape(tensor_proto, op);

  std::vector<int64_t> attr_value;
  op.GetAttr("shape", attr_value);
  EXPECT_EQ(attr_value.size(), 2U);
  if (attr_value.size() == 2U) {
    EXPECT_EQ(attr_value[0], 4);
    EXPECT_EQ(attr_value[1], 2);
  }
}

TEST_F(UtestOnnxParser, FileConstantParseDataType)
{
  OnnxFileConstantParser parser;
  ge::onnx::TensorProto tensor_proto;
  tensor_proto.set_data_type(OnnxDataType::UNDEFINED);
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);

  Status ret = parser.ParseDataType(tensor_proto, op);
  EXPECT_EQ(ret, FAILED);

  tensor_proto.set_data_type(OnnxDataType::UINT8);
  ret = parser.ParseDataType(tensor_proto, op);
  EXPECT_EQ(ret, SUCCESS);
  ge::DataType attr_value;
  op.GetAttr("dtype", attr_value);
  EXPECT_EQ(attr_value, ge::DataType::DT_UINT8);
}

TEST_F(UtestOnnxParser, FileConstantParseAttr)
{
  OnnxFileConstantParser parser;
  ge::onnx::StringStringEntryProto string_proto;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);

  // test location
  string_proto.set_key("location");
  string_proto.set_value("/usr/local");
  Status ret = parser.SetPathAttr(string_proto, op);
  EXPECT_EQ(ret, SUCCESS);
  std::string attr_value;
  AttrUtils::GetStr(op_desc_src, "location", attr_value);
  EXPECT_EQ(attr_value, "/usr/local");

  // test offset
  string_proto.set_key("offset");
  string_proto.set_value("123");
  ret = parser.SetPathAttr(string_proto, op);
  EXPECT_EQ(ret, SUCCESS);
  int64_t offset_value;
  AttrUtils::GetInt(op_desc_src, "offset", offset_value);
  EXPECT_EQ(offset_value, 123 * 4096);

  // offset overflow
  string_proto.set_key("offset");
  string_proto.set_value("9223372036854775800");
  ret = parser.SetPathAttr(string_proto, op);
  EXPECT_EQ(ret, FAILED);

  // itol exception
  string_proto.set_key("offset");
  string_proto.set_value("999999999999999999999999999999999999");
  ret = parser.SetPathAttr(string_proto, op);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, FileConstantParsePath)
{
  OnnxFileConstantParser parser;
  ge::onnx::TensorProto tensor_proto;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);


  // without location, error
  auto ret = parser.ParsePath(tensor_proto, op);
  EXPECT_EQ(ret, FAILED);

  // SetPathAttr error
  ge::onnx::StringStringEntryProto *offset_proto = tensor_proto.add_external_data();
  offset_proto->set_key("offset");
  offset_proto->set_value("999999999999999999999999999999");
  ret = parser.ParsePath(tensor_proto, op);
  EXPECT_EQ(ret, FAILED);

  // has location, success
  ge::onnx::StringStringEntryProto *string_proto = tensor_proto.add_external_data();
  string_proto->set_key("location");
  string_proto->set_value("/usr/local");
  offset_proto->set_key("offset");
  offset_proto->set_value("0");
  ret = parser.ParsePath(tensor_proto, op);
  EXPECT_EQ(ret, SUCCESS);

  // check location
  std::string attr_value;
  AttrUtils::GetStr(op_desc_src, "location", attr_value);
  EXPECT_EQ(attr_value, "/usr/local");
}

TEST_F(UtestOnnxParser, FileConstantParseParam)
{
  OnnxFileConstantParser parser;
  ge::onnx::NodeProto input_node;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);

  // get tensor proto failed
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&input_node), op);
  EXPECT_EQ(ret, FAILED);

  ge::onnx::TensorProto tensor_proto;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("value");
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  *attribute_tensor = tensor_proto;

  // parse data type failed
  attribute_tensor->set_data_type(OnnxDataType::UNDEFINED);
  ret = parser.ParseParams(reinterpret_cast<Message *>(&input_node), op);
  EXPECT_EQ(ret, FAILED);

  // parse path failed
  attribute_tensor->set_data_type(OnnxDataType::UINT16);
  ret = parser.ParseParams(reinterpret_cast<Message *>(&input_node), op);
  EXPECT_EQ(ret, FAILED);

  // success
  ge::onnx::StringStringEntryProto *string_proto = attribute_tensor->add_external_data();
  string_proto->set_key("location");
  string_proto->set_value("/usr/local");
  attribute_tensor->add_dims(4);
  ret = parser.ParseParams(reinterpret_cast<Message *>(&input_node), op);
  EXPECT_EQ(ret, SUCCESS);

  // check location, shape, dtype
  std::string file_path;
  AttrUtils::GetStr(op_desc_src, "location", file_path);
  EXPECT_EQ(file_path, "/usr/local");

  std::vector<int64_t> dims;
  op.GetAttr("shape", dims);
  EXPECT_EQ(dims.size(), 1);
  if (!dims.empty()) {
    EXPECT_EQ(dims[0], 4);
  }
  DataType dtype;
  op.GetAttr("dtype", dtype);
  EXPECT_EQ(dtype, ge::DataType::DT_UINT16);
}

TEST_F(UtestOnnxParser, OnnxModelParser_ParseInput_test)
{
  OnnxModelParser model_parser;
  ge::onnx::ModelProto model_proto;
  ge::onnx::GraphProto graph = model_proto.graph();
  std::map<std::string, ge::onnx::TensorProto> initializer_name_tensor;
  bool is_subgraph = false;

  Status ret = model_parser.ParseInput(initializer_name_tensor, is_subgraph, graph);
  EXPECT_EQ(ret, domi::FAILED);

  ret = model_parser.ParseOutput(graph);
  EXPECT_EQ(ret, domi::FAILED);
}

TEST_F(UtestOnnxParser, OnnxModelParser_ParseConstant_test)
{
  OnnxModelParser model_parser;
  ge::onnx::GraphProto onnx_graph = CreateOnnxGraph();

  model_parser.UpdateNodeNameAndOpType(onnx_graph);
  std::string type = onnx_graph.mutable_node(0)->op_type();
  EXPECT_EQ(type, kFileConstant);
}

TEST_F(UtestOnnxParser, onnx_test_ConstructOriType)
{
  ge::onnx::ModelProto model_proto;
  ge::onnx::GraphProto* graph = model_proto.mutable_graph();
  ge::onnx::NodeProto* add_node = graph->add_node();
  add_node->set_op_type("Add");
  add_node->set_domain("ai.onnx");

  OnnxModelParser onnx_parser ;
  onnx_parser.domain_verseion_["ai.onnx"] = 11;
  string ori_type;
  Status ret = onnx_parser.ConstructOriType(add_node, ori_type);
  EXPECT_EQ(ret, domi::SUCCESS);

  ge::onnx::NodeProto* add_node1 = graph->add_node();
  add_node1->set_op_type("Add1");
  add_node1->set_domain("add.onnx");
  string op_type;
  ret = onnx_parser.AdapterOpType(add_node1, ori_type, op_type);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  add_node->set_op_type("Add1");
  ret = onnx_parser.AdapterOpType(add_node, ori_type, op_type);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestOnnxParser, onnx_test_TransNodeToOperator)
{
  ge::onnx::ModelProto model_proto;
  ge::onnx::GraphProto* graph = model_proto.mutable_graph();
  ge::onnx::NodeProto *node_proto = graph->add_node();
  node_proto->set_op_type("Add1");
  node_proto->set_domain("add.onnx");
  node_proto->set_name("Conv2D");
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Add", "add.onnx");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  std::string op_type = "Add";

  OnnxModelParser onnx_parser;
  Status ret = onnx_parser.TransNodeToOperator(node_proto, op, op_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOnnxParser, onnx_test_ModelParseToGraph)
{
  OnnxModelParser modelParser;
  ge::onnx::ModelProto model_proto;
  ge::onnx::OperatorSetIdProto* op_st = model_proto.add_opset_import();
  op_st->set_domain("ai.onnx");
  op_st->set_version(11);

  ge::Graph root_graph;

  Status ret = modelParser.ModelParseToGraph(model_proto, root_graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, onnx_test_SetExternalPath)
{
  OnnxModelParser modelParser;
  ge::onnx::ModelProto model_proto;
  auto ret = modelParser.SetExternalPath("", model_proto);
  EXPECT_NE(ret, SUCCESS);

  ge::onnx::GraphProto &graph_proto = const_cast<ge::onnx::GraphProto &>(model_proto.graph());
  graph_proto.add_initializer();
  ge::onnx::TensorProto* tensor_proto = graph_proto.add_initializer();
  tensor_proto->set_data_location(ge::onnx::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  tensor_proto->add_external_data();
  ge::onnx::StringStringEntryProto *string_proto = tensor_proto->add_external_data();
  string_proto->set_key("location");
  string_proto->set_value("if.onnx");
  ret = modelParser.SetExternalPath("/usr/local", model_proto);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOnnxParser, onnx_test_ParseFromMemory)
{
  OnnxModelParser modelParser;
  char *data = nullptr;
  uint32_t size = 1;
  ge::Graph graph;

  Status ret = modelParser.ParseFromMemory(data, size, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, onnx_test_Parse)
{
  OnnxModelParser modelParser;
  const char *file = nullptr;
  ge::Graph graph;

  Status ret = modelParser.Parse(file, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, onnx_test_GetModelFromMemory)
{
  OnnxModelParser modelParser;
  const char *data = "ut/parser/testcase/onnx_parser_testcase";
  uint32_t size = 1;
  ge::onnx::ModelProto model_proto;

  Status ret = modelParser.GetModelFromMemory(data, size, model_proto);
  EXPECT_EQ(ret, FAILED);

  ret = modelParser.GetModelFromFile(data, model_proto);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOnnxParser, onnx_test_TransNodeToOperator_SetTensorData)
{
  ge::onnx::ModelProto model_proto;
  ge::onnx::GraphProto* graph = model_proto.mutable_graph();
  ge::onnx::NodeProto *node_proto = graph->add_node();
  node_proto->set_op_type("Add1");
  node_proto->set_domain("add.onnx");
  node_proto->set_name("Conv2D");
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Add", "add.onnx");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  std::string op_type = "Add";

  OnnxModelParser onnx_parser;
  Status ret = onnx_parser.TransNodeToOperator(node_proto, op, op_type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOnnxParser, onnx_test_const_input_op)
{
  ge::onnx::ModelProto model_proto;
  ge::onnx::GraphProto* graph = model_proto.mutable_graph();
  ge::onnx::NodeProto *node_proto = graph->add_node();
  node_proto->set_op_type("Constant");
  node_proto->set_domain("const.onnx");
  node_proto->set_name("const_11");
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Constant", "const.onnx");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  std::string op_type = "Constant";

  OnnxModelParser onnx_parser;
  std::vector<ge::Operator> input_ops;
  onnx_parser.name_operator_["const_11"] = op;
  Status ret = onnx_parser.GetGraphInputs(*graph, input_ops);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(input_ops.size() > 0, true);
}
} // namespace ge
