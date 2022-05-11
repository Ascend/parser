/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define protected public
#define private public
#include <iostream>
#include "parser/common/op_parser_factory.h"
#include "graph/operator_reg.h"
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "external/parser/caffe_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "external/ge/ge_api_types.h"
#include "tests/depends/ops_stub/ops_stub.h"
#include "proto/caffe/caffe.pb.h"
#include "parser/caffe/caffe_parser.h"
#include "parser/caffe/caffe_data_parser.h"
#include "parser/caffe/caffe_op_parser.h"
#include "parser/caffe/caffe_custom_parser_adapter.h"
#include "parser/caffe/caffe_op_parser.h"
#include "graph/operator_reg.h"
#include "parser/common/acl_graph_parser_util.h"
#include "parser/caffe/caffe_reshape_parser.h"
#include "common/op_map.h"
#undef protected
#undef private

#include <google/protobuf/compiler/importer.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/dynamic_message.h>

using namespace domi::caffe;
using namespace ge;

namespace ge {
class UtestCaffeParser : public testing::Test {
 protected:
  void SetUp() {
    ParerUTestsUtils::ClearParserInnerCtx();
    RegisterCustomOp();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
};

static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr opDesc){
  if (!opDesc) {
    return nullptr;
  }
  static auto g = std::make_shared<ge::ComputeGraph>("g");
  return g->AddNode(std::move(opDesc));
}

  ge::ComputeGraphPtr build_graph(bool with_leaf_node = false)
  {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
    ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
    data_op->SetType(parser::DATA);
    data_op->SetName("Data1");
    data_op->AddInputDesc(ge::GeTensorDesc());
    data_op->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr data1 = graph->AddNode(data_op);

    ge::OpDescPtr relu_op1 = std::make_shared<ge::OpDesc>();
    relu_op1->SetType(parser::ACTIVATION);
    relu_op1->SetName("Relu1");
    relu_op1->AddInputDesc(ge::GeTensorDesc());
    relu_op1->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr relu1 = graph->AddNode(relu_op1);

    ge::OpDescPtr relu_op2 = std::make_shared<ge::OpDesc>();
    relu_op2->SetType(parser::RELU);
    relu_op2->SetName("Relu2");
    relu_op2->AddInputDesc(ge::GeTensorDesc());
    relu_op2->AddOutputDesc(ge::GeTensorDesc());
    relu_op2->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr relu2 = graph->AddNode(relu_op2);

    ge::OpDescPtr relu_op3 = std::make_shared<ge::OpDesc>();
    relu_op3->SetType(parser::ACTIVATION);
    relu_op3->SetName("Relu3");
    relu_op3->AddInputDesc(ge::GeTensorDesc());
    relu_op3->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr relu3;
    if (with_leaf_node == true) {
        relu3 = graph->AddNode(relu_op3);
    }

    ge::OpDescPtr mul_op = std::make_shared<ge::OpDesc>();
    mul_op->SetType(parser::MUL);
    mul_op->SetName("Mul");
    mul_op->AddInputDesc(ge::GeTensorDesc());
    mul_op->AddInputDesc(ge::GeTensorDesc());
    mul_op->AddOutputDesc(ge::GeTensorDesc());
    mul_op->AddOutputDesc(ge::GeTensorDesc());
    mul_op->AddOutputDesc(ge::GeTensorDesc());
    mul_op->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr mul = graph->AddNode(mul_op);

    ge::OpDescPtr mul_op1 = std::make_shared<ge::OpDesc>();
    mul_op1->SetType(parser::MUL);
    mul_op1->SetName("Mul1");
    mul_op1->AddInputDesc(ge::GeTensorDesc());
    mul_op1->AddInputDesc(ge::GeTensorDesc());
    mul_op1->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr mul1 = graph->AddNode(mul_op1);

    ge::OpDescPtr mul_op2 = std::make_shared<ge::OpDesc>();
    mul_op2->SetType(parser::MUL);
    mul_op2->SetName("Mul2");
    mul_op2->AddInputDesc(ge::GeTensorDesc());
    mul_op2->AddInputDesc(ge::GeTensorDesc());
    mul_op2->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr mul2 = graph->AddNode(mul_op2);

    ge::OpDescPtr fc_op = std::make_shared<ge::OpDesc>();
    fc_op->SetType(parser::FULL_CONNECTION);
    fc_op->SetName("FullConnection");
    fc_op->AddInputDesc(ge::GeTensorDesc());
    fc_op->AddOutputDesc(ge::GeTensorDesc());
    fc_op->AddOutputDesc(ge::GeTensorDesc());
    ge::NodePtr fc = graph->AddNode(fc_op);

    ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), relu1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), fc->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(fc->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));
    if (with_leaf_node == true) {
        ge::GraphUtils::AddEdge(fc->GetOutDataAnchor(1), relu3->GetInDataAnchor(0));
    }
    ge::GraphUtils::AddEdge(relu2->GetOutDataAnchor(0), mul->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2->GetOutDataAnchor(1), mul->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(0), mul1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(1), mul1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(2), mul2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(mul->GetOutDataAnchor(3), mul2->GetInDataAnchor(1));

    return graph;
  }

void UtestCaffeParser::RegisterCustomOp() {
  std::vector<OpRegistrationData> reg_datas = domi::OpRegistry::Instance()->registrationDatas;
  for (auto reg_data : reg_datas) {
    domi::OpRegTbeParserFactory::Instance()->Finalize(reg_data);
    domi::OpRegistry::Instance()->Register(reg_data);
  }
  domi::OpRegistry::Instance()->registrationDatas.clear();
}

TEST_F(UtestCaffeParser, caffe_parser_user_output_with_name_and_index) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe_abs.pbtxt";
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::CAFFE);
  ASSERT_NE(model_parser, nullptr);
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  ASSERT_NE(compute_graph, nullptr);
  ge::Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  ge::GetParserContext().user_out_nodes.push_back({"abs", 0});
  auto ret = model_parser->Parse(model_file.c_str(), graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  AclGrphParseUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> parser_params;
  auto status = acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params);
  ASSERT_EQ(status, SUCCESS);

  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "abs");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "abs:0:abs_out");
}

TEST_F(UtestCaffeParser, caffe_parser_user_output_with_top_name) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe_abs.pbtxt";
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::CAFFE);
  ASSERT_NE(model_parser, nullptr);
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  ASSERT_NE(compute_graph, nullptr);
  ge::Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  ge::GetParserContext().user_out_tensors.push_back("abs_out");
  auto ret = model_parser->Parse(model_file.c_str(), graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  AclGrphParseUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> parser_params;
  auto status = acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params);
  ASSERT_EQ(status, SUCCESS);

  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "abs");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "abs:0:abs_out");
}

TEST_F(UtestCaffeParser, caffe_parser_user_output_with_default) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe_abs.pbtxt";
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::CAFFE);
  ASSERT_NE(model_parser, nullptr);
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  ASSERT_NE(compute_graph, nullptr);
  ge::Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto ret = model_parser->Parse(model_file.c_str(), graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  AclGrphParseUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> parser_params;
  auto status = acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params);
  ASSERT_EQ(status, SUCCESS);

  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "abs");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "abs:0:abs_out");
}

TEST_F(UtestCaffeParser, acl_caffe_parser) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe_add.pbtxt";
  std::string weight_file_txt = case_dir + "/caffe_model/caffe_add.caffemodel.txt";
  std::string weight_file = case_dir + "/caffe_model/caffe_add.caffemodel";

  domi::caffe::NetParameter proto;
  EXPECT_EQ(ParerUTestsUtils::ReadProtoFromText(weight_file_txt.c_str(), &proto), true);
  ParerUTestsUtils::WriteProtoToBinaryFile(proto, weight_file.c_str());

  ge::GetParserContext().caffe_proto_path = case_dir + "/../../../../metadef/proto/caffe/caffe.proto";

  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseCaffe(model_file.c_str(), weight_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = ge::aclgrphParseCaffe(model_file.c_str(), weight_file.c_str(), graph);
  EXPECT_EQ(ret, GRAPH_FAILED);

  caffe_op_map.clear();
  ret = ge::aclgrphParseCaffe(model_file.c_str(), weight_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_FAILED);

  {
    proto.set_name("empty_layer");
    auto &layers = *proto.add_layers();
    layers.set_name("layers");

    proto.clear_layer();
    const std::string empty_layer = case_dir + "/caffe_model/empty_layer.pbtxt";
    ParerUTestsUtils::WriteProtoToTextFile(proto, empty_layer.c_str());
    EXPECT_EQ(ge::aclgrphParseCaffe(empty_layer.c_str(), weight_file.c_str(), parser_params, graph), FAILED);

    proto.clear_layers();
    const std::string empty_layers = case_dir + "/caffe_model/empty_layers.pbtxt";
    ParerUTestsUtils::WriteProtoToTextFile(proto, empty_layers.c_str());
    EXPECT_EQ(ge::aclgrphParseCaffe(empty_layers.c_str(), weight_file.c_str(), parser_params, graph), FAILED);

    unlink(empty_layer.c_str());
    unlink(empty_layers.c_str());
  }
}

TEST_F(UtestCaffeParser, ParseFromMemory_success)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/caffe_model/caffe_add.pbtxt";
  std::string weight_file = caseDir + "/caffe_model/caffe_add.caffemodel";

  const char* tmp_tf_pb_model = modelFile.c_str();
  const char* tmp_tf_weight_model = weight_file.c_str();
  ge::Graph graph;

  Status ret = ge::aclgrphParseCaffe(modelFile.c_str(), weight_file.c_str(), graph);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  CaffeModelParser modelParser;
  MemBuffer* memBuffer1 = ParerUTestsUtils::MemBufferFromFile(tmp_tf_pb_model);
  ret = modelParser.ParseFromMemory((char*)memBuffer1->data, memBuffer1->size, compute_graph);
  EXPECT_EQ(ret, GRAPH_FAILED);

  CaffeWeightsParser weigthParser;
  MemBuffer* memBuffer2 = ParerUTestsUtils::MemBufferFromFile(tmp_tf_weight_model);
  ret = weigthParser.ParseFromMemory((char*)memBuffer2->data, memBuffer2->size, compute_graph);
  free(memBuffer1->data);
  free(memBuffer2->data);
  delete memBuffer1;
  delete memBuffer2;
}

TEST_F(UtestCaffeParser, caffe_parser_to_json) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe_add.pbtxt";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  CaffeModelParser caffe_parser;

  const char *json_file = "tmp.json";
  auto ret = caffe_parser.ToJson(model_file.c_str(), json_file);
  EXPECT_EQ(ret, SUCCESS);

  const char *json_null = nullptr;
  ret = caffe_parser.ToJson(model_file.c_str(), json_null);
  EXPECT_EQ(ret, FAILED);
  const char *model_null = nullptr;
  ret = caffe_parser.ToJson(model_null, json_null);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, caffe_parser_ParseParamsForDummyData_test)
{
  CaffeDataParser caffe_parser;
  domi::caffe::NetParameter net;
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("conv", "Convolution");
  domi::caffe::LayerParameter *lay = net.add_layer();
  Status ret = caffe_parser.ParseParamsForDummyData(lay, op);
  EXPECT_EQ(ret, FAILED);

  ret = caffe_parser.ParseParamsForInput(lay, op);
  EXPECT_EQ(ret, FAILED);

  domi::caffe::DummyDataParameter *dummyData = lay->mutable_dummy_data_param();
  ret = caffe_parser.ParseParamsForDummyData(lay, op);
  EXPECT_EQ(ret, FAILED);

  domi::caffe::BlobShape* dummpShape = dummyData->add_shape();
  ret = caffe_parser.ParseParamsForDummyData(lay, op);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, convertWeights_success)
{
  CaffeOpParser parser;
  ge::GeTensorDesc ge_tensor_desc = ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->set_int8_data("12");
  blob->add_data(1);
  blob->add_data(1);

  domi::caffe::BlobShape *shap1 = blob->mutable_shape();
  shap1->add_dim(1);
  shap1->add_dim(2);
  shap1->add_dim(-1);

  Status ret = parser.ConvertWeight(*blob, "", weight);
  EXPECT_EQ(FAILED, ret);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeCustomParserAdapter_ParseWeights_success)
{
  CaffeCustomParserAdapter parserAdapter;
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);
  LayerParameter* layer = new LayerParameter();
  Status ret = parserAdapter.ParseWeights(layer, node_tmp);
  EXPECT_EQ(ret, SUCCESS);

  BlobProto* blob = layer->add_blobs();
  blob->add_data(1);
  blob->add_data(1);
  BlobShape* shap = blob->mutable_shape();
  shap->add_dim(1);
  shap->add_dim(2);

  ret = parserAdapter.ParseWeights(layer, node_tmp);
  EXPECT_EQ(ret, SUCCESS);

  delete layer;
}

TEST_F(UtestCaffeParser, CaffeCustomParserAdapter_ParseParams_success)
{
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>("Data", "Input");

  CaffeCustomParserAdapter parserAdapter;
  Status ret = parserAdapter.ParseParams(op_src, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestCaffeParser, CaffeDataParser_ParseParams_success)
{
  domi::caffe::NetParameter net;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  domi::caffe::LayerParameter* lay0 = net.add_layer();
  lay0->set_name("conv");
  lay0->set_type(ge::parser::DUMMY_DATA);

  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  CaffeDataParser parserAdapter;
  Status ret = parserAdapter.ParseParams(lay0, opDef);
  EXPECT_EQ(ret, FAILED);

  lay0->set_type(ge::parser::ATTR_NAME_INPUT_TENSOR_DESC);
  ret = parserAdapter.ParseParams(lay0, opDef);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_Parse_test)
{
  CaffeWeightsParser weightParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe_add.caffemodel";
  const char *file = nullptr;
  ge::ComputeGraphPtr graph;
  Status ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  file = model_file.c_str();
  ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  graph = std::make_shared<ComputeGraph>("test");
  ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, FAILED);

  std::string caffe_proto = case_dir + "/../../../../metadef/proto/caffe/";
  std::string custom_proto = case_dir + "/caffe_model/";
  ge::GetParserContext().caffe_proto_path.assign(caffe_proto);
  ge::GetParserContext().custom_proto_path.assign(custom_proto);
  ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, FAILED);

  custom_proto = case_dir + "/caffe_models/";
  caffe_proto = case_dir + "/../../../../../metadef/proto/caffe/";
  ge::GetParserContext().caffe_proto_path.assign(caffe_proto);
  ge::GetParserContext().custom_proto_path.assign(custom_proto);
  ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ParseWeightByFusionProto_test)
{
  CaffeWeightsParser weightParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string weight_file = case_dir + "/caffe_model/caffe_add.caffemodel";
  std::string model_file = case_dir + "/../../../../metadef/proto/caffe/caffe.proto";
  const char *weight_path = model_file.c_str();
  std::string fusion_proto_path = model_file;
  std::string fusion_proto_name = "caffe";
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  Status ret = weightParser.ParseWeightByFusionProto(weight_path, fusion_proto_path, fusion_proto_name, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ParseFromMemory_test)
{
  CaffeWeightsParser weightParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string weight_file = case_dir + "/caffe_model/caffe_add.caffemodel";
  ge::ComputeGraphPtr graph;
  const char *data = nullptr;
  Status ret = weightParser.ParseFromMemory(data, 1, graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  data = weight_file.c_str();
  ret = weightParser.ParseFromMemory(data, 1, graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  graph = std::make_shared<ComputeGraph>("test");
  ret = weightParser.ParseFromMemory(data, 1, graph);
  EXPECT_EQ(ret, domi::PARSE_WEIGHTS_FAILED);

  CaffeModelParser model_parser;
  ret = model_parser.ParseFromMemory(data, 1, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_CreateCustomOperator_test)
{
  CaffeModelParser model_parser;

  vector<ge::Operator> operators;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  operators.emplace_back(op_src);
  std::string op_name = "";
  std::string op_type = "";
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *lay0 = net.add_layer();
  lay0->set_name("Data");
  lay0->set_type("Input");
  Status ret = model_parser.CreateCustomOperator(op_name, op_type, &net, 1, operators);
  EXPECT_EQ(ret, FAILED);

  op_name = "Data";
  op_type = "Input";
  ret = model_parser.CreateCustomOperator(op_name, op_type, &net, 1, operators);
  EXPECT_EQ(ret, SUCCESS);

  model_parser.AddOutputInfoToContext(op_name, 1);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ParseOutputNodeTopInfo_test)
{
  CaffeModelParser model_parser;
  AclGrphParseUtil acl_graph_parse_util;

  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *lay0 = net.add_layer();
  lay0->set_name("Data");
  lay0->set_type("Input");
  Status ret = model_parser.ParseOutputNodeTopInfo(net);
  EXPECT_EQ(ret, SUCCESS);

  GetParserContext().type = domi::CAFFE;
  string graph_name;
  std::map<AscendString, AscendString> out_nodes_with_tensor_name1 = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2")}};
  acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_tensor_name1, graph_name);
  ret = model_parser.ParseOutputNodeTopInfo(net);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestCaffeParser, CaffeOpParser_ParseWeightType_test)
{
  CaffeOpParser opParser;
  ge::GeTensorDesc ge_tensor_desc =  ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->set_int8_data("10");
  std::string lay_name = "DATA";
  GeShape shape({1,1,3,4});
  Status ret = opParser.ParseWeightType(*blob, shape, 1, lay_name, weight);
  EXPECT_EQ(ret, FAILED);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeOpParser_ParseWeightType_test2)
{
  CaffeOpParser opParser;
  ge::GeTensorDesc ge_tensor_desc =  ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->add_int32_data(10);

  std::string lay_name = "DATA";
  GeShape shape({1,1,3,4});
  Status ret = opParser.ParseWeightType(*blob, shape, 1, lay_name, weight);
  EXPECT_EQ(ret, SUCCESS);

  ret = opParser.ParseWeightType(*blob, shape, 2, lay_name, weight);
  EXPECT_EQ(ret, FAILED);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeOpParser_ParseWeightType_test3)
{
  CaffeOpParser opParser;
  ge::GeTensorDesc ge_tensor_desc =  ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  double value = 2.0;
  blob->add_double_data(value);

  std::string lay_name = "DATA";
  GeShape shape({1,1,3,4});
  Status ret = opParser.ParseWeightType(*blob, shape, 1, lay_name, weight);
  EXPECT_EQ(ret, SUCCESS);

  ret = opParser.ParseWeightType(*blob, shape, 3, lay_name, weight);
  EXPECT_EQ(ret, FAILED);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeOpParser_ParseWeightType_test4)
{
  CaffeOpParser opParser;
  ge::GeTensorDesc ge_tensor_desc = ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->add_uint64_data(10);

  std::string lay_name = "DATA";
  GeShape shape({1,1,3,4});
  Status ret = opParser.ParseWeightType(*blob, shape, 1, lay_name, weight);
  EXPECT_EQ(ret, SUCCESS);

  ret = opParser.ParseWeightType(*blob, shape, 2, lay_name, weight);
  EXPECT_EQ(ret, FAILED);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeOpParser_ParseWeightType_test5)
{
  CaffeOpParser opParser;
  ge::GeTensorDesc ge_tensor_desc =  ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->add_data(10);

  std::string lay_name = "DATA";
  GeShape shape({1,1,3,4});
  Status ret = opParser.ParseWeightType(*blob, shape, 10, lay_name, weight);
  EXPECT_EQ(ret, FAILED);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeOpParser_ConvertShape_test)
{
  CaffeOpParser opParser;
  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->set_num(1);
  blob->set_channels(2);
  blob->set_height(1);
  blob->set_width(1);
  std::vector<int64_t> shape;

  opParser.ConvertShape(*blob, shape);
  delete layer;
}

TEST_F(UtestCaffeParser, CaffeModelParser_ParseInput_test)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  net.add_input("111");
  net.add_input_dim(1);
  bool input_data_flag = true;

  Status ret = modelParser.ParseInput(net, input_data_flag);
  EXPECT_EQ(ret, FAILED);

  net.add_input_dim(2);
  net.add_input_dim(3);
  net.add_input_dim(4);
  domi::caffe::LayerParameter *lay0 = net.add_layer();
  BlobProto* blob = lay0->add_blobs();
  blob->add_data(1);
  blob->add_data(1);
  BlobShape* shap = blob->mutable_shape();
  shap->add_dim(1);
  shap->add_dim(2);
  ret = modelParser.ParseInput(net, input_data_flag);
  EXPECT_EQ(ret, SUCCESS);

  net.add_input_shape();
  ret = modelParser.ParseInput(net, input_data_flag);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeModelParser_ParseInput_test2)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  net.add_input("111");
  bool input_data_flag = true;

  net.add_input_shape();
  Status ret = modelParser.ParseInput(net, input_data_flag);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_CustomProtoParse_test)
{
  CaffeModelParser modelParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/";
  const char *model_path = model_file.c_str();

  std::string custom_proto = model_file;
  std::string caffe_proto = model_file;
  std::vector<ge::Operator> operators;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  operators.emplace_back(op_src);

  Status ret = modelParser.CustomProtoParse(model_path, custom_proto, caffe_proto, operators);
  EXPECT_EQ(ret, PARAM_INVALID);

  model_file = case_dir + "/caffe_model/caffe_add.pbtxt";
  custom_proto = case_dir + "/../../../../../metadef/proto/caffe/caffe.proto";
  model_path = model_file.c_str();
  std::string caffe_proto_path = case_dir + "/../../../../../metadef/proto/caffe/caffe.proto";
  ret = modelParser.CustomProtoParse(model_path, custom_proto, caffe_proto_path, operators);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ParseGraph_test)
{
  CaffeWeightsParser weightParser;
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmp_graph");
  ge::Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string weight_file = case_dir + "/caffe_model/caffe_add.caffemodel";
  const char *file = weight_file.c_str();

  Status ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ConvertNetParameter_test)
{
  CaffeWeightsParser weightParser;
  domi::caffe::NetParameter net;

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  domi::caffe::LayerParameter *lay0 = net.add_layer();
  lay0->set_name("Data");
  lay0->set_type("Input");

  Status ret = weightParser.ConvertNetParameter(net, graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_IsOpAttrEmpty_test)
{
  CaffeModelParser model_parser;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  std::string type = "custom";

  bool ret = model_parser.IsOpAttrEmpty(op_src, type);
  EXPECT_EQ(ret, true);

  type = "built-in";
  ret = model_parser.IsOpAttrEmpty(op_src, type);
  EXPECT_EQ(ret, true);
}

TEST_F(UtestCaffeParser, CaffeModelParser_GetCustomOp_test)
{
  CaffeModelParser model_parser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Data");
  layer->set_type("Input");

  vector<ge::Operator> operators;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  operators.emplace_back(op_src);

  Status ret = model_parser.GetCustomOp(*layer, operators);
  EXPECT_EQ(ret, SUCCESS);

  ge::Operator ops2("Conv", "Convolution");
  model_parser.custom_operator_.push_back(ops2);
  ret = model_parser.GetCustomOp(*layer, operators);
  EXPECT_EQ(ret, SUCCESS);

  ge::Operator ops("Data", "Input");
  model_parser.custom_operator_.push_back(ops);
  ret = model_parser.GetCustomOp(*layer, operators);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_AddTensorDescToOpDesc_test)
{
  CaffeModelParser model_parser;
  domi::caffe::NetParameter net;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Abs", "AbsVal");
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");
  layer->add_bottom("Abs");

  Status ret = model_parser.AddTensorDescToOpDesc(op_desc_src, *layer);
  EXPECT_EQ(ret, SUCCESS);

  op_desc_src = std::make_shared<ge::OpDesc>("Abs", "YoloDetectionOutput");
  layer->set_type("YoloDetectionOutput");
  layer->add_top("top");
  ret = model_parser.AddTensorDescToOpDesc(op_desc_src, *layer);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ConvertLayerParameter_test)
{
  CaffeWeightsParser weightParser;
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmp_graph");
  domi::caffe::NetParameter net;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Abs", "AbsVal");
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  Status ret = weightParser.ConvertLayerParameter(layer, compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string caffe_proto = case_dir + "/../../../../../metadef/proto/caffe/";
  google::protobuf::compiler::DiskSourceTree sourceTree;
  sourceTree.MapPath("project_root", caffe_proto);
  google::protobuf::compiler::Importer importer(&sourceTree, nullptr);
  importer.Import("project_root/caffe.proto");

  auto descriptor = importer.pool()->FindMessageTypeByName("domi.caffe.LayerParameter");
  google::protobuf::DynamicMessageFactory factory;
  const google::protobuf::Message *proto = factory.GetPrototype(descriptor);
  const google::protobuf::Message *message = proto->New();

  ret = weightParser.ConvertLayerParameter(layer, compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  delete message;
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_CheckLayersSize_test)
{
  CaffeWeightsParser weightParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  Status ret = weightParser.CheckLayersSize(layer);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ConvertLayerProto_test)
{
  CaffeWeightsParser weightParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  Status ret = weightParser.ConvertLayerProto(&net, &net);
  EXPECT_EQ(ret, SUCCESS);

  BlobProto* blob = layer->add_blobs();
  blob->add_data(1);
  blob->add_data(1);
  BlobShape* shap = blob->mutable_shape();
  shap->add_dim(1);
  shap->add_dim(2);
  ret = weightParser.ConvertBlobsProto(&net, &net);
  EXPECT_EQ(ret, SUCCESS);

  ret = weightParser.ConvertBlobShapeProto(&net, &net);
  EXPECT_EQ(ret, SUCCESS);

  ret = weightParser.ConvertConvParamProto(&net, &net);
  EXPECT_EQ(ret, SUCCESS);

  ret = weightParser.ConvertInnerProdcutProto(&net, &net);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_CheckNodes_test)
{
  CaffeWeightsParser weightParser;
  ge::ComputeGraphPtr compute_graph = build_graph(true);
  Status ret = weightParser.CheckNodes(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_RemapTopNameByLayer_test)
{
  CaffeModelParser model_parser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  std::string top_name = "Abs";
  int index = 1;

  model_parser.RemapTopNameByLayer(*layer, top_name, index);
}

TEST_F(UtestCaffeParser, CaffeModelParser_SaveDataLayerTops_test)
{
  CaffeModelParser model_parser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  Status ret = model_parser.SaveDataLayerTops(*layer);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeModelParser_ReadCaffeModelFromText_test)
{
  CaffeModelParser modelParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe.pbtxt";
  const char *model_path = model_file.c_str();

  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");
  Status ret = modelParser.ReadCaffeModelFromText(model_path, &net);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeReshapeParser_ParseParams_test)
{
  CaffeReshapeParser reshapeParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  domi::caffe::ReshapeParameter* reshape_param = layer->mutable_reshape_param();
  layer->add_bottom("bottom");
  layer->add_top("top");

  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Abs", "AbsVal");
  Status ret = reshapeParser.ParseParams(layer, op_desc_src);
  EXPECT_EQ(ret, FAILED);

  domi::caffe::BlobShape *blob_shape = reshape_param->mutable_shape();
  blob_shape->add_dim(2);
  blob_shape->add_dim(3);
  reshape_param->set_axis(0);
  reshape_param->set_num_axes(-1);
  ret = reshapeParser.ParseParams(layer, op_desc_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeReshapeParser_ParseWeights_test)
{
  CaffeReshapeParser reshapeParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  domi::caffe::ReshapeParameter* reshape_param = layer->mutable_reshape_param();
  layer->add_bottom("bottom");
  layer->add_top("top");

  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Abs", "AbsVal");
  Status ret = reshapeParser.ParseWeights(layer, op_desc_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_ParseOpParam_test)
{
  CaffeModelParser modelParser;

  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("AbsVal");
  layer->set_type("AbsVal");

  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Abs", "AbsVal");

  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::CAFFE);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("Abs");

  Status ret = modelParser.ParseOpParam(*layer, op_desc_src, op_parser);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestCaffeParser, CaffeModelParser_AddNode_test)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("AbsVal");
  layer->set_type("DetectionOutput");
  ge::ComputeGraphPtr compute_graph = build_graph(true);

  Status ret = modelParser.AddNode(*layer, compute_graph);
  EXPECT_EQ(ret, FAILED);

  layer->set_type("ProposalLayer");
  ret = modelParser.AddNode(*layer, compute_graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeModelParser_CheckValidLayer_test)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");
  layer->add_include();
  bool ret = modelParser.CheckValidLayer(*layer);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestCaffeParser, CaffeModelParser_ParseProto_test)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");
  ge::ComputeGraphPtr compute_graph = build_graph(true);

  Status ret = modelParser.ParseProto(&net, compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  domi::GetGraphCallback callback;
  ret = modelParser.ParseProtoWithSubgraph(&net, callback, compute_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeOpParser_ParseParams_test)
{
  CaffeOpParser opParser;
  domi::caffe::NetParameter net;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  domi::caffe::LayerParameter* lay0 = net.add_layer();
  lay0->set_name("conv");
  lay0->set_type(ge::parser::DUMMY_DATA);

  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");

  Status ret = opParser.ParseParams(lay0, opDef);
  EXPECT_EQ(ret, SUCCESS);

  ge::NodePtr node;
  ret = opParser.ParseWeights(lay0, node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_FindShareParamLayers_test)
{
  CaffeModelParser modelParser;
  std::map<std::string, std::vector<std::string>> layer_params_map;
  std::vector<std::string> layer_params;
  layer_params.emplace_back("Conv");
  layer_params.emplace_back("Data");
  layer_params.emplace_back("Abs");
  layer_params_map.insert(std::make_pair("Abs", layer_params));
  layer_params_map.insert(std::make_pair("Data", layer_params));
  layer_params_map.insert(std::make_pair("Conv", layer_params));

  Status ret = modelParser.FindShareParamLayers(layer_params_map);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ParseLayerParameter_test)
{
  CaffeWeightsParser weightParser;

  domi::caffe::NetParameter net;
  GetParserContext().type = domi::CAFFE;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  ge::ComputeGraphPtr compute_graph = build_graph(true);
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string caffe_proto = case_dir + "/../../../../../metadef/proto/caffe/";
  google::protobuf::compiler::DiskSourceTree sourceTree;
  sourceTree.MapPath("project_root", caffe_proto);
  google::protobuf::compiler::Importer importer(&sourceTree, nullptr);
  importer.Import("project_root/caffe.proto");

  auto descriptor = importer.pool()->FindMessageTypeByName("domi.caffe.LayerParameter");
  google::protobuf::DynamicMessageFactory factory;
  const google::protobuf::Message *proto = factory.GetPrototype(descriptor);
  const google::protobuf::Message *message = proto->New();

  Status ret = weightParser.ParseLayerParameter(descriptor, message, compute_graph);
  delete message;
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCaffeParser, CaffeModelParser_ParseLayerParameter_test)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  vector<ge::Operator> operators;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  operators.emplace_back(op_src);

  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string caffe_proto = case_dir + "/../../../../../metadef/proto/caffe/";
  google::protobuf::compiler::DiskSourceTree sourceTree;
  sourceTree.MapPath("project_root", caffe_proto);
  google::protobuf::compiler::Importer importer(&sourceTree, nullptr);
  importer.Import("project_root/caffe.proto");

  auto descriptor = importer.pool()->FindMessageTypeByName("domi.caffe.LayerParameter");
  google::protobuf::DynamicMessageFactory factory;
  const google::protobuf::Message *proto = factory.GetPrototype(descriptor);
  const google::protobuf::Message *message = proto->New();
  Status ret = modelParser.ParseLayerParameter(descriptor, message, operators);
  EXPECT_EQ(ret, SUCCESS);
  delete message;
}

TEST_F(UtestCaffeParser, CaffeModelParser_ReadModelWithoutWarning_test)
{
  CaffeModelParser modelParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/caffe_model/caffe.pbtxt";
  const char *model_path = model_file.c_str();

  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  Status ret = modelParser.ReadModelWithoutWarning(model_path, &net);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCaffeParser, CaffeModelParser_AddBlobsToMap_test)
{
  CaffeModelParser modelParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");
  std::map<std::string, std::string> inplace_blob_name_remapping{{"bottom", "AbsVal"}};

  layer->add_top("top");
  layer->add_bottom("bottom");
  modelParser.AddBlobsToMap(*layer, inplace_blob_name_remapping);
}

TEST_F(UtestCaffeParser, CaffeWeightsParser_ReorderInput_test)
{
  CaffeModelParser modelParser;

  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer1 = net.add_layer();
  layer1->set_name("Abs");
  layer1->set_type("AbsVal");

  domi::caffe::LayerParameter *layer2 = net.add_layer();
  layer2->set_name("Data");
  layer2->set_type("Input");
  modelParser.ReorderInput(net);
}

} // namespace ge
