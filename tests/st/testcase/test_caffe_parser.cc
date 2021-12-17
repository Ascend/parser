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
#include "parser/common/op_parser_factory.h"
#include "graph/operator_reg.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "external/parser/caffe_parser.h"
#include "st/parser_st_utils.h"
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
#undef protected
#undef private

using namespace domi::caffe;
using namespace ge;

namespace ge {
class STestCaffeParser : public testing::Test {
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

static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr opDesc){
  if (!opDesc) {
    return nullptr;
  }
  static auto g = std::make_shared<ge::ComputeGraph>("g");
  return g->AddNode(std::move(opDesc));
}

void STestCaffeParser::RegisterCustomOp() {
  REGISTER_CUSTOM_OP("Data")
  .FrameworkType(domi::CAFFE)
  .OriginOpType("Input")
  .ParseParamsFn(ParseParams);

  REGISTER_CUSTOM_OP("Abs")
    .FrameworkType(domi::CAFFE)
    .OriginOpType("AbsVal")
    .ParseParamsFn(ParseParams);

  std::vector<OpRegistrationData> reg_datas = domi::OpRegistry::Instance()->registrationDatas;
  for (auto reg_data : reg_datas) {
    OpRegistrationTbe::Instance()->Finalize(reg_data);
    domi::OpRegistry::Instance()->Register(reg_data);
  }
  domi::OpRegistry::Instance()->registrationDatas.clear();
}

TEST_F(STestCaffeParser, caffe_parser_user_output_with_default) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/caffe_abs.pbtxt";
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::CAFFE);
  ASSERT_NE(model_parser, nullptr);
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmp_graph");
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

TEST_F(STestCaffeParser, acl_caffe_parser) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/caffe_add.pbtxt";
  std::string weight_file_txt = case_dir + "/origin_models/caffe_add.caffemodel.txt";
  std::string weight_file = case_dir + "/origin_models/caffe_add.caffemodel";

  domi::caffe::NetParameter proto;
  EXPECT_EQ(ParerSTestsUtils::ReadProtoFromText(weight_file_txt.c_str(), &proto), true);
  ParerSTestsUtils::WriteProtoToBinaryFile(proto, weight_file.c_str());

  ge::GetParserContext().caffe_proto_path = case_dir + "/../../../../metadef/proto/caffe/caffe.proto";

  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseCaffe(model_file.c_str(), weight_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = ge::aclgrphParseCaffe(model_file.c_str(), weight_file.c_str(), graph);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(STestCaffeParser, modelparser_parsefrommemory_success)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/caffe_add.pbtxt";
  const char* tmp_tf_pb_model = modelFile.c_str();
  ge::Graph graph;

  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  CaffeModelParser modelParser;
  MemBuffer* memBuffer = ParerSTestsUtils::MemBufferFromFile(tmp_tf_pb_model);
  auto ret = modelParser.ParseFromMemory((char*)memBuffer->data, memBuffer->size, compute_graph);
  free(memBuffer->data);
  delete memBuffer;
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(STestCaffeParser, caffe_parser_to_json) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/caffe_add.pbtxt";
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

TEST_F(STestCaffeParser, caffe_parser_ParseParamsForDummyData_test)
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

TEST_F(STestCaffeParser, convertWeights_success)
{
  CaffeOpParser parser;
  ge::GeTensorDesc ge_tensor_desc =  ge::GeTensorDesc();
  ge::GeTensorPtr weight = std::make_shared<ge::GeTensor>(ge_tensor_desc);
  ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>("","");
  auto node_tmp = GenNodeFromOpDesc(opDef);

  domi::caffe::LayerParameter *layer = new domi::caffe::LayerParameter();
  domi::caffe::BlobProto *blob = layer->add_blobs();
  blob->set_int8_data("12");
  blob->add_data(1);
  blob->add_data(1);

  domi::caffe::BlobShape *shap = blob->mutable_shape();
  shap->add_dim(1);
  shap->add_dim(2);

  Status ret = parser.ConvertWeight(*blob, "", weight);
  EXPECT_EQ(domi::SUCCESS, ret);
  delete layer;
}

TEST_F(STestCaffeParser, CaffeCustomParserAdapter_ParseWeights_success)
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

TEST_F(STestCaffeParser, CaffeCustomParserAdapter_ParseParams_success)
{
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>("Data", "Input");

  CaffeCustomParserAdapter parserAdapter;
  Status ret = parserAdapter.ParseParams(op_src, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(STestCaffeParser, CaffeDataParser_ParseParams_success)
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

TEST_F(STestCaffeParser, CaffeWeightsParser_Parse_test)
{
  CaffeWeightsParser weightParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/ResNet-50-model.caffemodel";
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
}

TEST_F(STestCaffeParser, CaffeWeightsParser_ParseWeightByFusionProto_test)
{
  CaffeWeightsParser weightParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string weight_file = case_dir + "/origin_models/ResNet-50-model.caffemodel";
  std::string model_file = case_dir + "/origin_models/caffe.proto";
  const char *weight_path = model_file.c_str();
  std::string fusion_proto_path = model_file;
  std::string fusion_proto_name = "caffe";
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  Status ret = weightParser.ParseWeightByFusionProto(weight_path, fusion_proto_path, fusion_proto_name, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestCaffeParser, CaffeWeightsParser_ParseFromMemory_test)
{
  CaffeWeightsParser weightParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string weight_file = case_dir + "/origin_models/ResNet-50-model.caffemodel";
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

TEST_F(STestCaffeParser, CaffeWeightsParser_CreateCustomOperator_test)
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

TEST_F(STestCaffeParser, CaffeWeightsParser_ParseOutputNodeTopInfo_test)
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

TEST_F(STestCaffeParser, CaffeOpParser_ParseWeightType_test)
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

TEST_F(STestCaffeParser, CaffeOpParser_ParseWeightType_test2)
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

TEST_F(STestCaffeParser, CaffeOpParser_ParseWeightType_test3)
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

TEST_F(STestCaffeParser, CaffeOpParser_ParseWeightType_test4)
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

TEST_F(STestCaffeParser, CaffeOpParser_ParseWeightType_test5)
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

TEST_F(STestCaffeParser, CaffeOpParser_ConvertShape_test)
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

TEST_F(STestCaffeParser, CaffeModelParser_ParseInput_test)
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

TEST_F(STestCaffeParser, CaffeModelParser_CustomProtoParse_test)
{
  CaffeModelParser modelParser;
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/";
  const char *model_path = model_file.c_str();

  std::string custom_proto = model_file;
  std::string caffe_proto = model_file;
  std::vector<ge::Operator> operators;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("Data", "Input");
  ge::Operator op_src = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);
  operators.emplace_back(op_src);

  Status ret = modelParser.CustomProtoParse(model_path, custom_proto, caffe_proto, operators);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(STestCaffeParser, CaffeWeightsParser_ParseGraph_test)
{
  CaffeWeightsParser weightParser;
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmp_graph");
  ge::Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string weight_file = case_dir + "/origin_models/ResNet-50-model.caffemodel";
  const char *file = weight_file.c_str();

  Status ret = weightParser.Parse(file, graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestCaffeParser, CaffeWeightsParser_ConvertNetParameter_test)
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

TEST_F(STestCaffeParser, CaffeModelParser_IsOpAttrEmpty_test)
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

TEST_F(STestCaffeParser, CaffeModelParser_GetCustomOp_test)
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
}

TEST_F(STestCaffeParser, CaffeModelParser_AddTensorDescToOpDesc_test)
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
}

TEST_F(STestCaffeParser, CaffeWeightsParser_ConvertLayerParameter_test)
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
}

TEST_F(STestCaffeParser, CaffeWeightsParser_CheckLayersSize_test)
{
  CaffeWeightsParser weightParser;
  domi::caffe::NetParameter net;
  domi::caffe::LayerParameter *layer = net.add_layer();
  layer->set_name("Abs");
  layer->set_type("AbsVal");

  Status ret = weightParser.CheckLayersSize(layer);
  EXPECT_EQ(ret, FAILED);
}

} // namespace ge
