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

TEST_F(STestCaffeParser, acal_caffe_parser) {
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

} // namespace ge
