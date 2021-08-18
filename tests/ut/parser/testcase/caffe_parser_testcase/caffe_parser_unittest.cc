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

static Status ParseParams(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}
void UtestCaffeParser::RegisterCustomOp() {
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

namespace {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

REG_OP(Abs)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Abs)
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

} // namespace ge