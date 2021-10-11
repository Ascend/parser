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
#define private public
#define protected public
#include "parser/tensorflow/tensorflow_parser.h"
#undef protected
#undef private
#include "framework/omg/parser/parser_factory.h"
#include "graph/operator_reg.h"
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"
#include "external/parser/tensorflow_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "graph/model.h"

namespace ge {
class UtestTensorflowParser : public testing::Test {
 protected:
  void SetUp() {
    ParerUTestsUtils::ClearParserInnerCtx();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
};

static Status ParseParams(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

void UtestTensorflowParser::RegisterCustomOp() {
  REGISTER_CUSTOM_OP("Add")
  .FrameworkType(domi::TENSORFLOW)
  .OriginOpType("Add")
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

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(Add)
}

TEST_F(UtestTensorflowParser, tensorflow_parser_success) {
  RegisterCustomOp();

  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/tensorflow_model/add.pb";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseTensorFlow(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, domi::SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_parser_with_serialized_proto1) {
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::TENSORFLOW);
  ge::graphStatus ret = model_parser->ParseProtoWithSubgraph(std::string(""),
      [](std::string)->std::string{ return "";}, compute_graph);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_parser_with_serialized_proto2) {
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::TENSORFLOW);
  ge::graphStatus ret = model_parser->ParseProtoWithSubgraph(std::string("null"),
      [](std::string)->std::string{ return "";}, compute_graph);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_parser_with_serialized_proto3) {
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::TENSORFLOW);

  domi::tensorflow::GraphDef graph_def;
  auto arg_node = graph_def.add_node();
  arg_node->set_name("noop");
  arg_node->set_op("NoOp");
  ge::TensorFlowModelParser modelparser;
  modelparser.nodedef_map_ = {0};
  
  ge::graphStatus ret = model_parser->ParseProtoWithSubgraph(graph_def.SerializeAsString(),
      [](std::string)->std::string{ return "";}, compute_graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_parser_with_external_graph) {
  auto make_graph = [](const string &name) {
    auto builder = ut::GraphBuilder(name);
    auto data1 = builder.AddNode(name + "_input1", "Data", 1, 1);
    auto data2 = builder.AddNode(name + "_input2", "Data", 1, 1);
    auto add = builder.AddNode(name + "_add", "Add", 2, 1);
    auto net_output = builder.AddNode(name + "_net_output", "NetOutput", 1, 1);
    builder.AddDataEdge(data1, 0, add, 0);
    builder.AddDataEdge(data2, 0, add, 1);
    builder.AddDataEdge(add, 0, net_output, 0);
    return builder.GetGraph();
  };
  // 1. Create root graph
  ComputeGraphPtr root_graph = make_graph("root_graph");

  // 2. Create ONNX sub graph
  // 2.1 Sub graph of onnx graph
  ge::ComputeGraphPtr sub_sub_graph = ge::parser::MakeShared<ge::ComputeGraph>("sub_sub");
  // 2.2 ONNX graph
  ComputeGraphPtr sub_graph = make_graph("sub_graph");
  auto add = sub_graph->FindNode("sub_graph_add");
  ASSERT_NE(add, nullptr);
  add->GetOpDesc()->AddSubgraphName("sub_sub_graph");
  add->GetOpDesc()->SetSubgraphInstanceName(0, sub_sub_graph->GetName());
  sub_graph->AddSubGraph(sub_sub_graph);
  auto input1 = sub_graph->FindNode("sub_graph_input1");
  ASSERT_NE(input1, nullptr);
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto input2 = sub_graph->FindNode("sub_graph_input2");
  ASSERT_NE(input2, nullptr);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_INDEX, 1);

  // 3. Serialize ONNX graph to string
  // 3.1 normal
  ge::Model model("model", "");
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(sub_graph));
  Buffer buffer;
  graphStatus save_ret = model.Save(buffer, false);
  ASSERT_EQ(save_ret, GRAPH_SUCCESS);
  std::string external_graph(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize());
  // model will failed
  input1->GetOpDesc()->DelAttr(ATTR_NAME_INDEX);
  ge::Model model_will_fail("model_will_fail", "");
  model_will_fail.SetGraph(GraphUtils::CreateGraphFromComputeGraph(sub_graph));
  Buffer buffer_fail;
  save_ret = model_will_fail.Save(buffer_fail, false);
  ASSERT_EQ(save_ret, GRAPH_SUCCESS);
  std::string external_graph_fail(reinterpret_cast<const char *>(buffer_fail.GetData()), buffer_fail.GetSize());

  // 4. Set string to function node
  auto root_add = root_graph->FindNode("root_graph_add");
  ASSERT_NE(root_add, nullptr);
  AttrUtils::SetStr(root_add->GetOpDesc(), "_external_model", external_graph);
  auto root_input1 = root_graph->FindNode("root_graph_input1");
  ASSERT_NE(root_input1, nullptr);
  AttrUtils::SetInt(root_input1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  auto root_input2 = root_graph->FindNode("root_graph_input2");
  ASSERT_NE(root_input2, nullptr);
  AttrUtils::SetInt(root_input2->GetOpDesc(), ATTR_NAME_INDEX, 1);

  // 5. Run test (normal)
  auto ret = TensorFlowModelParser::AddExternalGraph(root_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(root_graph->GetAllSubgraphs().size(), 2);
  EXPECT_EQ(sub_graph->GetAllSubgraphs().size(), 1);
  EXPECT_NE(root_graph->GetSubgraph(sub_graph->GetName()), nullptr);
  EXPECT_EQ(root_graph->GetSubgraph(sub_graph->GetName())->GetAllSubgraphs().size(), 0);

  // 6. Run test (failed)
  // 6.1 Failed to load model
  AttrUtils::SetStr(root_add->GetOpDesc(), "_external_model", "dummy string");
  ret = TensorFlowModelParser::AddExternalGraph(root_graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // 6.2 Failed to map sub graph
  AttrUtils::SetStr(root_add->GetOpDesc(), "_external_model", external_graph_fail);
  ret = TensorFlowModelParser::AddExternalGraph(root_graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // 6.3 Failed to set sub graph to node
  AttrUtils::SetStr(root_add->GetOpDesc(), "_external_model", external_graph);
  root_add->SetOwnerComputeGraph(nullptr);
  ret = TensorFlowModelParser::AddExternalGraph(root_graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // 6.4 Failed to add sub sub graph
  root_add->SetOwnerComputeGraph(nullptr);
  root_graph->RemoveSubGraph(sub_graph);
  ret = TensorFlowModelParser::AddExternalGraph(root_graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}
} // namespace ge