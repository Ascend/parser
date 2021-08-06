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
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"
#include "external/parser/onnx_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "external/ge/ge_api_types.h"

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

REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)

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

REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Identity)
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
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
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
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
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
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
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

} // namespace ge