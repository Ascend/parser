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
#include "parser/tensorflow/tensorflow_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "graph/operator_reg.h"
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"
#include "external/parser/tensorflow_parser.h"


namespace ge {
class UtestTensorflowParser : public testing::Test {
 protected:
  void SetUp() {}

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

  ge::graphStatus ret = model_parser->ParseProtoWithSubgraph(graph_def.SerializeAsString(),
      [](std::string)->std::string{ return "";}, compute_graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

} // namespace ge