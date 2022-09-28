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

#define protected public
#define private public
#include <iostream>
#include "parser/common/op_parser_factory.h"
#include "graph/operator_reg.h"
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"
#include "external/parser/onnx_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "external/ge/ge_api_types.h"
#include "parser/common/proto_file_parser.h"
#include "omg/parser/parser_factory.h"
#include "parser/caffe/caffe_parser.h"
#include "register/register.h"
#include "parser/common/pass_manager.h"
#include "parser/common/tbe_plugin_loader.h"
#include "parser/common/parser_fp16_t.h"
#include "parser/common/pre_checker.h"
#undef protected
#undef private

#include <dlfcn.h>
using namespace domi;
using namespace testing;
using namespace ge;

namespace ge {
class UtestAclGraphParser : public testing::Test {
 protected:
  void SetUp() {

  }
  void TearDown() {}
};

TEST_F(UtestAclGraphParser, test_parse_acl_output_nodes) {
  AclGraphParseUtil acl_graph_parse_util;
  string graph_name;
  // case 1: Normal with 'node and index'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_with_node_and_index = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out1:0;Out2:1")}};
  ParerUTestsUtils::ClearParserInnerCtx();
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 2);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 2);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 0);

  // case 2: Normal with 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_with_tensor_name = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_tensor_name, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 2);

  // case 3: Failed with 'node and index' before 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_pre = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out1:0;Out2:1;Out_tensor_1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_pre, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 2);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 2);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 0);

  // case 4: Failed with 'node and index' inserted in 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_mid = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out1:0;Out2:1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_mid, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 1);

  // case 5: Failed with 'node and index' after 'tensor name'
  ParerUTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_post = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2;Out1:0;Out2:1")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_post, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 2);

}

TEST_F(UtestAclGraphParser, test_CheckConflictOp)
{
  ge::ProtoFileParser op;
  std::string custom_file = "/dev/null";
  const char *caffe_proto_file = custom_file.c_str();
  const char *custom_proto_file = custom_file.c_str();
  std::map<std::string, std::pair<int, string>> caffe_op_identifier_map;
  std::map<std::string, std::pair<int, string>> custom_op_identifier_map;
  custom_op_identifier_map.insert(std::make_pair("ge", std::make_pair(1, "ge")));
  caffe_op_identifier_map.insert(std::make_pair("ge", std::make_pair(1, "ge")));
  op.CheckConflictOp(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map);

  caffe_op_identifier_map.clear();
  caffe_op_identifier_map.insert(std::make_pair("ge", std::make_pair(2, "ge")));
  op.CheckConflictOp(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map);
}

TEST_F(UtestAclGraphParser, test_CheckConflictIdentifier)
{
  ge::ProtoFileParser op;
  char *caffe_proto_file = "/dev/null";
  char *custom_proto_file = "/dev/null";
  std::map<int, std::pair<string, string>> caffe_op_identifier_map;
  std::map<int, std::pair<string, string>> custom_op_identifier_map;
  custom_op_identifier_map.insert(std::make_pair(1, std::make_pair("ge", "ge")));
  caffe_op_identifier_map.insert(std::make_pair(1, std::make_pair("ge", "ge")));
  op.CheckConflictIdentifier(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map);

  caffe_op_identifier_map.clear();
  caffe_op_identifier_map.insert(std::make_pair(1, std::make_pair("acl", "ge")));
  op.CheckConflictIdentifier(caffe_proto_file, custom_proto_file, caffe_op_identifier_map, custom_op_identifier_map);
}

TEST_F(UtestAclGraphParser, test_AddCustomAndConflictLayer)
{
  Status ret;
  char *custom_proto_file = "../parser/caffe/caffe_parser.h";
  ge::ProtoFileParser op;
  std::ofstream write_tmp;
  ret = op.ProtoFileParser::AddCustomAndConflictLayer(custom_proto_file, write_tmp);
  EXPECT_EQ(ret, SUCCESS);

  custom_proto_file = "/dev/ge";
  ret = op.ProtoFileParser::AddCustomAndConflictLayer(custom_proto_file, write_tmp);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_FindConflictLine)
{
  Status ret;
  ProtoFileParser op;
  int identifier = 0;
  std::string dest_line;
  string search_string("message=1,LayerParameter=1");
  string search_string1("optional=1 repeated=2 required=3 ");
  ret = op.FindConflictLine("../tests/ut/parser/testcase/common/acl_graph_parser_unittest.cc", identifier, dest_line);
  EXPECT_EQ(ret, FAILED);

  identifier = 1;
  ret = op.FindConflictLine("../tests/ut/parser/testcase/common/acl_graph_parser_unittest.cc", identifier, dest_line);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_ParseProtoFile)
{
  Status ret;
  ProtoFileParser op;
  std::string dest_line;
  std::map<int, std::pair<string, string>> identifier_op_map;
  std::map<std::string, std::pair<int, string>> op_identifier_map;
  string proto_file = "../tests/ut/parser/testcase/tensorflow_parser_testcase/tensorflow_parser_unittest.cc";
  ret = op.ParseProtoFile(proto_file, identifier_op_map, op_identifier_map);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_AddCustomAndConflictMessage)
{
  Status ret;
  ProtoFileParser op;
  std::ofstream write_tmp;
  std::string file = "../parser/caffe/caffe_parser.h";
  const char *proto_file = file.c_str();
  ret = op.AddCustomAndConflictMessage(proto_file, write_tmp);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_RecordProtoMessage)
{
  Status ret;
  ProtoFileParser op;
  std::string file = "../parser/caffe/caffe_parser.h";
  const char *proto_file = file.c_str();
  ret = op.RecordProtoMessage(proto_file);
  EXPECT_EQ(ret, SUCCESS);
}


TEST_F(UtestAclGraphParser, test_WriteCaffeProtoFile)
{
  Status ret;
  ProtoFileParser op;
  std::string file = "../parser/caffe/caffe_parser.h";
  const char *proto_file = file.c_str();
  std::ifstream read_caffe("../parser/caffe/caffe_parser.h", std::ifstream::in);
  std::ofstream write_tmp("/dev/null", std::ifstream::in);
  ret = op.WriteCaffeProtoFile(proto_file, read_caffe, write_tmp);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_CreatProtoFile)
{
  Status ret;
  ProtoFileParser op;
  op.fusion_proto_path = "/ge/ge/ge/ge.c";
  ret = op.CreatProtoFile();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_Finalize)
{
  bool ret;
  bool is_train = true;
  ge::OpRegistrationTbe op;
  ge::OpRegistrationData reg_data("c");
  ret = op.Finalize(reg_data, is_train);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestAclGraphParser, test_WriteProtoFile)
{
  Status ret;
  ProtoFileParser op;
  char *caffe_proto_file = "/dev/null";
  char *custom_proto_file = "/ge/ge/ge/ge.c";
  ret = op.WriteProtoFile(caffe_proto_file, custom_proto_file);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestAclGraphParser, test_GraphPasses)
{
  std::vector<std::pair<std::string, GraphPass *>> v;
  ge::parser::PassManager manager;
  v = manager.GraphPasses();
}

TEST_F(UtestAclGraphParser, test_ClearHandles_)
{
  Status ret;
  TBEPluginLoader loader;
  void *handle = dlopen("/lib/libdmmp.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (handle == nullptr) {
    return;
  }
  loader.handles_vec_.push_back(handle);
  dlclose(handle);
  ret = loader.ClearHandles_();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestAclGraphParser, test_operatoreq)
{
  float f_val1= 2139095000.2;
  ge::parser::fp16_t fp16_1,fp16_2;
  fp16_1.operator=(fp16_2);
  fp16_1.operator=(f_val1);

  float f_val2= 0.0000112;
  fp16_1.operator=(f_val2);

  float f_val3= 0.0000000299;
  fp16_1.operator=(f_val3);

  float f_val4= 0.00000000299;
  fp16_1.operator=(f_val4);

  uint32_t  u_val1 = 4095;
  fp16_1.operator=(u_val1);

  uint16_t u16_val1 = 4095;
  fp16_1.operator=(u16_val1);

  int16_t int_val1 = 0;
  fp16_1.operator=(int_val1);

  int16_t int_val2 = -32767;
  fp16_1.operator=(int_val2);

  int32_t i_val = -0x7FFFFFFF;
  fp16_1.operator=(i_val);

  parser::fp16_t fp16;
  fp16.operator=(f_val1);
  float f = fp16; //float();
  double d = fp16;
  int8_t int8 = fp16;
  uint8_t uint8 = fp16;
  uint16_t uint16 = fp16;
  int32_t int32 = fp16;
  uint32_t uint32 = fp16;
  int64_t int64 = fp16;
  uint64_t uint64 = fp16;

  (void)f;
  (void)d;
  (void)int8;
  (void)uint8;
  (void)uint8;
  (void)uint16;
  (void)int32;
  (void)uint32;
  (void)int64;
  (void)uint64;

  parser::fp16_t val;
  val.val = 0x7C00;
  val.IsInf();

  val.val = 0xFC00;
  val.IsInf();

  parser::fp16_t fp16_3, fp16_4;
  fp16_3.val = 1;
  fp16_4.val = 2;
  fp16_4.operator/(fp16_3);

  fp16.val = 21504;
  int16_t int16 = fp16;
  int8 = fp16;
}

TEST_F(UtestAclGraphParser, test_pre_checker) {
  TBEPluginLoader tbe_plugin;
  PreChecker::Instance().fmk_op_types_ = nullptr;
  const char* str = "iiii";
  PreChecker::OpId id = str;
  std::string type("ddd");
  std::string name("lll");
  Status ret = PreChecker::Instance().CheckTypeSupported(id, type, name, false);
  EXPECT_EQ(ret, FAILED);
  ret = PreChecker::Instance().CheckTypeSupported(id, type, name, true);
  EXPECT_EQ(ret, FAILED);
}
} // namespace ge