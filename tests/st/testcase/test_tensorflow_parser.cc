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
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_parser.h"
#include "graph/operator_reg.h"
#include "register/op_registry.h"
#include "external/register/register.h"
#include "parser/common/register_tbe.h"
#include "st/parser_st_utils.h"
#include "tests/depends/ops_stub/ops_stub.h"
#include "parser/common/acl_graph_parser_util.h"
#include "metadef/third_party/graphengine/inc/external/ge/ge_api_types.h"
#include "omg/parser/parser_factory.h"
#include "common/pre_checker.h"
#include "common/util.h"
#include "external/parser/tensorflow_parser.h"
#include "parser/tensorflow/tensorflow_constant_parser.h"
#include "common/types.h"
#include "parser/common/op_def/variable_op.h"
#include "parser/tensorflow/tensorflow_ref_switch_parser.h"
#undef protected
#undef private

using namespace std;
using namespace domi::tensorflow;
using namespace domi;
using namespace testing;
using namespace std;
using namespace google::protobuf;

static const string GRAPH_DEFAULT_NAME = "default";

namespace ge {
class STestTensorflowParser : public testing::Test {
 protected:
  void SetUp() {
    ParerSTestsUtils::ClearParserInnerCtx();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
};

static Status ParseParams(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

void STestTensorflowParser::RegisterCustomOp() {
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
  NodeDef *initNodeDef()
  {
      NodeDef * nodeDef = new NodeDef();
      nodeDef->set_op("Const");
      ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >* node_attr_map = nodeDef->mutable_attr();

      //设置 T属性
      domi::tensorflow::AttrValue t_attr_value;
      t_attr_value.set_type(domi::tensorflow::DT_INT32);
      (*node_attr_map)[TENSORFLOW_ATTR_T] = t_attr_value;

      domi::tensorflow::AttrValue dtype_attr_value;
      dtype_attr_value.set_type(domi::tensorflow::DT_INT32);
      (*node_attr_map)[TENSORFLOW_ATTR_DTYPE] = dtype_attr_value;

      // out_put
      domi::tensorflow::AttrValue outputs_attr_value;
      ::tensorflow::AttrValue_ListValue* list = outputs_attr_value.mutable_list();
      list->add_s("MatMul");
      (*node_attr_map)[TENSORFLOW_ATTR_OUTPUT_OP] = outputs_attr_value;

      // 设置 tensor 属性
      domi::tensorflow::AttrValue value_attr_value;
      ::tensorflow::TensorProto* tensor = value_attr_value.mutable_tensor();
      ::tensorflow::TensorShapeProto* tensor_shape = tensor->mutable_tensor_shape();
      tensor_shape->clear_dim();
      tensor_shape->add_dim()->set_size(4);
      tensor_shape->add_dim()->set_size(6);
      tensor->set_dtype(domi::tensorflow::DT_INT32);

      float *addr = new float[24];
      for (int32_t i = 0; i < 24; i++)
      {
          *(addr + i) = 1.0 + i;
      }
      tensor->set_tensor_content((void *)addr, 24 * sizeof(float));

      (*node_attr_map)[TENSORFLOW_ATTR_VALUE] = value_attr_value;
      delete[] addr;
      return nodeDef;
  }

  NodeDef *MallocNodeDef(const string &name, const string &type) {
    NodeDef* node_def = new (std::nothrow) NodeDef();
    if (node_def != nullptr) {
      node_def->set_name(name);
      node_def->set_op(type);
    }
    return node_def;
  }

  void GenOriginNodeDef(ge::TensorFlowModelParser *tensorflow_parser, vector<string> &node_name_list) {
    NodeDef* pre_node_a = MallocNodeDef("pre_node_a", "Const");
    EXPECT_NE(pre_node_a, nullptr);
    {
      ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >* node_attr_map = pre_node_a->mutable_attr();
      tensorflow::AttrValue attr_dtype;
      attr_dtype.set_type(tensorflow::DT_FLOAT);
      (*node_attr_map)["dtype"] = attr_dtype;
      tensorflow::AttrValue attr_value;
      tensorflow::TensorProto* tensor = attr_value.mutable_tensor();
      tensor->add_bool_val(true);
      tensor->set_dtype(tensorflow::DT_BOOL);
      (*node_attr_map)["value"] = attr_value;
    }
    tensorflow_parser->nodedef_map_["pre_node_a"] = pre_node_a;
    node_name_list.push_back("pre_node_a");

    NodeDef* pre_node_ctrl_in = MallocNodeDef("pre_node_ctrl_in", "Const");
    EXPECT_NE(pre_node_ctrl_in, nullptr);
    {
      ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >* node_attr_map = pre_node_ctrl_in->mutable_attr();
      tensorflow::AttrValue attr_dtype;
      attr_dtype.set_type(tensorflow::DT_FLOAT);
      (*node_attr_map)["dtype"] = attr_dtype;
      tensorflow::AttrValue attr_value;
      tensorflow::TensorProto* tensor = attr_value.mutable_tensor();
      tensor->add_bool_val(true);
      tensor->set_dtype(tensorflow::DT_BOOL);
      (*node_attr_map)["value"] = attr_value;
    }
    tensorflow_parser->nodedef_map_["pre_node_ctrl_in"] = pre_node_ctrl_in;
    node_name_list.push_back("pre_node_ctrl_in");

    NodeDef* post_node_b = MallocNodeDef("post_node_b", "Identity");
    EXPECT_NE(post_node_b, nullptr);
    tensorflow_parser->nodedef_map_["post_node_b"] = post_node_b;
    node_name_list.push_back("post_node_b");

    NodeDef* post_node_c = MallocNodeDef("post_node_c", "Identity");
    EXPECT_NE(post_node_c, nullptr);
    tensorflow_parser->nodedef_map_["post_node_c"] = post_node_c;
    node_name_list.push_back("post_node_c");

    NodeDef* post_node_d = MallocNodeDef("post_node_d", "Identity");
    EXPECT_NE(post_node_d, nullptr);
    tensorflow_parser->nodedef_map_["post_node_d"] = post_node_d;
    node_name_list.push_back("post_node_d");
  }

  void FreeNodeDefMap(ge::TensorFlowModelParser *tensorflow_parser, set<string> &malloc_node_name_list) {
    for (auto &item : tensorflow_parser->nodedef_map_) {
      if (item.second != nullptr && malloc_node_name_list.count(item.first) > 0) {
        delete (item.second);
        item.second = nullptr;
      }
    }
  }
  void GenFusionScopesResult(shared_ptr<ScopeGraph> &scope_graph, FusionScopesResult *fusion_rlt,
                            const string &fusion_op_name) {
    if (fusion_rlt == nullptr) {
      return;
    }
    fusion_rlt->InsertInputs("scope_node_1", {0});   // scope input 0
    fusion_rlt->InsertOutputs("scope_node_m", {0});  // scope output 0
    fusion_rlt->InsertOutputs("scope_node_n", {1});  // scope output 1

    fusion_rlt->SetType(ge::kScopeToMultiNodes);
    fusion_rlt->SetName(fusion_op_name);
    fusion_rlt->SetDescription("Description for fusion node");

    // Add inner nodes in sequence.
    auto node1 = fusion_rlt->AddInnerNode("inner_node_1", "Unique");  // add inner node1
    CHECK_INNER_NODE_CONDITION(node1 != nullptr, fusion_rlt);
    auto ret = node1
      ->InsertInput(ge::kInputFromFusionScope, 0)   // Input from 0th of boundary (a)
      .InsertOutput(ge::kOutputToFusionScope, 0)                 // Output to 0th of boundary (b)
      .InsertOutput("inner_node_2", 0)  // Output to input 0th of internal node 2
      .BuildInnerNode();                                         // Construct an internal Operator
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    string str_val = "This is a string.";
    node1->MutableOperator()->SetAttr("key1", 2);        // Set integer attribute
    node1->MutableOperator()->SetAttr("key2", str_val);  // Set the string attribute
    node1->MutableOperator()->SetAttr("key3", true);     // Set boolean attribute

    auto node2 = fusion_rlt->AddInnerNode("inner_node_2", "Identity");  // add inner node2
    CHECK_INNER_NODE_CONDITION(node2 != nullptr, fusion_rlt);
    ret = node2
      ->InsertInput("inner_node_1", 1)  // The input comes from the 1st output of internal node 1
      .InsertOutput("inner_node_3", 0)  // Output to input 0th of internal node 3
      .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    node2->SetInputFormat("x", "NHWC");
    node2->SetOutputFormat("y", "NHWC");

    auto node3 = fusion_rlt->AddInnerNode("inner_node_3", "Identity");  // add inner node3
    CHECK_INNER_NODE_CONDITION(node3 != nullptr, fusion_rlt);
    ret = node3
      ->InsertInput("inner_node_2", 0)  // The input comes from the 0th output of internal node 2
      .InsertOutput(ge::kOutputToFusionScope, 1)     // Output to 1st of boundary (c)
      .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    scope_graph->impl_->AddFusionScopesResult(fusion_rlt);
  }

  void GenOriginContext(ge::TensorFlowModelParser *tensorflow_parser, const string &fusion_op_name) {
    // op_node_context for fusion op
    ge::OpNodeContext op_node_context;
    op_node_context.input_map["pre_node_a"].push_back({0, 0});
    op_node_context.input_map["pre_node_ctrl_in"].push_back({-1, -1}); // ctrl edges
    op_node_context.output_map["post_node_b"].push_back({0, 0});
    op_node_context.output_map["post_node_c"].push_back({1, 0});
    op_node_context.output_map["post_node_d"].push_back({-1, -1});  // ctrl edges
    tensorflow_parser->op_node_context_map_[fusion_op_name] = op_node_context;
    tensorflow_parser->SaveEdgesControlInfo(fusion_op_name, -1);

    // op_node_context for pre_node_a
    ge::OpNodeContext op_node_context_a;
    op_node_context_a.output_map[fusion_op_name].push_back({0, 0});
    tensorflow_parser->op_node_context_map_["pre_node_a"] = op_node_context_a;

    // op_node_context for pre_node_ctrl_in
    ge::OpNodeContext op_node_context_ctrl_in;
    op_node_context_ctrl_in.output_map[fusion_op_name].push_back({-1, -1}); // ctrl edges
    tensorflow_parser->op_node_context_map_["pre_node_ctrl_in"] = op_node_context_ctrl_in;

    // op_node_context for post_node_b
    ge::OpNodeContext op_node_context_b;
    op_node_context_b.input_map[fusion_op_name].push_back({0, 0});
    tensorflow_parser->op_node_context_map_["post_node_b"] = op_node_context_b;

    // op_node_context for post_node_c
    ge::OpNodeContext op_node_context_c;
    op_node_context_c.input_map[fusion_op_name].push_back({1, 0});
    op_node_context_c.output_map["post_node_d"].push_back({0, 0});
    tensorflow_parser->op_node_context_map_["post_node_c"] = op_node_context_c;

    // op_node_context for post_node_d
    ge::OpNodeContext op_node_context_d;
    op_node_context_d.input_map["post_node_c"].push_back({0, 0});
    op_node_context_d.input_map[fusion_op_name].push_back({-1, -1}); // ctrl edges
    tensorflow_parser->op_node_context_map_["post_node_d"] = op_node_context_d;
    tensorflow_parser->SaveEdgesControlInfo("post_node_d", -1);

    string fusion_op_type = ge::kScopeToMultiNodes;
    string description = "fusion op description";
    tensorflow_parser->fusion_op_type_map_[fusion_op_name].push_back(fusion_op_type);
    tensorflow_parser->fusion_op_type_map_[fusion_op_name].push_back(description);
  }
  void register_tbe_op()
  {
      std::vector<OpRegistrationData> registrationDatas = OpRegistry::Instance()->registrationDatas;
      for(OpRegistrationData reg_data : registrationDatas)
      {
          OpRegistrationTbe::Instance()->Finalize(reg_data);
          OpRegistry::Instance()->Register(reg_data);
      }
      OpRegistry::Instance()->registrationDatas.clear();
  }
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

static MemBuffer* MemBufferFromFile(const char *path)
{
    char path_temp[PATH_MAX + 1] = {0x00};
    if(strlen(path) > PATH_MAX || nullptr == realpath(path, path_temp)) {
        return nullptr;
    }
    FILE *fp = fopen(path_temp, "r+");
    if (fp == nullptr) {
        return nullptr;
    }

    // get model file length
    if (0 != fseek(fp, 0, SEEK_END)) {
        fclose(fp);
        return nullptr;
    }
    long file_length = ftell(fp);
    if (fseek(fp, 0, SEEK_SET)) {
        fclose(fp);
        return nullptr;
    }
    if (file_length <= 0) {
        fclose(fp);
        return nullptr;
    }

    // alloc model buffer
    void *data = malloc((unsigned int)file_length);
    if (!data) {
        fclose(fp);
        return nullptr;
    }

    // read file into memory
    uint32_t read_size = (uint32_t)fread(data, 1, (unsigned int)file_length, fp);

    // check if read success
    if ((long)read_size != file_length) {
        free(data);
        data = nullptr;
        fclose(fp);
        return nullptr;
    }

    // close model file
    fclose(fp);

    // create an MemBuffer
    MemBuffer* membuf = new MemBuffer();
    if (!membuf) {
        free(data);
        data = nullptr;
        return nullptr;
    }
    membuf->data = malloc((unsigned int)read_size);

    // set size && data
    membuf->size = (uint32_t)read_size;
    memcpy((char*)membuf->data, (char*)data, read_size);
    free(data);
    return membuf;
}


///        placeholder0  placeholder1
///          |       /\  /\       |
///          |      /  \/  \      |
///          |     /   /\   \     |
///          |     |  /  \  |     |
///          |     add0   mul0    |
///          |     /     /c | \   |
///            mul1 --- /   |   add1
///              \          |    |
///               \ ---- add2    |
///                      |       |
///                    retval0 retval1

void CreateGraphDef(domi::tensorflow::GraphDef &graph_def) {
  // 1. add node
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");

  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");

  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");

  add1->add_input("mul0");
  add1->add_input("placeholder1");

  add2->add_input("mul1");
  add2->add_input("mul0");

  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}

TEST_F(STestTensorflowParser, tensorflow_parser_success) {
  RegisterCustomOp();

  std::string case_dir = __FILE__;
  ParserOperator unused("Add");
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/tf_add.pb";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseTensorFlow(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "add_test_1");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "add_test_1:0");
}

TEST_F(STestTensorflowParser, tensorflow_model_Failed) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);

  std::string modelFile = caseDir + "/origin_models/model.pb";
  auto status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::SUCCESS);

  modelFile = caseDir + "/origin_models/test_depth_wise_conv2d.pb";
  status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(STestTensorflowParser, tensorflow_model_not_exist) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);

  // model file is not exist
  std::string modelFile = caseDir + "/origin_models/conv2d_explicit1_pad.pb";
  auto status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(STestTensorflowParser, parser_tensorflow_model) {
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  const char *model_file = modelFile.c_str();
  std::string op_name = "ge_ascend_irgraph";
  ge::Graph graph(op_name);

  std::map<ge::AscendString, ge::AscendString> parser_options = {
    {ge::AscendString(ge::ir_option::INPUT_FORMAT), ge::AscendString("NHWC")},
  };
  auto ret_graph = ge::aclgrphParseTensorFlow(model_file, parser_options, graph);
  EXPECT_EQ(ret_graph, ge::FAILED);

  // parser tensorflow model out_node_size is equal to index
  string graph_name;
  AclGrphParseUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> out_nodes_with_node_and_index = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  ParerSTestsUtils::ClearParserInnerCtx();
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ret_graph = ge::aclgrphParseTensorFlow(model_file, graph);
  EXPECT_EQ(ret_graph, domi::FAILED);

  // parser tensorflow model success
  modelFile = caseDir + "/origin_models/model.pb";
  model_file = modelFile.c_str();
  out_nodes_with_node_and_index = {{AscendString(ge::ir_option::OUT_NODES), AscendString("x:0;y:0")}};
  ParerSTestsUtils::ClearParserInnerCtx();
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ret_graph = ge::aclgrphParseTensorFlow(model_file, graph);
  EXPECT_EQ(ret_graph, domi::SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_parser_to_json)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  std::string jsonFile = caseDir + "/origin_models/test.json";
  const char *model_file = modelFile.c_str();
  const char *json_file = jsonFile.c_str();
  Status ret = modelParser.ToJson(model_file, json_file);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_parserfrommemory_failed)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  const char *data = modelFile.c_str();
  uint32_t size = 1;
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  modelFile = caseDir + "/origin_models/tf_add.pb";
  parser_params = {{AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:0")}};
  ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  ret = modelParser.ParseFromMemory(data, size, compute_graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(STestTensorflowParser, modelparser_parsefrommemory_success)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  const char* tmp_tf_pb_model = modelFile.c_str();
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  TensorFlowModelParser modelParser;
  MemBuffer* memBuffer = MemBufferFromFile(tmp_tf_pb_model);
  PreChecker::Instance().HasError() == false;
  ret = modelParser.ParseFromMemory((char*)memBuffer->data, memBuffer->size, compute_graph);
  free(memBuffer->data);
  delete memBuffer;
}

TEST_F(STestTensorflowParser, weightsparser_parsefrommemory_success)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  const char* tmp_tf_pb_model = modelFile.c_str();
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  auto weights_parser = domi::WeightsParserFactory::Instance()->CreateWeightsParser(domi::TENSORFLOW);
  MemBuffer* memBuffer = MemBufferFromFile(tmp_tf_pb_model);
  ret = weights_parser->ParseFromMemory((char*)memBuffer->data, memBuffer->size, compute_graph);
  free(memBuffer->data);
  delete memBuffer;
  EXPECT_EQ(SUCCESS, ret);
}

std::string getGraphCallbackV2(string subgraph_name)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  subgraph_name = caseDir + "/origin_models/tf_add.pb";
  return subgraph_name;
}

TEST_F(STestTensorflowParser, parser_ParseProtoWithSubgraphV2)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/origin_models/tf_add.pb";
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(root_proto.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr root_graph = ge::GraphUtils::GetComputeGraph(graph);
  domi::GetGraphCallbackV2 callback(&getGraphCallbackV2);
  TensorFlowModelParser parser;
  ret = parser.ParseProtoWithSubgraph(root_proto, callback, root_graph);
}

TEST_F(STestTensorflowParser, parser_ConvertToGeDataType)
{
  // convert to ge type success
  const uint32_t type1 = domi::tensorflow::DataType::DT_FLOAT;
  TensorFlowModelParser parser;
  ge::DataType dataType = parser.ConvertToGeDataType(type1);
  ASSERT_EQ(dataType, ge::DataType::DT_FLOAT);

  const uint32_t type2 = 80; // invalid type
  dataType = parser.ConvertToGeDataType(type2);
  ASSERT_EQ(dataType, ge::DataType::DT_UNDEFINED);
}

TEST_F(STestTensorflowParser, tensorflow_ParserProto_failed)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/origin_models/avgpool3dgrad.pb.txt";
  domi::tensorflow::GraphDef graphDef;
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(root_proto.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr root_graph = ge::GraphUtils::GetComputeGraph(graph);
  TensorFlowModelParser tensorflow_parser;
  ret = tensorflow_parser.ParseProto(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  EXPECT_EQ(PARAM_INVALID, ret);

  // proto解析失败
  bool protoRet = parser::ReadProtoFromText(root_proto.c_str(), &graphDef);
  ASSERT_EQ(protoRet, false);
  ret = tensorflow_parser.ParseProto(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  ASSERT_EQ(ret, PARAM_INVALID);
}

TEST_F(STestTensorflowParser, tensorflow_parserAllGraph_failed)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/origin_models/conv2d.pb";
  domi::tensorflow::GraphDef graphDef;
  CreateGraphDef(graphDef);

  auto no_op = graphDef.add_node();
  no_op->set_name("no_op");
  no_op->set_op("NoOp");
  no_op->add_input("placeholder0");
  no_op->add_input("placeholder1");

  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(root_proto.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr root_graph = ge::GraphUtils::GetComputeGraph(graph);
  TensorFlowModelParser tensorflow_parser;
  ret = tensorflow_parser.ParseAllGraph(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  EXPECT_EQ(INTERNAL_ERROR, ret);
}

TEST_F(STestTensorflowParser, test_parse_acl_output_nodes)
{
  AclGrphParseUtil acl_graph_parse_util;
  string graph_name;
  // case 1: Normal with 'node and index'
  ParerSTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_with_node_and_index = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out1:0;Out2:1")}};
  ParerSTestsUtils::ClearParserInnerCtx();
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 2);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 2);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 0);

  // case 2: Normal with 'tensor name'
  ParerSTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_with_tensor_name = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_tensor_name, graph_name);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 2);

  // case 3: Failed with 'node and index' before 'tensor name'
  ParerSTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_pre = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out1:0;Out2:1;Out_tensor_1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_pre, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 2);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 2);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 0);

  // case 4: Failed with 'node and index' inserted in 'tensor name'
  ParerSTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_mid = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out1:0;Out2:1;Out_tensor_2")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_mid, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 1);

  // case 5: Failed with 'node and index' after 'tensor name'
  ParerSTestsUtils::ClearParserInnerCtx();
  GetParserContext().type = domi::ONNX;
  std::map<AscendString, AscendString> out_nodes_mode_mixex_post = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Out_tensor_1;Out_tensor_2;Out1:0;Out2:1")}};
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_mode_mixex_post, graph_name);
  ASSERT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(ge::GetParserContext().user_out_nodes.size(), 0);
  EXPECT_EQ(ge::GetParserContext().out_nodes_map.size(), 0);
  EXPECT_EQ(ge::GetParserContext().user_out_tensors.size(), 2);
}

TEST_F(STestTensorflowParser, parse_AutoMappingByOp) {
  static const string KEY_STRING = "key_string";
  static const string KEY_INT = "key_int";
  static const string KEY_FLOAT = "key_float";
  static const string KEY_BOOL = "key_bool";
  static const string KEY_TYPE = "key_type";
  static const string VALUE_STRING = "string";
  static const int64_t VALUE_INT = 1;
  static const float VALUE_FLOAT = 1.0;
  static const bool VALUE_BOOL = true;
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;

  std::cout << "test data_type value_type: " << (int64_t)VALUE_TYPE << std::endl;
  static const string VALUE_NAME = "test_name";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  NodeDef node_def;
  domi::tensorflow::AttrValue value;
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  node_def.set_name(VALUE_NAME);
  value.set_s(VALUE_STRING);
  TensorFlowUtil::AddNodeAttr(KEY_STRING, value, &node_def);
  value.set_i(VALUE_INT);
  TensorFlowUtil::AddNodeAttr(KEY_INT, value, &node_def);
  value.set_f(VALUE_FLOAT);
  TensorFlowUtil::AddNodeAttr(KEY_FLOAT, value, &node_def);
  value.set_b(VALUE_BOOL);
  TensorFlowUtil::AddNodeAttr(KEY_BOOL, value, &node_def);
  value.set_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE, value, &node_def);

  domi::Status status = domi::AutoMappingFn(reinterpret_cast<google::protobuf::Message *>(&node_def), op);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(VALUE_NAME, op_desc->GetName());

  string value_string = "";
  ge::AttrUtils::GetStr(op_desc, KEY_STRING, value_string);
  EXPECT_EQ(VALUE_STRING, value_string);

  int64_t value_int = 0;
  ge::AttrUtils::GetInt(op_desc, KEY_INT, value_int);
  EXPECT_EQ(VALUE_INT, value_int);

  float value_float = 0.0;
  ge::AttrUtils::GetFloat(op_desc, KEY_FLOAT, value_float);
  EXPECT_EQ(VALUE_FLOAT, value_float);

  bool value_bool = false;
  ge::AttrUtils::GetBool(op_desc, KEY_BOOL, value_bool);
  EXPECT_EQ(VALUE_BOOL, value_bool);

  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::AttrUtils::GetDataType(op_desc, KEY_TYPE, data_type);
  EXPECT_EQ(ge::DT_FLOAT, data_type);

  // test AutoMappingByOpFn
  ge::OpDescPtr op_desc_dest = std::make_shared<ge::OpDesc>();
  ge::Operator op_dest = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_dest);

  status = domi::AutoMappingByOpFn(op, op_dest);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(VALUE_NAME, op_dest.GetName());

  value_string = "";
  ge::AttrUtils::GetStr(op_desc_dest, KEY_STRING, value_string);
  EXPECT_EQ(VALUE_STRING, value_string);

  value_int = 0;
  ge::AttrUtils::GetInt(op_desc_dest, KEY_INT, value_int);
  EXPECT_EQ(VALUE_INT, value_int);

  value_float = 0.0;
  ge::AttrUtils::GetFloat(op_desc_dest, KEY_FLOAT, value_float);
  EXPECT_EQ(VALUE_FLOAT, value_float);

  value_bool = false;
  ge::AttrUtils::GetBool(op_desc_dest, KEY_BOOL, value_bool);
  EXPECT_EQ(VALUE_BOOL, value_bool);

  data_type = ge::DT_UNDEFINED;
  ge::AttrUtils::GetDataType(op_desc_dest, KEY_TYPE, data_type);
  EXPECT_EQ(ge::DT_FLOAT, data_type);
}

TEST_F(STestTensorflowParser, parse_ParseNodeDef)
{
  NodeDef * node_def = new NodeDef();
  node_def->set_name("test_name");
  node_def->set_op("PlaceholderWithDefault");

  bool isDatasetInit = true;
  TensorFlowModelParser model_parser;
  Status ret = model_parser.AdaptOpType(node_def, isDatasetInit);
  EXPECT_EQ(domi::SUCCESS, ret);

  node_def->set_op("Add");
  ret = model_parser.AdaptOpType(node_def, isDatasetInit);
  EXPECT_EQ(domi::SUCCESS, ret);
  delete node_def;
}

TEST_F(STestTensorflowParser, parse_AddFmkNode)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  ge::Graph graph;
  string graph_name;
  AclGrphParseUtil acl_graph_parse_util;
  std::map<ge::AscendString, ge::AscendString> parser_options = {{AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:0")}};
  ParerSTestsUtils::ClearParserInnerCtx();
  Status ret = acl_graph_parse_util.ParseParamsBeforeGraph(parser_options, graph_name);
  ret = aclgrphParseTensorFlow(modelFile.c_str(), parser_options, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  tensorflow::GraphDef *graphDef = new (std::nothrow) tensorflow::GraphDef();
  ScopePassManager pass_manager;
  std::shared_ptr<ScopeGraph> scope_graph = pass_manager.BuildScopeGraph(graphDef);

  std::string fusion_op_name = "fusion_op_name";
  FusionScopesResult *fusion_rlt = new (std::nothrow) FusionScopesResult();
  EXPECT_NE(fusion_rlt, nullptr);
  fusion_rlt->Init();
  GenFusionScopesResult(scope_graph, fusion_rlt, fusion_op_name);
  GenOriginContext(&modelParser, fusion_op_name);

  // origin inner node def
  NodeDef* node_def = MallocNodeDef("scope_node_1", "Add");
  EXPECT_NE(node_def, nullptr);
  modelParser.fusion_op_nodedef_map_[fusion_op_name].push_back(node_def);

  bool train_flag_backup = ge::GetParserContext().train_flag;
  ge::GetParserContext().train_flag = true;

  REGISTER_CUSTOM_OP("Identity")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("Identity")
    .ParseParamsFn(ParseParams)
    .ImplyType(ImplyType::TVM);
  REGISTER_CUSTOM_OP("Constant")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("Const")
    .ParseParamsFn(ParseParams)
    .ImplyType(ImplyType::TVM);

  register_tbe_op();

  std::vector<std::string> node_name_list;
  GenOriginNodeDef(&modelParser, node_name_list);
  std::set<std::string> malloc_node_name_list(node_name_list.begin(), node_name_list.end());
  node_name_list.push_back(fusion_op_name);

  ret = modelParser.AddFmkNode(compute_graph, scope_graph, node_name_list, false);
  EXPECT_EQ(ret, PARAM_INVALID);
  EXPECT_EQ(modelParser.scope_inner_node_map_.size(), 0);
  EXPECT_EQ(modelParser.nodedef_map_.size(), 5);

  ret = modelParser.AddEdges(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  // release resource
  delete graphDef;
  delete node_def;
  modelParser.DeleteFuisonNodeDef();
  FreeNodeDefMap(&modelParser, malloc_node_name_list);
  ge::GetParserContext().train_flag = train_flag_backup;
}

TEST_F(STestTensorflowParser, parse_AddScopeInnerNode)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  std::string op_name = "ge_ascend_irgraph";
  ge::Graph graph(op_name);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  std::map<ge::AscendString, ge::AscendString> parser_params = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:0")}};
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(ret, SUCCESS);

  std::mutex graph_mutex;
  tensorflow::NodeDef *node_def = new NodeDef();
  node_def->set_name("FastrcnnPredictions");
  node_def->set_op("FastrcnnPredictions");
  // can't find in scope_inner_node_map
  ret = modelParser.AddScopeInnerNode(&modelParser, compute_graph, &graph_mutex, node_def);
  EXPECT_EQ(ret, PARAM_INVALID);
  delete node_def;
}

TEST_F(STestTensorflowParser, dyncmic_rnn_scope_pass_plugin_test) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tensor_array.pb";
  std::map<ge::AscendString, ge::AscendString> params;
  string key ="enable_scope_fusion_passes";
  string value ="ScopeDynamicRNNPass";
  params.insert(std::make_pair(ge::AscendString(key.c_str()), ge::AscendString(value.c_str())));
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), params, graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, avgpool3dgrad_plugin_test_format_NDHWC) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/avgpool3dgrad_case_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_merge_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/merge.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}

TEST_F(STestTensorflowParser, tensorflow_no_op_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_no_op.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_identity_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_identity.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_constant_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_constant.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);

  TensorFlowConstantParser constantParser;
  ge::OpDescPtr op_dest = make_shared<ge::OpDesc>("constant", ge::parser::CONSTANT);
  NodeDef* node_def = initNodeDef();
  node_def->set_name("Constant");
  auto params = constantParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(params, SUCCESS);

  auto value = constantParser.ParseValue(node_def, op_dest);
  EXPECT_EQ(value, SUCCESS);

  ConstantOperator op;
  auto type = constantParser.ParseDType(node_def, &op);
  EXPECT_EQ(type, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_reshpae_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_reshape.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_squeeze_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_sequeeze.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_fill_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_fill.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_shape_n_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_shape_n.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_switch_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_switch.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);

  TensorFlowRefSwitchParser refSwitchParser;
  ge::OpDescPtr op_dest = make_shared<ge::OpDesc>("constant", ge::parser::CONSTANT);
  NodeDef* node_def = initNodeDef();
  node_def->set_name("RefSwitch");
  auto params = refSwitchParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(params, SUCCESS);

  RefSwitchOperator op;
  auto parseRet = refSwitchParser.ParseT(node_def, &op);
  EXPECT_EQ(parseRet, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_enter_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_enter.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_VariableV2_test) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_VariableV2.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

} // namespace ge
