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
#include "framework/omg/parser/parser_factory.h"
#include "graph/operator_reg.h"
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "external/register/register.h"
#include "tests/depends/ops_stub/ops_stub.h"
#include "parser/common/acl_graph_parser_util.h"
#include "external/ge/ge_api_types.h"
#include "omg/parser/parser_factory.h"
#include "common/pre_checker.h"
#include "common/util.h"
#include "external/parser/tensorflow_parser.h"
#include "ut/parser/parser_ut_utils.h"
#include "graph/model.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "tests/depends/ops_stub/ops_stub.h"
#include "parser/tensorflow/tensorflow_constant_parser.h"
#include "common/types.h"
#include "parser/common/op_def/variable_operator.h"
#include "parser/tensorflow/tensorflow_ref_switch_parser.h"
#include "parser/tensorflow/tensorflow_fusion_op_parser.h"
#include "parser/tensorflow/tensorflow_auto_mapping_parser_adapter.h"
#include "parser/common/op_def/arg_op_operator.h"
#include "parser/tensorflow/tensorflow_fusion_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_reshape_parser.h"
#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_squeeze_parser.h"
#include "parser/tensorflow/graph_to_function_def.h"
#include "parser/tensorflow/parser_graph_optimizer.h"
#include "cce/dnn_base_def.hpp"
#include "parser/tensorflow/scope/scope_pass_manager.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "compute_graph_impl.h"
#include "parser/tensorflow/tensorflow_enter_parser.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/tuple.h"
#include "common/op_def/framework_op_operator.h"
#include "common/op_def/shape_n_operator.h"
#include "common/op_def/var_is_initialized_op_operator.h"
#include "common/op_def/fill_operator.h"
#include "common/convert/pb2json.h"
#include "common/convert/message2operator.h"
#include "parser/common/proto_file_parser.h"
#include "parser/common/pre_checker.h"
#include "parser/common/tbe_plugin_loader.h"
#include "parser/common/data_op_parser.h"
#include "parser/common/model_saver.h"
#include "framework/omg/parser/parser_api.h"
#include "parser/common/parser_fp16_t.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/prototype_pass_manager.h"
#include "parser/common/op_registration_tbe.h"
#include "parser/common/pass_manager.h"
#include "parser/tensorflow/parser_graph_optimizer.h"
#include "metadef/inc/register/scope/scope_pass_registry_impl.h"
#include "register/scope/scope_fusion_pass_register.h"
#include "common/op_map.h"
#undef protected
#undef private

using namespace std;
using namespace domi::tensorflow;
using namespace domi;
using namespace cce;
using namespace testing;
using namespace std;
using namespace google::protobuf;

static const string GRAPH_DEFAULT_NAME = "default";

namespace ge {
struct DelTransposeInfo {
  domi::tensorflow::NodeDef *node_def;     // transpose
  domi::tensorflow::NodeDef *nextNodeDef;  // transpose --> [next]
  int inputIdx;
};

/*
  message=1,LayerParameter=1
  optional =1 repeated =1 required =1
  */

Status GetTransposeInfo(GraphDef *graph_def, std::map<std::string, std::string> &softmaxInfo,
                        std::map<std::string, DelTransposeInfo> &transposeInfo);

Status EraseTransposeNode(std::map<std::string, std::string> &softmaxInfo,
                          std::map<std::string, DelTransposeInfo> &transposeInfo);

Status ComputeArgRange(const domi::tensorflow::NodeDef &node_def, const domi::tensorflow::OpDef::ArgDef &arg_def,
                       int *num);

class UtestTensorflowParser : public testing::Test {
 protected:
  void SetUp() {
    ParerUTestsUtils::ClearParserInnerCtx();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
};

class TestOperator : public ParserOperator
{
public:
    TestOperator()
      : ParserOperator("test")
    {
    }

    ~TestOperator()
    {
    }
};

class ErrorGraphPass: public GraphPass
{
    Status Run(ComputeGraphPtr graph)
    {
        return domi::FAILED;
    }
};

class ScopeTestPass : public ScopeBasePass {
 protected:
  vector<ScopeFusionPatterns> DefinePatterns() {
    vector<ScopeFusionPatterns> patterns_list;
    return patterns_list;
  };
  string PassName() {
    return "test";
  };
  Status LastMatchScopesAndOPs(shared_ptr<ScopeGraph> &scope_graph, vector<ScopesResult> &results) {
    return domi::SUCCESS;
  };
  void GenerateFusionResult(const vector<Scope *> &scopes, FusionScopesResult *fusion_rlt) {
    return;
  };
};

static Status ParseParams(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

static Status ParseParamByOpFunc(const ge::Operator &op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

void AddDumpOriginName(const ge::NodePtr parent_node, const std::string& subgraph_name, ge::ComputeGraphPtr graph);

void UtestTensorflowParser::RegisterCustomOp() {
  REGISTER_CUSTOM_OP("Add")
  .FrameworkType(domi::TENSORFLOW)
  .OriginOpType("Add")
  .ParseParamsFn(ParseParams);
  std::vector<OpRegistrationData> reg_datas = domi::OpRegistry::Instance()->registrationDatas;
  for (auto reg_data : reg_datas) {
    domi::OpRegTbeParserFactory::Instance()->Finalize(reg_data);
    domi::OpRegistry::Instance()->Register(reg_data);
  }
  domi::OpRegistry::Instance()->registrationDatas.clear();
}

namespace {
  NodeDef* AddNode(GraphDef& graph, string type, string name) {
    NodeDef* nodeDef = graph.add_node();
    nodeDef->set_op(type);
    nodeDef->set_name(name);

    tensorflow::OpDef op_def;
    string op_def_string;
    op_def.SerializeToString(&op_def_string);

    tensorflow::AttrValue value;
    value.set_s(op_def_string);
    nodeDef->mutable_attr()->insert({"op_def", value});
    return nodeDef;
  }

  void AddInput(NodeDef* src, NodeDef* dst, int srcIndex) {
    if(srcIndex == -1){
        dst->add_input("^"+src->name());
    } else {
      if (srcIndex == 0) {
        dst->add_input(src->name());
      } else {
        dst->add_input(src->name() + ":" + std::to_string(srcIndex));
      }
      {
        auto input = (*dst->mutable_attr())[ge::ATTR_NAME_INPUT_TENSOR_DESC].mutable_list()->add_func();
        tensorflow::AttrValue val1;
        val1.set_i(0);
        (*input->mutable_attr())["serialize_format"] = val1;
        tensorflow::AttrValue val2;
        val2.set_i(tensorflow::DT_FLOAT);
        (*input->mutable_attr())["serialize_datatype"] = val2;
        tensorflow::AttrValue val3;
        val3.mutable_list()->add_i(10);
        (*input->mutable_attr())["serialize_shape"] = val3;
      }

      {
        auto output = (*src->mutable_attr())[ge::ATTR_NAME_OUTPUT_TENSOR_DESC].mutable_list()->add_func();
        tensorflow::AttrValue val1;
        val1.set_i(0);
        (*output->mutable_attr())["serialize_format"] = val1;
        tensorflow::AttrValue val2;
        val2.set_i(tensorflow::DT_FLOAT);
        (*output->mutable_attr())["serialize_datatype"] = val2;
        tensorflow::AttrValue val3;
        val3.mutable_list()->add_i(10);
        (*output->mutable_attr())["serialize_shape"] = val3;
      }
    }
  }

  NodeDef *initNodeDef() {
    NodeDef * nodeDef = new NodeDef();
    nodeDef->set_op("Const");
    ::google::protobuf::Map<std::string, tensorflow::AttrValue >* node_attr_map = nodeDef->mutable_attr();

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

    // ����?? tensor ��?D?
    domi::tensorflow::AttrValue value_attr_value;
    tensorflow::TensorProto* tensor = value_attr_value.mutable_tensor();
    tensorflow::TensorShapeProto* tensor_shape = tensor->mutable_tensor_shape();
    tensor_shape->clear_dim();
    tensor_shape->add_dim()->set_size(4);
    tensor_shape->add_dim()->set_size(6);
    tensor->set_dtype(domi::tensorflow::DT_INT32);

    float *addr = new float[24];
    for (int32_t i = 0; i < 24; i++) {
        *(addr + i) = 1.0 + i;
    }
    tensor->set_tensor_content((void *)addr, 24 * sizeof(float));

    (*node_attr_map)[TENSORFLOW_ATTR_VALUE] = value_attr_value;
    delete[] addr;
    return nodeDef;
  }

  NodeDef * initOpNodeDef_VariableV2() {
    NodeDef * nodeDef = new NodeDef();
    nodeDef->set_op("VariableV2");
    google::protobuf::Map<std::string, tensorflow::AttrValue > *node_attr_map = nodeDef->mutable_attr();

    domi::tensorflow::AttrValue format_attr_value;
    format_attr_value.set_s("_FZ");
    (*node_attr_map)[VAR_ATTR_FORMAT] = format_attr_value;

    domi::tensorflow::AttrValue type_attr;
    type_attr.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[VAR_ATTR_DTYPE] = type_attr;

    domi::tensorflow::AttrValue container_attr_value;
    container_attr_value.set_s("container");
    (*node_attr_map)[VAR_ATTR_CONTAINER] = container_attr_value;

    domi::tensorflow::AttrValue shard_name_attr_value;
    shard_name_attr_value.set_s("shard_name");
    (*node_attr_map)[VAR_ATTR_SHARED_NAME] = shard_name_attr_value;

    domi::tensorflow::AttrValue shape_attr_value;
    shape_attr_value.mutable_shape()->add_dim()->set_size(1);
    shape_attr_value.mutable_shape()->add_dim()->set_size(2);
    shape_attr_value.mutable_shape()->add_dim()->set_size(3);
    shape_attr_value.mutable_shape()->add_dim()->set_size(4);
    (*node_attr_map)[ge::VAR_ATTR_SHAPE] = shape_attr_value;

    domi::tensorflow::AttrValue shape;
    shape.mutable_list()->add_i((int64)32);
    shape.mutable_list()->add_i((int64)32);
    shape.mutable_list()->add_i((int64)14);
    shape.mutable_list()->add_i((int64)14);

    domi::tensorflow::AttrValue df_attr_value;
    domi::tensorflow::AttrValue df_attr_value2;
    df_attr_value2.set_s(TENSORFLOWF_TENSOR_NHWC);

    df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);
    (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value2;

    domi::tensorflow::AttrValue pad_attr_value;
    domi::tensorflow::AttrValue pad_attr_value2;
    pad_attr_value2.set_s(TENSORFLOWF_OP_PADDING_SAME);
    (*node_attr_map)[TENSORFLOW_ATTR_PADDING] = pad_attr_value2;
    pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

    domi::tensorflow::NameAttrList name_attr_list;
    name_attr_list.set_name(std::to_string(0));
    name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
    name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
    name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
    domi::tensorflow::AttrValue output_tensor_descs;
    *(output_tensor_descs.mutable_list()->add_func()) = name_attr_list;
    nodeDef->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_tensor_descs});
    return nodeDef;
  }

  NodeDef *initOpNodeDef_TemporaryVariable() {
    NodeDef * nodeDef = new NodeDef();
    nodeDef->set_op("TemporaryVariable");
    google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = nodeDef->mutable_attr();

    domi::tensorflow::AttrValue type_attr;
    type_attr.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[VAR_ATTR_DTYPE] = type_attr;

    domi::tensorflow::AttrValue var_name_attr_value;
    var_name_attr_value.set_s("temporary_variable_name");
    (*node_attr_map)[ge::VAR_ATTR_NAME] = var_name_attr_value;

    domi::tensorflow::AttrValue shape_attr_value;
    shape_attr_value.mutable_shape()->add_dim()->set_size(1);
    shape_attr_value.mutable_shape()->add_dim()->set_size(2);
    shape_attr_value.mutable_shape()->add_dim()->set_size(3);
    shape_attr_value.mutable_shape()->add_dim()->set_size(4);
    (*node_attr_map)[ge::VAR_ATTR_SHAPE] = shape_attr_value;

    domi::tensorflow::AttrValue shape;
    shape.mutable_list()->add_i((int64)32);
    shape.mutable_list()->add_i((int64)32);
    shape.mutable_list()->add_i((int64)14);
    shape.mutable_list()->add_i((int64)14);

    domi::tensorflow::AttrValue df_attr_value2;
    df_attr_value2.set_s(TENSORFLOWF_TENSOR_NHWC);
    (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value2;
    domi::tensorflow::AttrValue df_attr_value;
    df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

    domi::tensorflow::AttrValue pad_attr_value2;
    pad_attr_value2.set_s(TENSORFLOWF_OP_PADDING_SAME);
    (*node_attr_map)[TENSORFLOW_ATTR_PADDING] = pad_attr_value2;
    domi::tensorflow::AttrValue pad_attr_value;
    pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

    domi::tensorflow::NameAttrList name_attr_list;
    name_attr_list.set_name(std::to_string(0));
    name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
    name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
    name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
    domi::tensorflow::AttrValue output_tensor_descs;
    *(output_tensor_descs.mutable_list()->add_func()) = name_attr_list;
    nodeDef->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_tensor_descs});
    return nodeDef;
  }

  NodeDef *fusioninitNodeDef(int index) {
    NodeDef *nodeDef = new NodeDef();
    google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = nodeDef->mutable_attr();

    domi::tensorflow::AttrValue dtype_attr_value ;

    if (index == 0) {
        dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
    } else if (index == 1) {
        dtype_attr_value.set_type(domi::tensorflow::DT_INT32);
    } else if (index == 2) {
        dtype_attr_value.set_type(tensorflow::DT_HALF);
    }
    (*node_attr_map)[ge::TENSORFLOW_ATTR_DTYPE] = dtype_attr_value;

    domi::tensorflow::AttrValue df_attr_value;
    df_attr_value.set_s(TENSORFLOWF_TENSOR_NCHW);
    (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value;

    domi::tensorflow::AttrValue value_attr_value;
    ::tensorflow::TensorProto* tensor = value_attr_value.mutable_tensor();
    ::tensorflow::TensorShapeProto* tensor_shape = tensor->mutable_tensor_shape();
    tensor_shape->clear_dim();
    ::tensorflow::TensorShapeProto_Dim* dim = tensor_shape->add_dim();
    dim->set_name("tensor dim");
    dim->set_size(1);

    if (index == 0) {
        tensor->set_dtype(domi::tensorflow::DT_FLOAT);
        float *addr = new float[1];
        *addr = 1.0;
        tensor->set_tensor_content((void *)addr, sizeof(float));
        (*node_attr_map)[TENSORFLOW_ATTR_VALUE] = value_attr_value;
        delete[] addr;
    } else if (index == 1) {
        tensor->set_dtype(domi::tensorflow::DT_INT32);
        int32_t *addr = new int32_t[1];
        *addr = 1;
        tensor->set_tensor_content((void *)addr, sizeof(int32_t));
        (*node_attr_map)[TENSORFLOW_ATTR_VALUE] = value_attr_value;
        delete[] addr;
    } else if (index == 2) {
        tensor->set_dtype(tensorflow::DT_HALF);
        tensor->add_half_val(1);
        (*node_attr_map)[TENSORFLOW_ATTR_VALUE] = value_attr_value;
    }
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
      google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >* node_attr_map = pre_node_a->mutable_attr();
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
    op_node_context.output_map["post_node_d"].push_back({-1, -1});
    op_node_context.output_map["_Retval"].push_back({0, 1});
    // ctrl edges
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
    op_node_context_c.output_map["post_node_d"].push_back({0, 0});
    tensorflow_parser->op_node_context_map_["post_node_c"] = op_node_context_c;

    // op_node_context for post_node_d
    ge::OpNodeContext op_node_context_d;
    op_node_context_d.input_map[fusion_op_name].push_back({-1, -1}); // ctrl edges
    tensorflow_parser->op_node_context_map_["post_node_d"] = op_node_context_d;

    // op_node_context for Retval
    ge::OpNodeContext op_node_context_Retval;
    op_node_context_d.input_map["post_node_d"].push_back({-1, -1});
    op_node_context_c.output_map["fusion_op_name"].push_back({0,1});
    tensorflow_parser->op_node_context_map_["_Retval"] = op_node_context_Retval;
    tensorflow_parser->SaveEdgesControlInfo("op_node_context_Retval", -1);

    string fusion_op_type = ge::kScopeToMultiNodes;
    string description = "fusion op description";
    tensorflow_parser->fusion_op_type_map_[fusion_op_name].push_back(fusion_op_type);
    tensorflow_parser->fusion_op_type_map_[fusion_op_name].push_back(description);
  }

  void register_tbe_op() {
    std::vector<OpRegistrationData> registrationDatas = OpRegistry::Instance()->registrationDatas;
    for (OpRegistrationData reg_data : registrationDatas) {
      domi::OpRegTbeParserFactory::Instance()->Finalize(reg_data);
      OpRegistry::Instance()->Register(reg_data);
    }
    OpRegistry::Instance()->registrationDatas.clear();
  }

  NodeDef *initNodeDef_axis_dims() {
    NodeDef *nodeDef = new NodeDef();
    google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = nodeDef->mutable_attr();

    domi::tensorflow::AttrValue dtype_attr_value ;
    dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

    //����??strides��?D?
    domi::tensorflow::AttrValue axis_attr_value;
    ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
    list->add_i(1);
    list->add_i(2);
    (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;
    (*node_attr_map)[ge::SQUEEZE_ATTR_DIMS] = axis_attr_value;

    return nodeDef;
  }

  NodeDef *initNodeDef_dims() {
    NodeDef *nodeDef = new NodeDef();
    ::google::protobuf::Map<std::string, tensorflow::AttrValue > *node_attr_map = nodeDef->mutable_attr();

    //����??T��?D?
    domi::tensorflow::AttrValue dtype_attr_value ;
    dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

    //����??strides��?D?
    domi::tensorflow::AttrValue axis_attr_value;
    ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
    list->add_i(1);
    list->add_i(2);
    (*node_attr_map)[ge::SQUEEZE_ATTR_DIMS] = axis_attr_value;
    return nodeDef;
  }

  void CreateOpDef(const string& _name, const string& _type, ge::OpDescPtr opDef) {
    tensorflow::OpDef tsOpDef;
    tsOpDef.set_name(_name);
    tensorflow::OpDef_ArgDef* outArgDef = tsOpDef.add_output_arg();
    outArgDef->set_name(_name);
    outArgDef->set_description("outArgDef");
    outArgDef->set_type((tensorflow::DataType)3);

    if ((_name == "A") || (_name == "B")) {
      tensorflow::OpDef_ArgDef* argDef1 = tsOpDef.add_output_arg();
      string name = _name+"t";
      argDef1->set_name(name);
      argDef1->set_description("this is a test 2");
      argDef1->set_type((tensorflow::DataType)3);
    }
    if ((_name == "C") ) {
      outArgDef->set_number_attr("num");
    }
    if ((_name == "D") ) {
      outArgDef->set_type_list_attr("type_list");
    }

    string strTsOpDef;
    tsOpDef.SerializeToString(&strTsOpDef);
    ge::AttrUtils::SetStr(opDef, "op_def", strTsOpDef);

    tensorflow::NodeDef nodedef;
    nodedef.set_name(_name);
    nodedef.set_op(_name);

    string name("op_def");
    tensorflow::AttrValue value;
    value.set_s(strTsOpDef);
    TensorFlowUtil::AddNodeAttr(name, value, &nodedef);
    value.set_i(1);
    TensorFlowUtil::AddNodeAttr("num", value, &nodedef);
    value.mutable_list();
    TensorFlowUtil::AddNodeAttr("type_list", value, &nodedef);

    string strNodeDef;
    nodedef.SerializeToString(&strNodeDef);
    ge::GeAttrValue::BYTES nodedefBytes;
    nodedefBytes = ge::GeAttrValue::BYTES::CopyFrom((uint8_t*)strNodeDef.data(), strNodeDef.length());
    ge::AttrUtils::SetBytes(opDef, "node_def", nodedefBytes);

    if ((_name== "S") || (_name == "K")) {
      int index = 0;
      ge::AttrUtils::SetInt(opDef, "T", 1);
      ge::AttrUtils::SetInt(opDef, "arg_index", index);
      ge::AttrUtils::SetInt(opDef, "ret_index", index);
    }
  }

  ge::NodePtr AddNode(ge::ComputeGraphPtr graph, const string& _name, const string& _type,int32_t i_n, int32_t o_n) {
    ge::OpDescPtr opDef = std::make_shared<ge::OpDesc>();
    opDef->SetName(_name);
    ge::OpDescUtilsEx::SetType(opDef, _type);
    for(int32_t i = 0; i < i_n; i++) {
      ge::GeTensorDesc input;
      input.SetDataType((ge::DataType)1);
      opDef->AddInputDesc(input);
    }

    for(int32_t i = 0;i < o_n; i++) {
      ge::GeTensorDesc output;
      output.SetDataType((ge::DataType)1);
      opDef->AddOutputDesc(output);
    }
    CreateOpDef(_name, _type, opDef);
    return graph->AddNode(opDef);
  }

  void MakeDagGraph(ge::ComputeGraphPtr graph, const string& input_node_type) {
    ge::NodePtr node_s = AddNode(graph, "S", parser::DATA,1,1);
    ge::NodePtr node_a = AddNode(graph, "A", "testa",1,2);
    ge::NodePtr node_b = AddNode(graph, "B", "testb",1,2);
    ge::NodePtr node_c = AddNode(graph, "C", "testc",1,1);
    ge::NodePtr node_d = AddNode(graph, "D", "testd",1,1);
    ge::NodePtr node_e = AddNode(graph, "E", "teste",1,1);
    ge::NodePtr node_f = AddNode(graph, "F", "testf",1,1);
    ge::NodePtr node_g = AddNode(graph, "G", "testg",2,1);
    ge::NodePtr node_h = AddNode(graph, "H", "testh",1,1);
    ge::NodePtr node_i = AddNode(graph, "I", "testi",1,1);
    ge::NodePtr node_j = AddNode(graph, "J", "testj",2,1);
    ge::NodePtr node_k = AddNode(graph, "K", parser::NETOUTPUT,1,1);

    ge::GraphUtils::AddEdge(node_s->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(1), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_g->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_j->GetOutDataAnchor(0), node_k->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutControlAnchor(), node_j->GetInControlAnchor());
  }

  void ChangeDataType(tensorflow::NodeDef* node_tf, int32_t data_type)
  {
    domi::tensorflow::AttrValue input_attr_value;
    google::protobuf::Map<std::string, tensorflow::AttrValue>* attr = node_tf->mutable_attr();
    google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr->find(ge::ATTR_NAME_INPUT_TENSOR_DESC);
    if (it != attr->end()) {
        input_attr_value = it->second;
    }
    (*attr)[ge::ATTR_NAME_INPUT_TENSOR_DESC] = input_attr_value;
  }

  NodeDef* AddGraphNode(GraphDef *graph, string name, string optype, string input) 
  {
    NodeDef * node_def = graph->add_node();
    node_def->set_name(name);
    node_def->set_op(optype);
    node_def->add_input(input);
    return node_def;
  }

ge::ComputeGraphPtr build_graph(bool with_leaf_node = false)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr data_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(data_op, parser::DATA);
  data_op->SetName("Data1");
  data_op->AddInputDesc(ge::GeTensorDesc());
  data_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr data1 = graph->AddNode(data_op);

  ge::OpDescPtr relu_op1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(relu_op1, parser::ACTIVATION);
  relu_op1->SetName("Relu1");
  relu_op1->AddInputDesc(ge::GeTensorDesc());
  relu_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr relu1 = graph->AddNode(relu_op1);

  ge::OpDescPtr relu_op2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(relu_op2, parser::RELU);
  relu_op2->SetName("Relu2");
  relu_op2->AddInputDesc(ge::GeTensorDesc());
  relu_op2->AddOutputDesc(ge::GeTensorDesc());
  relu_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr relu2 = graph->AddNode(relu_op2);

  ge::OpDescPtr relu_op3 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(relu_op3, parser::ACTIVATION);
  relu_op3->SetName("Relu3");
  relu_op3->AddInputDesc(ge::GeTensorDesc());
  relu_op3->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr relu3;
  if (with_leaf_node == true) {
      relu3 = graph->AddNode(relu_op3);
  }

  ge::OpDescPtr mul_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(mul_op, parser::MUL);
  mul_op->SetName("Mul");
  mul_op->AddInputDesc(ge::GeTensorDesc());
  mul_op->AddInputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  mul_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr mul = graph->AddNode(mul_op);

  ge::OpDescPtr mul_op1 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(mul_op1, parser::MUL);
  mul_op1->SetName("Mul1");
  mul_op1->AddInputDesc(ge::GeTensorDesc());
  mul_op1->AddInputDesc(ge::GeTensorDesc());
  mul_op1->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr mul1 = graph->AddNode(mul_op1);

  ge::OpDescPtr mul_op2 = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(mul_op2, parser::MUL);
  mul_op2->SetName("Mul2");
  mul_op2->AddInputDesc(ge::GeTensorDesc());
  mul_op2->AddInputDesc(ge::GeTensorDesc());
  mul_op2->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr mul2 = graph->AddNode(mul_op2);

  ge::OpDescPtr fc_op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(fc_op, parser::FULL_CONNECTION);
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

static Status FusionParserParams(const std::vector<const google::protobuf::Message *> inside_nodes, ge::Operator &op) {
  return domi::SUCCESS;
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
  auto softmax0 = graph_def.add_node();
  auto softmax1 = graph_def.add_node();

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

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  softmax0->set_name("Softmax0");
  softmax0->set_op("Softmax");
  softmax1->set_name("Softmax1");
  softmax1->set_op("Softmax");

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

  softmax0->add_input("add3:0");
  softmax0->add_input("add2:0");
}

TEST_F(UtestTensorflowParser, tensorflow_parser_success) {
  RegisterCustomOp();

  std::string case_dir = __FILE__;
  ParserOperator unused("Add");
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/tensorflow_model/tf_add.pb";
  std::map<ge::AscendString, ge::AscendString> parser_params = {
      {ge::AscendString(ge::ir_option::INPUT_DATA_NAMES), ge::AscendString("Placeholder,Placeholder_1")},
  };
  ge::Graph graph;
  auto ret = ge::aclgrphParseTensorFlow(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "add_test_1");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "add_test_1:0");
}

TEST_F(UtestTensorflowParser, tensorflow_parser_input_data_names_failed) {
  RegisterCustomOp();

  std::string case_dir = __FILE__;
  ParserOperator unused("Add");
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/tensorflow_model/tf_add.pb";
  std::map<ge::AscendString, ge::AscendString> parser_params = {
    {ge::AscendString(ge::ir_option::INPUT_DATA_NAMES), ge::AscendString("Placeholder_1,Placeholder_2")},
  };
  ge::Graph graph;
  auto ret = ge::aclgrphParseTensorFlow(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_model_Failed) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);

  std::string modelFile = caseDir + "/tensorflow_model/model.pb";
  auto status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::SUCCESS);

  modelFile = caseDir + "/tensorflow_model/test_depth_wise_conv2d.pb";
  status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_model_not_exist) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);

  // model file is not exist
  std::string modelFile = caseDir + "/tensorflow_model/conv2d_explicit1_pad.pb";
  auto status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(UtestTensorflowParser, parser_tensorflow_model) {
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
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
  AclGraphParserUtil acl_graph_parse_util;
  std::map<AscendString, AscendString> out_nodes_with_node_and_index = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  ParerUTestsUtils::ClearParserInnerCtx();
  auto ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ret_graph = ge::aclgrphParseTensorFlow(model_file, graph);
  EXPECT_EQ(ret_graph, domi::FAILED);

  // parser tensorflow model success
  modelFile = caseDir + "/tensorflow_model/model.pb";
  model_file = modelFile.c_str();
  out_nodes_with_node_and_index = {{AscendString(ge::ir_option::OUT_NODES), AscendString("x:0;y:0")}};
  ParerUTestsUtils::ClearParserInnerCtx();
  ret = acl_graph_parse_util.ParseParamsBeforeGraph(out_nodes_with_node_and_index, graph_name);
  ret_graph = ge::aclgrphParseTensorFlow(model_file, graph);
  EXPECT_EQ(ret_graph, domi::SUCCESS);
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
  model.SetGraph(sub_graph);
  Buffer buffer;
  graphStatus save_ret = model.Save(buffer, false);
  ASSERT_EQ(save_ret, GRAPH_SUCCESS);
  std::string external_graph(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize());
  // model will failed
  input1->GetOpDesc()->DelAttr(ATTR_NAME_INDEX);
  ge::Model model_will_fail("model_will_fail", "");
  model_will_fail.SetGraph(sub_graph);
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

TEST_F(UtestTensorflowParser, optimize_snapshot) {
  domi::tensorflow::GraphDef graph_def;

  auto mul_node = graph_def.add_node();
  mul_node->set_name("optimizer/Mul");
  mul_node->set_op("Mul");
  mul_node->add_input("Snapshot:0");

  auto snapshot_node = graph_def.add_node();
  snapshot_node->set_name("Snapshot");
  snapshot_node->set_op("Snapshot");
  snapshot_node->add_input("loss_scale/read:0");
  snapshot_node->add_input("^ShuffleNet/AssignMovingAvg");

  auto identity_node = graph_def.add_node();
  identity_node->set_name("loss_scale/read");
  identity_node->set_op("Identity");
  identity_node->add_input("loss_scale/ref:0");

  auto assign_node = graph_def.add_node();
  assign_node->set_name("ShuffleNet/AssignMovingAvg");
  assign_node->set_op("AssignSub");
  assign_node->add_input("ShuffleNet/moving_mean:0");

  Status ret = TensorFlowModelParser().GraphDefOptimize(&graph_def);
  EXPECT_EQ(ret, ge::SUCCESS);
}
TEST_F(UtestTensorflowParser, tensorflow_parser_to_json)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
  std::string jsonFile = caseDir + "/tensorflow_model/test.json";
  const char *model_file = modelFile.c_str();
  const char *json_file = jsonFile.c_str();
  Status ret = modelParser.ToJson(model_file, json_file);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_parserfrommemory_failed)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
  uint32_t size = 1;
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  parser_params = {{AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:0")}};
  ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  ret = modelParser.ParseFromMemory(modelFile.c_str(), size, compute_graph);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, modelparser_parsefrommemory_success)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
  const char* tmp_tf_pb_model = modelFile.c_str();
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  TensorFlowModelParser modelParser;
  MemBuffer* memBuffer = MemBufferFromFile(tmp_tf_pb_model);
  PreChecker::Instance().HasError() == false;
  ret = modelParser.ParseFromMemory((char*)memBuffer->data, memBuffer->size, compute_graph);
  free(memBuffer->data);
  delete memBuffer;
}

TEST_F(UtestTensorflowParser, weightsparser_parsefrommemory_success)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
  const char* tmp_tf_pb_model = modelFile.c_str();
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
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
  subgraph_name = caseDir + "/tensorflow_model/tf_add.pb";
  return subgraph_name;
}

TEST_F(UtestTensorflowParser, parser_ParseProtoWithSubgraphV2)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/tensorflow_model/tf_add.pb";
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(root_proto.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  domi::GetGraphCallbackV2 callback(&getGraphCallbackV2);
  TensorFlowModelParser parser;
  ret = parser.ParseProtoWithSubgraph(root_proto, callback, root_graph);
}

TEST_F(UtestTensorflowParser, parser_ConvertToGeDataType)
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

TEST_F(UtestTensorflowParser, tensorflow_ParserProto_failed)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/tensorflow_model/avgpool3dgrad.pb.txt";
  domi::tensorflow::GraphDef graphDef;
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params;
  Status ret = ge::aclgrphParseTensorFlow(root_proto.c_str(), parser_params, graph);
  ASSERT_EQ(ret, SUCCESS);

  ge::ComputeGraphPtr root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  TensorFlowModelParser tensorflow_parser;
  ret = tensorflow_parser.ParseProto(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  EXPECT_EQ(PARAM_INVALID, ret);

  // proto?a??����㨹
  bool protoRet = parser::ReadProtoFromText(root_proto.c_str(), &graphDef);
  ASSERT_EQ(protoRet, false);
  ret = tensorflow_parser.ParseProto(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  ASSERT_EQ(ret, PARAM_INVALID);
}

std::unique_ptr<google::protobuf::Message> getGraphCallback(const google::protobuf::Message *root_proto, const std::string &graph)
{
  (void)root_proto;
  (void)graph;
  return nullptr;
}

TEST_F(UtestTensorflowParser, tensorflow_parserAllGraph_failed)
{
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/tensorflow_model/conv2d.pb";
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

  ge::ComputeGraphPtr root_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  TensorFlowModelParser tensorflow_parser;
  ret = tensorflow_parser.ParseAllGraph(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  ASSERT_NE(ret, SUCCESS);

  domi::GetGraphCallback callback(&getGraphCallback);
  const auto message_root_proto = reinterpret_cast<google::protobuf::Message *>(&graphDef);
  ret = tensorflow_parser.ParseProtoWithSubgraph(message_root_proto, callback, root_graph);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, test_parse_acl_output_nodes)
{
  AclGraphParserUtil acl_graph_parse_util;
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

TEST_F(UtestTensorflowParser, parse_AutoMappingByOp) {
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

TEST_F(UtestTensorflowParser, parse_ParseNodeDef)
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

TEST_F(UtestTensorflowParser, parse_AddFmkNode)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
  ge::Graph graph;
  string graph_name;
  AclGraphParserUtil acl_graph_parse_util;
  std::map<ge::AscendString, ge::AscendString> parser_options = {{AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:0")}};
  ParerUTestsUtils::ClearParserInnerCtx();
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

TEST_F(UtestTensorflowParser, parse_AddScopeInnerNode)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tf_add.pb";
  std::string op_name = "ge_ascend_irgraph";
  ge::Graph graph(op_name);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
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

  std::string msg = "FastrcnnPredictions";
  ge::Operator *op = new Operator(); 
  modelParser.scope_inner_node_map_.insert({msg, op});
  // can't find in scope_inner_node_map
  ret = modelParser.AddScopeInnerNode(&modelParser, compute_graph, &graph_mutex, node_def);
  EXPECT_EQ(ret, PARAM_INVALID);
  delete op;
  delete node_def;
}

TEST_F(UtestTensorflowParser, dyncmic_rnn_scope_pass_plugin_test) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/tensor_array.pb";
  std::map<ge::AscendString, ge::AscendString> params;
  string key ="enable_scope_fusion_passes";
  string value ="ScopeDynamicRNNPass";
  params.insert(std::make_pair(ge::AscendString(key.c_str()), ge::AscendString(value.c_str())));
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), params, graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, avgpool3dgrad_plugin_test_format_NDHWC) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/avgpool3dgrad_case_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_merge_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/merge.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}

TEST_F(UtestTensorflowParser, tensorflow_no_op_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_no_op.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_identity_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_identity.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_constant_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_constant.pb";
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

TEST_F(UtestTensorflowParser, tensorflow_reshpae_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_reshape.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);

  TensorFlowReshapeParser parser;
  NodeDef * nodeDef = new NodeDef();
  ge::OpDescPtr opdef_ = make_shared<::ge::OpDesc>("","");
  google::protobuf::Map<std::string, tensorflow::AttrValue > *attr_map = nodeDef->mutable_attr();
  domi::tensorflow::AttrValue tshape_attr_value;
  tshape_attr_value.set_type(domi::tensorflow::DT_INT32);
  (*attr_map)[TENSORFLOW_ATTR_TSHAPE] = tshape_attr_value;
  domi::tensorflow::AttrValue t_attr_value;
  t_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*attr_map)[TENSORFLOW_ATTR_T] = t_attr_value;

  Status ret = parser.ParseParams(nodeDef, opdef_);
  EXPECT_EQ(domi::SUCCESS, ret);
  delete nodeDef;
}

TEST_F(UtestTensorflowParser, tensorflow_squeeze_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_sequeeze.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);

  TensorFlowSqueezeParser parser;
  NodeDef *nodeDef = initNodeDef();
  ge::OpDescPtr opDef = make_shared<::ge::OpDesc>("Squeeze","Squeeze");
  Status ret = parser.ParseParams(nodeDef, opDef);
  EXPECT_EQ(ret, SUCCESS);

  NodeDef *nodeDef_dim = initNodeDef_dims();
  ret = parser.ParseParams(nodeDef_dim, opDef);
  EXPECT_EQ(SUCCESS, ret);

  NodeDef *nodeDef_axis_dims = initNodeDef_axis_dims();
  ret = parser.ParseParams(nodeDef_axis_dims, opDef);
  EXPECT_EQ(GRAPH_PARAM_INVALID, ret);

  static const string KEY_SHAPE_LIST = "key_shape_list";
  static const string KEY_TENSOR_LIST = "key_tensor_list";
  static const string KEY_DEFAULT = "key_default";

  NodeDef *nodeDef2 = new NodeDef();
  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = nodeDef2->mutable_attr();
  domi::tensorflow::AttrValue dtype_attr_value ;
  dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;
  //����??strides��?D?
  tensorflow::AttrValue axis_attr_value;
  tensorflow::AttrValue_ListValue *list = axis_attr_value.mutable_list();
  list->add_i(1);
  list->add_i(2);
  (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;
  domi::tensorflow::AttrValue value;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::AttrValue shape;
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)14);
  
  static const string KEY_TYPE_LIST = "key_type_list";
  const std::string ATTR_NAME_INPUT_TENSOR_DESC  = "input_tensor_desc";
  const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;
  value.clear_value();
  value.mutable_list()->add_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE_LIST, value, nodeDef2);

  value.clear_value();
  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
  *(value.mutable_list()->add_func()) = name_attr_list;

  nodeDef2->mutable_attr()->insert({ge::ATTR_NAME_INPUT_TENSOR_DESC, value});
  nodeDef2->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, value});
  ret = parser.ParseParams(nodeDef2, opDef);
  EXPECT_EQ(domi::SUCCESS, ret);

  GeTensorDesc ge_desc;
  ge_desc.SetFormat(ge::FORMAT_C1HWNCoC0);
  ge_desc.SetDataType(ge::DT_FLOAT);
  ge_desc.SetShape(GeShape({1,1,1,1,1,1}));
  ret = parser.ParseDesc(value, ge_desc);
  EXPECT_EQ(ret, SUCCESS);

  delete nodeDef2;
  delete nodeDef_axis_dims;
  delete nodeDef_dim;
  delete nodeDef;
}

TEST_F(UtestTensorflowParser, tensorflow_fill_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_fill.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_shape_n_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_shape_n.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_switch_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_switch.pb";
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

TEST_F(UtestTensorflowParser, tensorflow_enter_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_enter.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);

  TensorFlowEnterParser enterParser;
  ge::OpDescPtr op_dest = make_shared<ge::OpDesc>("Enter", ge::parser::ENTER);
  NodeDef* node_def = initNodeDef();
  node_def->set_name("Enter");
  Status ret = enterParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, FAILED);

  static const string KEY_SHAPE_LIST = "key_shape_list";
  static const string KEY_TENSOR_LIST = "key_tensor_list";
  static const string KEY_DEFAULT = "key_default";

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = node_def->mutable_attr();
  domi::tensorflow::AttrValue dtype_attr_value;
  dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

  //设置strides属性
  domi::tensorflow::AttrValue axis_attr_value;
  ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
  list->add_i(1);
  list->add_i(2);
  (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;

  domi::tensorflow::AttrValue value;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::AttrValue shape;
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)14);
  
  static const string KEY_TYPE_LIST = "key_type_list";
  const std::string ENTER_ATTR_FRAME_NAME = "frame_name";
  const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;
  value.clear_value();
  value.mutable_list()->add_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE_LIST, value, node_def);

  value.clear_value();
  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
  *(value.mutable_list()->add_func()) = name_attr_list;

  node_def->mutable_attr()->insert({ge::ENTER_ATTR_FRAME_NAME, value});
  node_def->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, value});
  ret = enterParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_VariableV2_test) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/tensorflow_model/test_VariableV2.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_fusion_op_parser_test)
{
  TensorFlowFusionOpParser fusionOpParser;
  ge::OpDescPtr op_dest = make_shared<ge::OpDesc>("FusionOp", ge::parser::CONSTANT);
  int index = 0;
  NodeDef* node_def = fusioninitNodeDef(index);
  node_def->set_name("FusionOp");
  auto ret = fusionOpParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  int32_t param = 1;
  ret = fusionOpParser.ParseParamFromConst(node_def, param);
  EXPECT_EQ(ret, SUCCESS);

  ret = fusionOpParser.ParseParamFromConst(node_def, param, index);
  EXPECT_EQ(ret, SUCCESS);

  float params = 0.0;
  ret = fusionOpParser.ParseParamFromConst(node_def, params);
  EXPECT_EQ(ret, SUCCESS);

  index = 2;
  node_def = fusioninitNodeDef(index);
  ret = fusionOpParser.ParseParamFromConst(node_def, params, index);
  EXPECT_EQ(ret, domi::PARAM_INVALID);

  ret = fusionOpParser.ParseHalfFromConst(node_def, params, 0);
  EXPECT_EQ(ret, SUCCESS);

  ret = fusionOpParser.ParseHalfFromConst(node_def, params, 3);
  EXPECT_EQ(ret, domi::PARAM_INVALID);

  node_def = fusioninitNodeDef(0);
  ret = fusionOpParser.ParseHalfFromConst(node_def, params, 3);
  EXPECT_EQ(ret, domi::PARAM_INVALID);

  static const float VALUE_FLOAT = 1.0;
  ge::GeTensorPtr weight = nullptr;
  ret = fusionOpParser.ParseWeightFromConst(node_def, weight);
  EXPECT_EQ(ret, domi::SUCCESS);
  EXPECT_NE(weight, nullptr);

  ge::DataType ge_data_type = weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(ge_data_type, ge::DataType::DT_FLOAT);

  const uint8_t* data_buff = weight->GetData().GetData();
  size_t data_size = weight->GetData().size();
  EXPECT_NE(data_buff, nullptr);
  EXPECT_EQ(data_size, sizeof(float));

  float value_float = *((float*)data_buff);
  EXPECT_EQ(value_float, VALUE_FLOAT);
  delete node_def;
}

TEST_F(UtestTensorflowParser, tensorflow_auto_mapping_parser_adapter_test)
{
  ge::OpDescPtr op_dest = nullptr;
  Message *op_src = nullptr;
  TensorFlowAutoMappingParserAdapter autoMappingParser;
  NodeDef* node_def = initNodeDef();
  Status ret = autoMappingParser.ParseParams(op_src, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);

  op_dest = make_shared<ge::OpDesc>("AutoMapping", ge::parser::CONSTANT);
  ge::OpDescUtilsEx::SetType(op_dest, ge::parser::EMPTY);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  ge::OpDescUtilsEx::SetType(op_dest, ge::parser::IDENTITYN);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  ge::OpDescUtilsEx::SetType(op_dest, ge::parser::SIZE);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  ge::OpDescUtilsEx::SetType(op_dest, ge::parser::SHAPE);
  op_dest->AddOutputDesc(GeTensorDesc());
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_fusion_custom_parser_adapter_test)
{
  REGISTER_CUSTOM_OP("FusionCustom")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("FusionCustom")
    .FusionParseParamsFn(FusionParserParams)
    .ImplyType(ImplyType::TVM);
  register_tbe_op();

  auto graph = std::make_shared<ge::ComputeGraph>("FusionCustom");
  auto op_desc = std::make_shared<ge::OpDesc>("FusionCustom", "FusionCustom");
  auto node = graph->AddNode(op_desc);

  NodeDef *node_def = new NodeDef();
  std::vector<const NodeDef *> v_input_const1;
  v_input_const1.push_back(node_def);

  TensorFlowFusionCustomParserAdapter parser;
  domi::Status status = parser.ParseParams(v_input_const1, node);
  EXPECT_EQ(SUCCESS, status);

  ge::Operator op_src("pool", "pooling");
  std::vector<ge::Operator> v_input_const2;
  v_input_const2.push_back(op_src);
  Status ret = parser.ParseParams(v_input_const2, node);
  EXPECT_EQ(FAILED, ret);
  delete node_def;
}

TEST_F(UtestTensorflowParser, tensorflow_custom_parser_adapter_test)
{
  ge::Operator op_src("pool", "pooling");
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  TensorFlowCustomParserAdapter parser;
  Status ret = parser.ParseParams(op_src, op_dest);
  EXPECT_EQ(ret, FAILED);

  REGISTER_CUSTOM_OP("Variable")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("VariableV2")
    .ParseParamsFn(ParseParams)
    .ParseParamsByOperatorFn(ParseParamByOpFunc)
    .ImplyType(ImplyType::CUSTOM);
  register_tbe_op();

  Operator opSrc(ge::parser::VARIABLE, "VariableV2");
  ret = parser.ParseParams(opSrc, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_graph_functiondef_FindAttrValue_test)
{
  GraphToFunctionDef functionDef;
  NodeDef *node_def = nullptr;
  std::string attr_name = "Const";
  tensorflow::AttrValue attr_value;
  bool ret = functionDef.FindAttrValue(node_def, attr_name, attr_value);
  EXPECT_EQ(ret, false);

  node_def = initNodeDef();
  attr_name = ge::ATTR_NAME_INPUT_TENSOR_DESC;
  node_def->set_name("Const");
  ret = functionDef.FindAttrValue(node_def, attr_name, attr_value);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestTensorflowParser, tensorflow_graph_functiondef_BuildFunctionDef_test)
{
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  string inputNodeType = "DATA";
  MakeDagGraph(subGraph, inputNodeType);

  FunctionDefLibrary library;
  tensorflow::NodeDef call_node_def;
  call_node_def.set_op("fusionop");
  call_node_def.set_name("fusionop");

  vector<ge::InDataAnchorPtr> in_anchor;
  vector<ge::OutDataAnchorPtr> out_anchor;
  for (ge::NodePtr node : subGraph->GetAllNodes()) {
    for (auto in : node->GetAllInDataAnchors()) {
      if (in->GetPeerOutAnchor() != nullptr && in->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType() == parser::DATA) {
          in_anchor.push_back(in);
      }
    }
    for (auto out : node->GetAllOutDataAnchors()) {
      for (auto i : out->GetPeerInDataAnchors()) {
          if (i->GetOwnerNode()->GetOpDesc()->GetType() == parser::NETOUTPUT) {
              out_anchor.push_back(out);
          }
        }
    }
  }
  Status ret = GraphToFunctionDef::BuildFunctionDef(subGraph,
                          "fusionop",
                          &library,
                          &call_node_def,
                          in_anchor,
                          out_anchor);
  EXPECT_EQ(domi::INTERNAL_ERROR, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_CheckOpShapeDim_test)
{
  NodeDef *node_def = initNodeDef();
  std::set<int> dims;
  dims.insert(1);
  dims.insert(2);
  bool valid = true;
  TensorFlowModelParser parser;
  Status ret = parser.CheckOpShapeDim(node_def, dims, valid);
  EXPECT_EQ(ret, SUCCESS);

  static const string KEY_SHAPE_LIST = "key_shape_list";
  static const string KEY_TENSOR_LIST = "key_tensor_list";
  static const string KEY_DEFAULT = "key_default";

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = node_def->mutable_attr();
  domi::tensorflow::AttrValue dtype_attr_value;
  dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

  //����??strides��?D?
  domi::tensorflow::AttrValue axis_attr_value;
  ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
  list->add_i(1);
  list->add_i(2);
  (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;

  domi::tensorflow::AttrValue value;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::AttrValue shape;
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)14);
  
  static const string KEY_TYPE_LIST = "key_type_list";
  const std::string ATTR_NAME_INPUT_TENSOR_DESC  = "input_tensor_desc";
  const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;
  value.clear_value();
  value.mutable_list()->add_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE_LIST, value, node_def);

  value.clear_value();
  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
  *(value.mutable_list()->add_func()) = name_attr_list;

  node_def->mutable_attr()->insert({ge::ATTR_NAME_INPUT_TENSOR_DESC, value});
  node_def->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, value});
  ret = parser.CheckOpShapeDim(node_def, dims, valid);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_Scope_pass_test)
{
  ScopePassManager passmanager;
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }
  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  ge::TensorFlowModelParser tf_model_parser;
  std::vector<string> scope_passes_list = {"pass_1", "pass_2"};
  tf_model_parser.RunScopeFusionPass(scope_passes_list, passmanager, scope_graph);
  Status ret = tf_model_parser.RunScopeFusionPass(scope_passes_list, passmanager, scope_graph);
  EXPECT_NE(ge::SUCCESS, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_variable_v2_parser_test)
{
  TensorFlowCustomParserAdapter parser;
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  NodeDef *node_def = initNodeDef();
  TensorFlowModelParser modelParser;
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("Variable");
  shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  Status ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);

  node_def->set_name("TemporaryVariable");
  node_def->set_op("TemporaryVariable");
  op_parser = factory->CreateOpParser("TemporaryVariable");
  tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);

  NodeDef *nodeDef_temporaryVariable = initOpNodeDef_TemporaryVariable();
  op_parser = factory->CreateOpParser("TemporaryVariable");
  tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  ret = tensorflow_op_parser->ParseParams(nodeDef_temporaryVariable, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  NodeDef *nodeDef_VariableV2 = initOpNodeDef_VariableV2();
  op_parser = factory->CreateOpParser("Variable");
  tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  ret = tensorflow_op_parser->ParseParams(nodeDef_VariableV2, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_var_is_initialized_op_test)
{
  TensorFlowCustomParserAdapter parser;
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  NodeDef *node_def = initNodeDef();
  TensorFlowModelParser modelParser;
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("VarIsInitializedOp");
  shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  Status ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_arg_parser_test)
{
  TensorFlowCustomParserAdapter parser;
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  NodeDef *node_def = initNodeDef();
  TensorFlowModelParser modelParser;
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("_Arg");
  shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  Status ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  static const string KEY_SHAPE_LIST = "key_shape_list";
  static const string KEY_TENSOR_LIST = "key_tensor_list";
  static const string KEY_DEFAULT = "key_default";

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = node_def->mutable_attr();
  domi::tensorflow::AttrValue dtype_attr_value;
  dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

  //����??strides��?D?
  domi::tensorflow::AttrValue axis_attr_value;
  ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
  list->add_i(1);
  list->add_i(2);
  (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;

  domi::tensorflow::AttrValue value;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::AttrValue shape;
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)14);
  
  static const string KEY_TYPE_LIST = "key_type_list";
  const std::string ATTR_NAME_INPUT_TENSOR_DESC  = "input_tensor_desc";
  const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;
  value.clear_value();
  value.mutable_list()->add_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE_LIST, value, node_def);

  value.clear_value();
  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
  *(value.mutable_list()->add_func()) = name_attr_list;

  node_def->mutable_attr()->insert({ge::ATTR_NAME_INPUT_TENSOR_DESC, value});
  node_def->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, value});
  ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_frameworkop_parser_test1)
{
  TensorFlowCustomParserAdapter parser;
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  NodeDef *node_def = initNodeDef();
  TensorFlowModelParser modelParser;
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("FrameworkOp");
  shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  Status ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);

  ChangeDataType(node_def, tensorflow::DT_UINT16);
  ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestTensorflowParser, tensorflow_frameworkop_parser_test2)
{
  TensorFlowCustomParserAdapter parser;
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  NodeDef *node_def = initNodeDef();
  node_def->set_name("FrameworkOp");
  node_def->set_op("_Retval");
  TensorFlowModelParser modelParser;
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("FrameworkOp");
  shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  static const string KEY_SHAPE_LIST = "key_shape_list";
  static const string KEY_TENSOR_LIST = "key_tensor_list";
  static const string KEY_DEFAULT = "key_default";

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = node_def->mutable_attr();
  domi::tensorflow::AttrValue dtype_attr_value;
  dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

  //����??strides��?D?
  domi::tensorflow::AttrValue axis_attr_value;
  ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
  list->add_i(1);
  list->add_i(2);
  (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;

  domi::tensorflow::AttrValue value;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::AttrValue shape;
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)14);
  
  static const string KEY_TYPE_LIST = "key_type_list";
  const std::string ATTR_NAME_INPUT_TENSOR_DESC  = "ATTR_NAME_FRAMEWORK_OP_DEF";
  const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;
  value.clear_value();
  value.mutable_list()->add_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE_LIST, value, node_def);

  value.clear_value();
  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
  *(value.mutable_list()->add_func()) = name_attr_list;

  node_def->mutable_attr()->insert({ge::ATTR_NAME_INPUT_TENSOR_DESC, value});
  node_def->mutable_attr()->insert({ge::ATTR_NAME_OUTPUT_TENSOR_DESC, value});
  Status ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_reshape_parser_test)
{
  TensorFlowCustomParserAdapter parser;
  ge::OpDescPtr op_dest = std::make_shared<ge::OpDesc>();
  NodeDef *node_def = initNodeDef();
  TensorFlowModelParser modelParser;
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("Reshape");
  shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
  Status ret = tensorflow_op_parser->ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  NodeDef * nodeDef = new NodeDef();
  nodeDef->set_op("Reshape");
  google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >* node_attr_map = nodeDef->mutable_attr();
  domi::tensorflow::AttrValue attr_value;
  attr_value.mutable_list()->add_i((int64)32);
  attr_value.mutable_list()->add_i((int64)32);
  attr_value.mutable_list()->add_i((int64)14);

  domi::tensorflow::AttrValue df_attr_value2;
  df_attr_value2.set_s(TENSORFLOWF_TENSOR_NHWC);
  (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value2;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  //����??padding��?D?
  domi::tensorflow::AttrValue pad_attr_value2;
  pad_attr_value2.set_s(TENSORFLOWF_OP_PADDING_SAME);
  (*node_attr_map)[TENSORFLOW_ATTR_PADDING] = pad_attr_value2;
  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_shape", attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  *(attr_value.mutable_list()->add_func()) = name_attr_list;

  GeTensorDesc ge_desc;
  ge_desc.SetFormat(ge::FORMAT_C1HWNCoC0);
  ge_desc.SetDataType(ge::DT_FLOAT);
  ge_desc.SetShape(GeShape({1,1,1,1,1,1}));
  TensorFlowReshapeParser reshapeParser;
  ret = reshapeParser.ParseDesc(attr_value, ge_desc);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_DefunToPartitionedCall_parser_test)
{
  TensorFlowModelParser parser;
  NodeDef *node_def = initNodeDef();
  node_def->set_name("ShapeN");
  ge::OpDescPtr op = make_shared<ge::OpDesc>("ShapeN", ge::parser::PARTITIONEDCALL);
  Status ret = parser.DefunToPartitionedCall(node_def, op);
  EXPECT_EQ(ret, FAILED);

  static const string KEY_SHAPE_LIST = "key_shape_list";
  static const string KEY_TENSOR_LIST = "key_tensor_list";
  static const string KEY_DEFAULT = "key_default";

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = node_def->mutable_attr();
  domi::tensorflow::AttrValue dtype_attr_value;
  dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
  (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

  //����??strides��?D?
  domi::tensorflow::AttrValue axis_attr_value;
  ::tensorflow::AttrValue_ListValue* list = axis_attr_value.mutable_list();
  list->add_i(1);
  list->add_i(2);
  (*node_attr_map)[ge::SQUEEZE_ATTR_AXIS] = axis_attr_value;

  domi::tensorflow::AttrValue value;
  domi::tensorflow::AttrValue df_attr_value;
  df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

  domi::tensorflow::AttrValue pad_attr_value;
  pad_attr_value.set_i((int64_t)tensorflow::DT_FLOAT);

  domi::tensorflow::AttrValue shape;
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)32);
  shape.mutable_list()->add_i((int64)14);
  
  static const string KEY_TYPE_LIST = "key_type_list";
  static const  domi::tensorflow::DataType VALUE_TYPE = domi::tensorflow::DataType::DT_FLOAT;
  value.clear_value();
  value.mutable_list()->add_type(VALUE_TYPE);
  TensorFlowUtil::AddNodeAttr(KEY_TYPE_LIST, value, node_def);

  value.clear_value();
  domi::tensorflow::NameAttrList name_attr_list;
  name_attr_list.mutable_attr()->insert({"serialize_datatype", pad_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_format", df_attr_value});
  name_attr_list.mutable_attr()->insert({"serialize_shape", shape});
  *(value.mutable_list()->add_func()) = name_attr_list;

  node_def->mutable_attr()->insert({"_disable_call_shape_inference", value});
  node_def->mutable_attr()->insert({"_disable_call_shape_inference", value});
  std::string fusion_op_name = "pre_node_a";
  GenOriginContext(&parser, fusion_op_name);
  node_def->set_name("pre_node_a");
  ret = parser.DefunToPartitionedCall(node_def, op);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_TransNodeToOpDesc_parser_test)
{
  TensorFlowModelParser parser;
  NodeDef *node_def = initNodeDef();
  node_def->set_name("ge::parser::DATA");
  std::string op_type = "ge::parser::DATA";
  ge::OpDescPtr op = make_shared<ge::OpDesc>("constant", ge::parser::CONSTANT);
  Status ret = parser.TransNodeToOpDesc(node_def, op, op_type);
  EXPECT_EQ(ret, FAILED);
}

domi::Status fusion_parse_param_by_op(const std::vector<ge::Operator> &op_src, ge::Operator &op) {
  return domi::SUCCESS;
}

TEST_F(UtestTensorflowParser, Fusion_node_parse_params_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);

  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  ASSERT_TRUE(NULL != model_parser);
  TensorFlowModelParser tensorflow_parser;
  domi::tensorflow::NodeDef node_def;
  node_def.set_name("data");
  node_def.set_op("FusionCustom");

  FusionParseParamByOpFunc function = fusion_parse_param_by_op;
  shared_ptr<ge::OpParserFactory> op_parser = ge::OpParserFactory::Instance(domi::TENSORFLOW);
  shared_ptr<OpParser> fusion_op_parser = op_parser->CreateFusionOpParser("FusionCustom");

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  ge::OpDescPtr op1 = std::make_shared<ge::OpDesc>("data", "FusionCustom");
  ge::NodePtr node1 = std::make_shared<ge::Node>(op1, graph);

  vector<const NodeDef *> node_defs;
  node_defs.push_back(&node_def);

  tensorflow_parser.fusion_op_nodedef_map_["data"] = node_defs;
  Status ret = tensorflow_parser.FusionNodeParseParams(fusion_op_parser, &node_def, node1);
  EXPECT_EQ(domi::SUCCESS, ret);
}

TEST_F(UtestTensorflowParser, Tensorflow_recordFusionResult_parser_test)
{
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }

  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  domi::tensorflow::NodeDef node_def;
  node_def.set_name("OP");
  FusionScopesResult *fusion_scope_rlt = new (std::nothrow) FusionScopesResult();
  if (fusion_scope_rlt == nullptr) {
    GELOGE(FAILED, "FusionScopesResult make shared failed.");
    return;
  }
  fusion_scope_rlt->Init();
  fusion_scope_rlt->SetName("OP");
  auto &impl_scope_graph = scope_graph->impl_;
  std::string scope_name = fusion_scope_rlt->Name();
  impl_scope_graph->fusion_results_.insert(std::make_pair(scope_name, fusion_scope_rlt));
  std::vector<ge::OperatorPtr> nodes;
  ge::OperatorPtr op = ge::parser::MakeShared<ge::Operator>("op_name", "op_type");
  if (op == nullptr) {
    GELOGE(FAILED, "Operator make shared failed.");
    return;
  }
  nodes.push_back(op);
  fusion_scope_rlt->impl_->AddNodes(nodes);

  ge::OpDescPtr opDesc = std::make_shared<ge::OpDesc>();
  ge::TensorFlowModelParser tf_model_parser;
  Status ret = tf_model_parser.RecordFusionResult(scope_graph, &node_def, opDesc);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestTensorflowParser, Tensorflow_UpdateFusionOpContext_test)
{
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser = factory->CreateModelParser(domi::TENSORFLOW);
  TensorFlowModelParser tensorflow_parser;
  ScopeFusionOpInfo info;
  ge::OpNodeContext normal_op_node_context;
  ge::OpNodeContext fusion_op_node_context;

  /* 1.?��??��??t */
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  ScopePassManager passmanager;
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);
  NodeDef * node1 = graph->add_node();
  node1->set_name("conv_conv5/BatchNorm/batchnorm/add");
  node1->set_op("Add");
  node1->add_input("conv_conv5/BatchNorm/moving_variance");
  node1->add_input("conv_conv5/BatchNorm/batchnorm/add/y");

  NodeDef * node2 = graph->add_node();
  node2->set_name("conv_conv5/BatchNorm/moving_variance");
  node2->set_op("Const");

  NodeDef * node3 = graph->add_node();
  node3->set_name("conv_conv5/BatchNorm/batchnorm/add/y");
  node3->set_op("Const");

  info.fusion_node_name = "conv_conv5/BatchNorm/batchnorm";
  info.fusion_op_type = ge::parser::FUSIONBATCHNORM;
  info.node_name = "conv_conv5/BatchNorm/batchnorm/add";
  info.description = "";
  info.scope_pass = false;

  EXPECT_EQ(scope_graph->impl_->GetFusionScopesResults(nullptr), nullptr);
  EXPECT_EQ(scope_graph->impl_->GetFusionScopesResults(node1), nullptr);

  Status ret = tensorflow_parser.UpdateFusionOpContext(scope_graph, info, fusion_op_node_context, normal_op_node_context);
  EXPECT_EQ(ret, domi::SUCCESS);


  delete graph;
}

TEST_F(UtestTensorflowParser, Tensorflow_GetInOutPutIndex_scope_pass)
{
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser = factory->CreateModelParser(domi::TENSORFLOW);
  TensorFlowModelParser tensorflow_parser;

  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  ScopePassManager passmanager;
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);
  FusionScopesResult* fusion_rlt = new FusionScopesResult();
  fusion_rlt->Init();
  fusion_rlt->impl_->inputs_.insert(std::make_pair<string, vector<int32_t>>("fw/fw/ToInt32" ,{0}));
  fusion_rlt->impl_->inputs_.insert(std::make_pair<string, vector<int32_t>>("bw/bw/ToInt32" ,{0}));
  fusion_rlt->impl_->inputs_.insert(std::make_pair<string, vector<int32_t>>("bw/ReverseSequence" ,{0, 1}));
  fusion_rlt->impl_->inputs_.insert(std::make_pair<string, vector<int32_t>>("bw/ReverseSequence" ,{1}));

  fusion_rlt->impl_->outputs_.insert(std::make_pair<string, vector<int32_t>>("concat" ,{0}));
  fusion_rlt->impl_->outputs_.insert(std::make_pair<string, vector<int32_t>>("fw/fw/while/Exit_3" ,{1}));
  fusion_rlt->impl_->outputs_.insert(std::make_pair<string, vector<int32_t>>("fw/fw/while/Exit_4" ,{2}));
  fusion_rlt->impl_->outputs_.insert(std::make_pair<string, vector<int32_t>>("bw/bw/while/Exit_3" ,{3}));
  fusion_rlt->impl_->outputs_.insert(std::make_pair<string, vector<int32_t>>("bw/bw/while/Exit_4" ,{4}));
  fusion_rlt->SetType("dynamic_rnn");
  fusion_rlt->SetName("dynamic_rnn_node1");
  scope_graph->impl_->AddFusionScopesResult(fusion_rlt);

  ScopeFusionOpInfo info1;
  info1.node_name = "fw/fw/ToInt32";
  info1.fusion_node_name = "dynamic_rnn_node1";
  info1.fusion_op_type = "dynamic_rnn";
  info1.description = "";
  info1.scope_pass = true;

  bool ignore = false;
  ignore = tensorflow_parser.FusionOpChildIgnore(scope_graph, info1);
  EXPECT_EQ(true, !ignore);

  ScopeFusionOpInfo info2;
  info2.node_name = "fw/fw/others";
  info2.fusion_node_name = "dynamic_rnn_node1";
  info2.fusion_op_type = "dynamic_rnn";
  info2.description = "";
  info2.scope_pass = true;

  ignore = tensorflow_parser.FusionOpChildIgnore(scope_graph, info2);
  EXPECT_EQ(true, ignore);

  ScopeFusionOpInfo input_node_info;
  input_node_info.node_name = "fw/fw/ToInt32";
  input_node_info.fusion_node_name = "dynamic_rnn_node1";
  input_node_info.fusion_op_type = "dynamic_rnn";
  input_node_info.description = "";
  input_node_info.scope_pass = true;

  ScopeFusionOpInfo output_node_info;
  output_node_info.node_name = "fw/fw/while/Exit_3";
  output_node_info.fusion_node_name = "dynamic_rnn_node1";
  output_node_info.fusion_op_type = "dynamic_rnn";
  output_node_info.description = "";
  output_node_info.scope_pass = true;

  int32_t old_index = 0, new_index = -1;
  Status ret = tensorflow_parser.GetInPutIndex(scope_graph, input_node_info, old_index, new_index);
  EXPECT_EQ(domi::SUCCESS, ret);
  EXPECT_EQ(true, (new_index == 0));

  ret = tensorflow_parser.GetOutPutIndex(scope_graph, output_node_info, old_index, new_index);
  EXPECT_EQ(domi::SUCCESS, ret);
  EXPECT_EQ(true, (new_index == 1));

  input_node_info.scope_pass = false;
  ret = tensorflow_parser.GetInPutIndex(scope_graph, input_node_info, old_index, new_index);
  EXPECT_EQ(INTERNAL_ERROR, ret);
  delete graph;
}

TEST_F(UtestTensorflowParser, Tensorflow_AddFusionNodeDef_add_fusion_op_succ)
{
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser = factory->CreateModelParser(domi::TENSORFLOW);
  TensorFlowModelParser tensorflow_parser;
  string fusion_op_name = "dropout";
  string fusion_op_type = "Dropout";
  string description = "test/dropout";
  tensorflow_parser.fusion_op_type_map_[fusion_op_name].push_back(fusion_op_type);
  tensorflow_parser.fusion_op_type_map_[fusion_op_name].push_back(description);

  // op_node_context for fusion op
  ge::OpNodeContext op_node_context;
  op_node_context.input_map["pre_node_a"].push_back({0, 0});
  op_node_context.input_map["pre_node_b"].push_back({0, 1});
  tensorflow_parser.op_node_context_map_[fusion_op_name] = op_node_context;

  // origin inner node def
  NodeDef* node_def = new (std::nothrow) NodeDef();
  node_def->set_name("scope_node_1");
  node_def->set_op("Add");
  tensorflow_parser.fusion_op_nodedef_map_[fusion_op_name].push_back(node_def);

  ScopePassManager pass_manager;
  tensorflow::GraphDef *graph = new (std::nothrow) tensorflow::GraphDef();
  shared_ptr<ScopeGraph> scope_graph = pass_manager.BuildScopeGraph(graph);
  vector<string> node_name_list = {fusion_op_name};
  Status ret = tensorflow_parser.AddFusionNodeDef(scope_graph, node_name_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(tensorflow_parser.nodedef_map_.size(), 1);
  auto fusion_node_def = tensorflow_parser.nodedef_map_[fusion_op_name];
  EXPECT_NE(fusion_node_def, nullptr);
  EXPECT_EQ(fusion_node_def->op(), fusion_op_type);

  delete node_def;
  delete graph;
  tensorflow_parser.DeleteFuisonNodeDef();
}

TEST_F(UtestTensorflowParser, remain_dpop_node)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("dpop_123", "FrameworkOp");
  ge::NodePtr node = std::make_shared<ge::Node>(op, graph);
  graph->AddNode(node);
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  ASSERT_TRUE(NULL != model_parser);

  TensorFlowModelParser tensorflow_parser;
  Status ret = tensorflow_parser.RemoveIsolateNode(graph);
  EXPECT_EQ(domi::SUCCESS, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_UpdateEdgesControlInfo_test)
{
  TensorFlowModelParser model_parser;
  ge::ScopeFusionOpInfo info;
  info.fusion_node_name = "conv_conv5/BatchNorm/batchnorm";
  info.fusion_op_type = ge::parser::FUSIONBATCHNORM;
  info.node_name = "conv_conv5/BatchNorm/batchnorm/add";
  info.description = "";
  info.scope_pass = false;
  model_parser.UpdateEdgesControlInfo(info);
}

TEST_F(UtestTensorflowParser, tensorflow_OptimizeSnapShot_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::NodeDef *curr_mode_def = initNodeDef();
  std::map<string, NodeDef *> nodedef_map;
  nodedef_map.emplace("pre_node_a", curr_mode_def);
  std::pair<string, int> input_data;
  std::vector<string> control_list;
  std::string curr_node_name = "pre_node_a";
  GenOriginContext(&model_parser, curr_node_name);
  Status ret = model_parser.OptimizeSnapShot(curr_mode_def, nodedef_map, input_data, control_list);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  curr_mode_def->set_name("pre_node_a");
  GenOriginContext(&model_parser, curr_node_name);
  ret = model_parser.OptimizeSnapShot(curr_mode_def, nodedef_map, input_data, control_list);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_GraphDefOptimizeSnapShot_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::GraphDef graph_def;
  tensorflow::NodeDef *curr_mode_def = initNodeDef();
  std::map<string, NodeDef *> nodedef_map;
  nodedef_map.emplace("pre_node_a", curr_mode_def);
  std::vector<NodeDef *> nodedef_to_optimize;
  nodedef_to_optimize.emplace_back(curr_mode_def);
  Status ret = model_parser.GraphDefOptimizeSnapShot(&graph_def, nodedef_map, nodedef_to_optimize);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_SetDestNodeName_test)
{
  TensorFlowModelParser model_parser;
  GraphDef graph;
  auto arg0 = AddNode(graph, "_Arg", "arg0");
  auto identity0 = AddNode(graph, "Identity", "identity0");
  auto add0 = AddNode(graph, "Add", "add0");

  int32_t input_idx = 0;
  bool is_control = true;
  bool clear_input_flag = true;
  AddInput(arg0, identity0, 0);
  AddInput(identity0, add0, 0);
  Status ret = model_parser.SetDestNodeName(identity0, add0, input_idx, is_control, clear_input_flag);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_OptimizeDestroyTemporaryVariable_test)
{
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  TensorFlowModelParser tensorflow_parser;

  GraphDef graph;
  auto const0 = AddNode(graph, "Const", "Const0");
  auto tmpVar0 = AddNode(graph, "TemporaryVariable", "TemporaryVariable0");
  auto assign0 = AddNode(graph, "Assign", "Assign0");
  auto destroy0 = AddNode(graph, "DestroyTemporaryVariable", "DestroyTemporaryVariable0");
  auto add0 = AddNode(graph, "Add", "Add0");

  google::protobuf::Map< std::string, tensorflow::AttrValue> *node_attr_map = tmpVar0->mutable_attr();
  tensorflow::AttrValue var_name_attr_value;
  var_name_attr_value.set_s("temporary_variable_name");
  (*node_attr_map)[ge::VAR_ATTR_NAME] = var_name_attr_value;

  google::protobuf::Map<std::string, tensorflow::AttrValue>* node_attr_map_destroy = destroy0->mutable_attr();
  tensorflow::AttrValue var_name_attr_value_destroy;
  var_name_attr_value_destroy.set_s("destroy_temporary_variable_name");
  (*node_attr_map_destroy)[ge::VAR_ATTR_NAME] = var_name_attr_value_destroy;

  AddInput(tmpVar0, assign0, 0);
  AddInput(assign0, destroy0, 0);
  AddInput(const0, add0, 0);
  AddInput(destroy0, add0, 1);

  GraphDef* graphDef = &graph;
  int32_t no_input_node_size_original = 0;
  for (int w = 0; w < graphDef->node_size(); w++) {
      tensorflow::NodeDef* nodeTmp = graphDef->mutable_node(w);
      if (nodeTmp->input_size() == 0) {
        no_input_node_size_original++;
      }
  }

  Status ret = tensorflow_parser.GraphDefOptimize(graphDef);
  int32_t no_input_node_size_result = 0;
  for (int w = 0; w < graphDef->node_size(); w++) {
      tensorflow::NodeDef* nodeTmp = graphDef->mutable_node(w);
      if (nodeTmp->input_size() == 0) {
        no_input_node_size_result ++;
      }
  }
  ASSERT_EQ(ret, domi::FAILED);
  ASSERT_EQ(no_input_node_size_original, no_input_node_size_result);
}

TEST_F(UtestTensorflowParser, tensorflow_OptimizeDestroyTemporaryVariable_test2)
{
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  TensorFlowModelParser tensorflow_parser;

  GraphDef graph;
  auto const0 = AddNode(graph, "Const", "Const0");
  auto tmpVar0 = AddNode(graph, "TemporaryVariable", "TemporaryVariable0");
  auto assign0 = AddNode(graph, "Assign", "Assign0");
  auto destroy0 = AddNode(graph, "DestroyTemporaryVariable", "DestroyTemporaryVariable0");
  auto add0 = AddNode(graph, "Add", "Add0");

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = tmpVar0->mutable_attr();
  tensorflow::AttrValue var_name_attr_value;
  var_name_attr_value.set_s("temporary_variable_name");
  (*node_attr_map)[ge::VAR_ATTR_NAME] = var_name_attr_value;

  google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map_destroy = destroy0->mutable_attr();
  tensorflow::AttrValue var_name_attr_value_destroy;
  var_name_attr_value_destroy.set_s("temporary_variable_name");
  (*node_attr_map_destroy)[ge::VAR_ATTR_NAME] = var_name_attr_value_destroy;

  AddInput(tmpVar0, assign0, 0);
  AddInput(assign0, destroy0, 0);
  AddInput(const0, add0, 0);
  AddInput(destroy0, add0, 1);

  GraphDef* graphDef = &graph;
  int32_t no_input_node_size_original = 0;
  for (int w = 0; w < graphDef->node_size(); w++) {
    tensorflow::NodeDef* nodeTmp = graphDef->mutable_node(w);
    if (nodeTmp->input_size() == 0) {
      no_input_node_size_original ++;
    }
  }

  Status ret = tensorflow_parser.GraphDefOptimize(graphDef);
  int32_t no_input_node_size_result = 0;
  for (int w = 0; w < graphDef->node_size(); w++) {
    tensorflow::NodeDef* nodeTmp = graphDef->mutable_node(w);
    if (nodeTmp->input_size() == 0) {
      no_input_node_size_result ++;
    }
  }
  ASSERT_EQ(ret, domi::SUCCESS);
  ASSERT_EQ(no_input_node_size_original, (no_input_node_size_result - 1));
}

TEST_F(UtestTensorflowParser, tensorflow_AddControlEdgeAfterRemoveInputs_test)
{
  tensorflow::GraphDef graph_def;
  TensorFlowModelParser tensorflow_parser;
  tensorflow::NodeDef *node_def = initNodeDef();
  node_def->set_name("Add0");
  node_def->set_op("add");
  std::map<std::string, NodeDef *> all_node_map;
  all_node_map.emplace("Add0", node_def);
  std::vector<std::string> removed_inputs_vec;
  removed_inputs_vec.emplace_back("Add0");
  Status ret = tensorflow_parser.AddControlEdgeAfterRemoveInputs(&graph_def, node_def, all_node_map, removed_inputs_vec);
  EXPECT_EQ(ret, SUCCESS);

  tensorflow::NodeDef *node_swith = initNodeDef();
  node_swith->set_name("switch_op");
  node_swith->set_op(parser::SWITCH);
  all_node_map.emplace("switch_op", node_swith);
  removed_inputs_vec.clear();
  removed_inputs_vec.emplace_back("switch_op");
  ret = tensorflow_parser.AddControlEdgeAfterRemoveInputs(&graph_def, node_swith, all_node_map, removed_inputs_vec);
  EXPECT_EQ(ret, SUCCESS);
}


TEST_F(UtestTensorflowParser, tensorflow_optimizer_snapshot_no_retval_test) {
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/tensorflow_model/test_snapshot.pb";
  domi::tensorflow::GraphDef graphDef;

  bool protoRet =
      parser::ReadProtoFromBinaryFile(root_proto.c_str(), &graphDef);
  ASSERT_EQ(protoRet, true);

  TensorFlowModelParser tensorflow_parser;
  ge::ComputeGraphPtr root_graph =
      ge::parser::MakeShared<ge::ComputeGraph>("tmp_graph");
  Status ret = tensorflow_parser.ParseProto(
      reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  EXPECT_EQ(FAILED, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_RemoveInputs_test)
{
  tensorflow::GraphDef graph_def;
  tensorflow::NodeDef *node_def = initNodeDef();
  node_def->set_name("OP");
  node_def->add_input("OP/Input_1");
   node_def->add_input("OP/Input_2");
  std::set<uint32_t> remove_index_set;
  std::map<std::string, NodeDef *> all_node_map;
  TensorFlowModelParser model_parser;
  Status ret = model_parser.RemoveInputs(&graph_def, node_def, remove_index_set, all_node_map);
  EXPECT_EQ(ret, SUCCESS);

  remove_index_set.emplace(0);
  ret = model_parser.RemoveInputs(&graph_def, node_def, remove_index_set, all_node_map);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_UpdateInnerNodeContext_test)
{
  std::string fusion_op_name = "post_node_a";
  std::vector<std::string> inner_nodes_name;
  inner_nodes_name.emplace_back("post_node_a");
  TensorFlowModelParser model_parser;
  Status ret = model_parser.UpdateInnerNodeContext(fusion_op_name, inner_nodes_name);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  GenOriginContext(&model_parser, fusion_op_name);
  ret = model_parser.UpdateInnerNodeContext(fusion_op_name, inner_nodes_name);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_UpdateInnerInputMap_test)
{
  string fusion_op_name = "post_node_a";
  OpNodeContext fusion_context;
  std::vector<std::string> inner_nodes_name;
  inner_nodes_name.emplace_back("post_node_a");
  std::set<string> fusion_input_nodes;
  fusion_input_nodes.insert("post_node_a");
  TensorFlowModelParser model_parser;
  GenOriginContext(&model_parser, fusion_op_name);
  model_parser.UpdateInnerInputMap(fusion_op_name, fusion_context, inner_nodes_name, fusion_input_nodes);
}

TEST_F(UtestTensorflowParser, tensorflow_UpdateInnerOutputMap_test)
{
  string fusion_op_name = "post_node_a";
  OpNodeContext fusion_context;
  std::vector<std::string> inner_nodes_name;
  inner_nodes_name.emplace_back("post_node_a");
  std::set<string> fusion_output_nodes;
  fusion_output_nodes.insert("post_node_a");
  TensorFlowModelParser model_parser;
  GenOriginContext(&model_parser, fusion_op_name);
  model_parser.UpdateInnerOutputMap(fusion_op_name, fusion_context, inner_nodes_name, fusion_output_nodes);
}

TEST_F(UtestTensorflowParser, tensorflow_ScopePassManager_AddPass_test)
{
  ScopePassManager passmanager;
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);

  unique_ptr<ScopeBasePass> pass;
  pass.reset(new ScopeTestPass());

  EXPECT_EQ(ge::SUCCESS, passmanager.AddPass(pass));
  EXPECT_NE(ge::SUCCESS, passmanager.Run(scope_graph));

  delete graph;
  graph = nullptr;
}

TEST_F(UtestTensorflowParser, tensorflow_CheckAttrHasType_test1)
{
  tensorflow::AttrValue attr_value;
  attr_value.mutable_list();
  Status ret = TensorFlowUtil::CheckAttrHasType(attr_value, "int");
  EXPECT_EQ(FAILED, ret);

  attr_value.set_type(DT_INVALID);
  ret = TensorFlowUtil::CheckAttrHasType(attr_value, "type");
  EXPECT_EQ(FAILED, ret);

  tensorflow::AttrValue attr_value2;
  AttrValue_ListValue *list = attr_value2.mutable_list();
  list->add_type(tensorflow::DT_FLOAT);
  list->add_type((tensorflow::DataType)30);
  ret = TensorFlowUtil::CheckAttrHasType(attr_value2, "list(type)");
  EXPECT_EQ(FAILED, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_CheckAttrHasType_test2)
{
  tensorflow::AttrValue attr_value;
  AttrValue_ListValue * list = attr_value.mutable_list();
  list->add_type(tensorflow::DT_FLOAT);
  list->add_type(tensorflow::DT_INVALID);
  Status ret = TensorFlowUtil::CheckAttrHasType(attr_value, "list(type)");
  EXPECT_EQ(FAILED, ret);

  attr_value.set_placeholder("test");
  ret = TensorFlowUtil::CheckAttrHasType(attr_value, "");
  EXPECT_EQ(FAILED, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_TransTensorDescriptor_test)
{
  tensorflow::AttrValue attr_value;
  AttrValue_ListValue *list = attr_value.mutable_list();
  list->add_type(tensorflow::DT_FLOAT);

  ParserOperator op;
  uint32_t io = TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG;
  std::string type = ge::parser::FUSEDBATCHNORMGRAD;
  Status ret = TensorFlowUtil::TransTensorDescriptor(attr_value, &op, io, type);
  EXPECT_EQ(ret, SUCCESS);

  io = TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG;
  ret = TensorFlowUtil::TransTensorDescriptor(attr_value, &op, io, type);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_GraphDefOptimizeDestroyTemporaryVariable_test)
{
  tensorflow::GraphDef *graph_def = nullptr;
  tensorflow::NodeDef *nodeCurrent = initNodeDef();
  TensorFlowModelParser model_parser;
  Status ret = model_parser.GraphDefOptimizeDestroyTemporaryVariable(graph_def, nodeCurrent);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_GetFunctionProto_test)
{
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string file = caseDir + "/tensorflow_model/test_enter.pb";
  domi::tensorflow::GraphDefLibrary graph_def_library;
  TensorFlowModelParser model_parser;
  Status ret = model_parser.GetFunctionProto(file, graph_def_library);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_GetNodeFormat_test)
{
  NodeDef *node_def1 = initNodeDef();
  node_def1->set_op("NoOp");
  node_def1->set_name("NoOp");

  NodeDef *node_def2 = initNodeDef();
  node_def2->set_op("Add");
  node_def2->set_name("Add0");
  TfTranspose pred_transpose = TO_NCHW;
  domiTensorFormat_t format = domi::DOMI_TENSOR_NC1HWC0;
  std::set<const NodeDef *> visited_node;
  visited_node.emplace(node_def2);
  TensorFlowModelParser model_parser;
  Status ret = model_parser.GetNodeFormat(node_def1, pred_transpose, format, visited_node);
  EXPECT_EQ(ret, FAILED);
  delete node_def1;
  delete node_def2;
}

TEST_F(UtestTensorflowParser, tensorflow_GetFormatTranspose_test)
{
  NodeDef *transpose_node = initNodeDef();
  transpose_node->set_op("Transpose");
  TfTranspose transpose_direc = NO_TRANSPOSE;
  TensorFlowModelParser modelParser;
  Status ret = modelParser.GetFormatTranspose(transpose_node, transpose_direc);
  EXPECT_EQ(ret, FAILED);

  ge::TensorFlowModelParser parser;
  GraphDef graph;
  auto arg0 = AddNode(graph, "_Arg", "arg0");
  auto snapshot0 = AddNode(graph, "Snapshot", "snapshot0");
  auto ret0 = AddNode(graph, "_Retval", "retval0");

  auto arg1 = AddNode(graph, "_Arg", "arg1");
  auto snapshot1 = AddNode(graph, "Snapshot", "snapshot1");
  auto ret1 = AddNode(graph, "_Retval", "retval1");

  auto arg2 = AddNode(graph, "_Arg", "arg2");
  auto snapshot2 = AddNode(graph, "Snapshot", "snapshot2");
  auto ret2 = AddNode(graph, "_Retval", "retval2");

  AddInput(arg0, snapshot0, 0);
  AddInput(snapshot0, ret0, 0);
  AddInput(arg1, snapshot1, 0);
  AddInput(snapshot1, ret1, 0);
  AddInput(arg2, snapshot2, 0);
  AddInput(snapshot2, ret2, 0);
  AddInput(snapshot0, snapshot1, -1);
  AddInput(snapshot1, snapshot2, -1);

  ASSERT_EQ(parser.GraphDefOptimize(&graph), domi::SUCCESS);
  ASSERT_EQ(ret1->input_size(), 2);
  ret = modelParser.GetFormatTranspose(ret1, transpose_direc);
  EXPECT_EQ(ret, SUCCESS);
  delete transpose_node;
}

TEST_F(UtestTensorflowParser, tensorflow_GetTensorflowGraphInOutMap_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  tensorflow::NodeDef *node_input = graph->add_node();
  node_input->set_name("name_input");
  node_input->set_op("op_input");

  AddGraphNode(graph, "t_lstm/t_lstm_cell/Sigmoid5", "Sigmoid", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Sigmoid6", "Sigmoid", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Sigmoid7", "Sigmoid", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Mul5", "Mul", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Mul6", "Mul", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Mul7", "Mul", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Relu5", "Relu", "node_input");
  AddGraphNode(graph, "t_lstm/t_lstm_cell/Relu6", "Relu", "node_input");
  Status ret = model_parser.GetTensorflowGraphInOutMap(graph);
  EXPECT_EQ(ret, SUCCESS);
  delete graph;
}

TEST_F(UtestTensorflowParser, tensorflow_RemoveIsolateNode_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::GraphDef graph;
  CreateGraphDef(graph);
  Status ret = model_parser.RemoveIsolateNode(&graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, tensorflow_AddNodeToGraphAndMarkFormat_test)
{
  TensorFlowModelParser model_parser;
  ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("default");
  std::vector<std::string> op_node_name_list = {"Const", "placeholder0"};
  GenOriginNodeDef(&model_parser, op_node_name_list);
  Status ret = model_parser.AddNodeToGraphAndMarkFormat(graph, op_node_name_list);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestTensorflowParser, tensorflow_ParserNodeDef1_test)
{
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);

  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  ASSERT_TRUE(NULL != model_parser);
  TensorFlowModelParser tensorflow_parser;
  tensorflow_parser.adaptedOpTypeMap_["test_name"] = "POOLING";
  std::mutex graphMutex;
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  ScopePassManager passmanager;
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);

  domi::tensorflow::NodeDef node_def;
  node_def.set_name("test_name");
  node_def.set_op("POOLING");
  error_message::Context error_context;
  Status ret = ge::TensorFlowModelParser::ParseNodeDef(&tensorflow_parser, compute_graph, &graphMutex, scope_graph, &node_def, error_context);
  EXPECT_EQ(FAILED, ret);
  delete graph;
}

TEST_F(UtestTensorflowParser, tensorflow_ParserNodeDef2_test)
{
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);

  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  ASSERT_TRUE(NULL != model_parser);
  TensorFlowModelParser tensorflow_parser;
  tensorflow_parser.adaptedOpTypeMap_["Pooling"] = "Pooling";
  std::mutex graphMutex;
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  ScopePassManager passmanager;
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);

  REGISTER_CUSTOM_OP("Pooling")
    .FrameworkType(domi::TENSORFLOW)
    .OriginOpType("Pooling")
    .ParseParamsFn(ParseParams)
    .ImplyType(ImplyType::TVM);
  register_tbe_op();
  domi::tensorflow::NodeDef node_def;
  node_def.set_name("Pooling");
  node_def.set_op("Pooling");
  error_message::Context error_context;
  Status ret = ge::TensorFlowModelParser::ParseNodeDef(&tensorflow_parser, compute_graph, &graphMutex, scope_graph, &node_def, error_context);
  EXPECT_EQ(FAILED, ret);
  delete graph;
}

TEST_F(UtestTensorflowParser, tensorflow_AddExternalGraph_test)
{
  TensorFlowModelParser modelParser;
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  std::string inputNodeType = "DATA";
  MakeDagGraph(subGraph, inputNodeType);
  Status ret = modelParser.AddExternalGraph(subGraph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_AddFmkNode_test)
{
  TensorFlowModelParser model_parser;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  tensorflow::GraphDef *graphDef = new (std::nothrow) tensorflow::GraphDef();
  ScopePassManager pass_manager;
  std::shared_ptr<ScopeGraph> scope_graph = pass_manager.BuildScopeGraph(graphDef);
  std::vector<std::string> op_node_name_list = {"Const", "placeholder0"};
  GenOriginNodeDef(&model_parser, op_node_name_list);
  Status ret = model_parser.AddFmkNode(compute_graph, scope_graph, op_node_name_list, false);
  EXPECT_EQ(ret, PARAM_INVALID);
  delete graphDef;
}

TEST_F(UtestTensorflowParser, tensorflow_OptimizeConstNodes4CustomOp_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::GraphDef graph_def;
  CreateGraphDef(graph_def);
  Status ret = model_parser.OptimizeConstNodes4CustomOp(&graph_def);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, OptimizeConstNodes4CustomOp_success)
{
  GraphDef graph;
  auto bn = AddNode(graph, "FusedBatchNormV3", "FusedBatchNormV3_0");
  auto bn_grad = AddNode(graph, "FusedBatchNormGradV3", "FusedBatchNormGradV3_0");

  AddInput(bn, bn_grad, 0);
  AddInput(bn, bn_grad, 1);
  AddInput(bn, bn_grad, 2);
  AddInput(bn, bn_grad, 3);
  AddInput(bn, bn_grad, 5);
  AddInput(bn, bn_grad, 5);

  GraphDef* graphDef = &graph;
  int before_bn_grad_input_size = bn_grad->input_size();
  ASSERT_EQ(before_bn_grad_input_size, 6);

  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser= factory->CreateModelParser(domi::TENSORFLOW);
  ge::TensorFlowModelParser tensorflow_parser;

  Status ret = tensorflow_parser.OptimizeConstNodes4CustomOp(graphDef);
  int after_bn_grad_input_size = bn_grad->input_size();
  ASSERT_EQ(after_bn_grad_input_size, 6);
  ASSERT_EQ(ret, domi::SUCCESS);

  REGISTER_CUSTOM_OP("BatchNormGrad")
      .FrameworkType(domi::TENSORFLOW)
      .OriginOpType({"FusedBatchNormGradV3", "FusedBatchNormGradV2", "FusedBatchNormGrad"})
      .ParseParamsFn(AutoMappingFn)
      .DelInputWithOriginalType(5, "FusedBatchNormGradV3")
      .ImplyType(ImplyType::TVM);
  register_tbe_op();

  ret = tensorflow_parser.OptimizeConstNodes4CustomOp(graphDef);
  after_bn_grad_input_size = bn_grad->input_size();
  ASSERT_EQ(after_bn_grad_input_size, 6);
  ASSERT_EQ(ret, domi::SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_ParseOpParams_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::NodeDef *node_def = initNodeDef();
  node_def->set_name("Pooling");
  node_def->set_op("Pooling");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser("Pooling");
  Status ret = model_parser.ParseOpParams(node_def, op, op_parser);
  EXPECT_EQ(ret, FAILED);

  node_def->set_name("TensorArrayWrite");
  node_def->set_op("TensorArrayWriteV3");
  op_parser = factory->CreateOpParser("TensorArrayWrite");
  ret = model_parser.ParseOpParams(node_def, op, op_parser);
  EXPECT_EQ(ret, SUCCESS);
  delete node_def;
}

TEST_F(UtestTensorflowParser, tensorflow_AddFusionInnerNodeDef_test)
{
  TensorFlowModelParser model_parser;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  tensorflow::GraphDef *graphDef = new (std::nothrow) tensorflow::GraphDef();
  ScopePassManager pass_manager;
  std::shared_ptr<ScopeGraph> scope_graph = pass_manager.BuildScopeGraph(graphDef);
  std::vector<std::string> op_node_name_list = {"Const", "placeholder0"};
  FusionScopesResult *fusion_scope_rlt = new (std::nothrow) FusionScopesResult();
  fusion_scope_rlt->Init();
  fusion_scope_rlt->SetName("FusionCustom");
  auto &impl_scope_graph = scope_graph->impl_;
  std::string scope_name = fusion_scope_rlt->Name();
  impl_scope_graph->fusion_results_.insert(std::make_pair(scope_name, fusion_scope_rlt));
  std::string fusion_op_name = "FusionCustom";
  GenOriginNodeDef(&model_parser, op_node_name_list);
  GenFusionScopesResult(scope_graph, fusion_scope_rlt, fusion_op_name);
  Status ret = model_parser.AddFusionInnerNodeDef(scope_graph, fusion_op_name, op_node_name_list);
  EXPECT_EQ(ret, INTERNAL_ERROR);
  delete graphDef;
}

TEST_F(UtestTensorflowParser, Scope_pass_test)
{
  ScopePassManager passmanager;
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);
  EXPECT_NE(nullptr, scope_graph);

  unique_ptr<ScopeBasePass> pass;
  pass.reset(new ScopeTestPass());
  EXPECT_EQ(domi::SUCCESS, passmanager.AddPass(pass));
  scope_graph = passmanager.BuildScopeGraph(graph);
  EXPECT_NE(nullptr, scope_graph);
  delete graph;
}

TEST_F(UtestTensorflowParser, operator_attr_set_and_get)
{
  TestOperator test_operator;
  test_operator.Name("test_op");
  EXPECT_EQ("test_op" , test_operator.GetName());

  test_operator.Input(test_operator, 0);
  test_operator.Input(test_operator, 1);
  test_operator.GetOpAttrs();

  int64_t pad = 1;
  test_operator.Attr("pad", pad);
  EXPECT_EQ(pad , test_operator.GetIntAttr("pad"));

  bool bool_value = true;
  test_operator.Attr("bool_value", bool_value);
  EXPECT_EQ(bool_value , test_operator.GetBoolAttr("bool_value"));

  float float_value = true;
  test_operator.Attr("float_value", float_value);
  EXPECT_EQ(float_value , test_operator.GetFloatAttr("float_value"));

  std::string str_value = "test_string";
  test_operator.Attr("str_value", str_value);
  EXPECT_EQ(str_value , test_operator.GetStringAttr("str_value"));

  BoolTuple boollist_value{true, false};
  test_operator.Attr("boollist_value", boollist_value);
  BoolTuple get_boollist_value = test_operator.GetBoolTupleAttr("boollist_value");
  EXPECT_EQ(boollist_value[0] , get_boollist_value[0]);

  StringTuple strlist_value{"a", "b"};
  test_operator.Attr("strlist_value", strlist_value);
  StringTuple get_strlist_value = test_operator.GetStringTupleAttr("strlist_value");
  EXPECT_EQ(strlist_value[0] , get_strlist_value[0]);

  int64_t num = 1;
  IntTuple intlist{num, num};
  test_operator.Attr("intlist", intlist);
  IntTuple get_intlist = test_operator.GetIntTupleAttr("intlist");
  EXPECT_EQ(intlist[0] , get_intlist[0]);

  FloatTuple floatlist{1.1, 1.1};
  test_operator.Attr("floatlist", floatlist);
  FloatTuple get_floatlist = test_operator.GetFloatTupleAttr("floatlist");
  EXPECT_EQ(floatlist[0] , get_floatlist[0]);

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  ParserOperator *op = &test_operator;
  Status ret = ConvertToOpDesc(*op, op_desc);
  EXPECT_EQ(domi::SUCCESS , ret);

  TestOperator test_operator_1;
  ParserOperator *op_convert = &test_operator_1;
  ret = ConvertFromOpDesc(op_desc, *op_convert);
  EXPECT_EQ(domi::SUCCESS , ret);

  op_desc = nullptr;
  ret = ConvertFromOpDesc(op_desc, *op_convert);
  EXPECT_EQ(FAILED , ret);

  ret = ConvertToOpDesc(*op, op_desc);
  EXPECT_EQ(FAILED, ret);
}

TEST_F(UtestTensorflowParser, success_frameworkop_get)
{
  FrameworkOpOperator *frameworkOp=new FrameworkOpOperator();
  int64_t index = 1;
  std::string opdef_string = "tensorflow_parser";
  frameworkOp->GetFrameworkType();
  frameworkOp->GetNodeDefPkg();
  frameworkOp->FuncDefPkg("func");
  frameworkOp->Index(index);
  frameworkOp->TfOpDef(opdef_string);
  EXPECT_EQ(SUCCESS, SUCCESS);
  delete frameworkOp;
}

TEST_F(UtestTensorflowParser, op_set_get_success)
{
  ConstantOperator op;
  vector<int64_t> v;
  op.VectorAttr("key", v);
  op.GetDType();
}

TEST_F(UtestTensorflowParser, success_argop_get)
{
  ArgOpOperator *argOp=new ArgOpOperator();
  int64_t index = 1;
  argOp->Index(index);
  argOp->GetIndex();
  EXPECT_EQ(domi::SUCCESS, SUCCESS);
  delete argOp;
}

TEST_F(UtestTensorflowParser, success_operator)
{
  ParserOperator tfOperator;
  ParserOperator in_op;
  uint32_t index = 0;
  std::string type = "add";
  std::string key = "Add";
  std::vector<int64_t> value;
  int64_t tmp = 0;
  value.emplace_back(tmp);
  tfOperator.Input(in_op, index);
  tfOperator.Type(type);
  tfOperator.AttrVector(key, value);
}

TEST_F(UtestTensorflowParser, success_shapen_get)
{
  ShapeNOperator *shapen =new ShapeNOperator();
  shapen->GetInType();
  shapen->GetInType();
  shapen->GetOutType();
  EXPECT_EQ(domi::SUCCESS, domi::SUCCESS);
  delete shapen;
}

TEST_F(UtestTensorflowParser, success_VarIsInitializedOpOperator_get)
{
  VarIsInitializedOpOperator op;
  op.Name("x");
  std::vector<int64_t> value;
  op.VectorAttr("key", value);
}

TEST_F(UtestTensorflowParser, success_variable_op_get)
{
  VariableOperator op;
  uint32_t mem_type = 1;
  op.Name("x");
  std::vector<int64_t> value;
  op.Placement("shared_name");
  op.MemType(mem_type);
}

TEST_F(UtestTensorflowParser, param_success_get)
{
  FillOperator* fillOp=new FillOperator();
  fillOp->GetDataType();
  fillOp->GetAlpha();
  fillOp->GetBeta();
  EXPECT_EQ(domi::SUCCESS, domi::SUCCESS);
  delete fillOp;
}

TEST_F(UtestTensorflowParser, tensorflow_Message2Operator_ParseOperatorAttrs_test)
{
  Message2Operator mess2Op;
  tensorflow::NodeDef *node_def = initNodeDef();
  int depth = 6;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  ge::Operator ops = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  Status ret = mess2Op.ParseOperatorAttrs(node_def, depth, ops);
  EXPECT_EQ(ret, FAILED);

  depth = 4;
  ret = mess2Op.ParseOperatorAttrs(node_def, depth, ops);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_Pb2Json_RepeatedEnum2Json_test)
{
  Pb2Json toJson;
  ProtobufEnumValueDescriptor *enum_value_desc = new google::protobuf::EnumValueDescriptor();
  bool enum2str = true;
  Json json;
  ProtobufFieldDescriptor *field = nullptr;
  toJson.RepeatedEnum2Json(enum_value_desc, enum2str, json);
  toJson.Enum2Json(enum_value_desc, field, enum2str, json);

  enum2str = false;
  toJson.RepeatedEnum2Json(enum_value_desc, enum2str, json);
  delete enum_value_desc;
}

TEST_F(UtestTensorflowParser, tensorflow_Pb2Json_TypeBytes2String_test)
{
  Pb2Json toJson;
  std::string field_name = "offset";
  std::string type_bytes = "offset";
  toJson.TypeBytes2String(field_name, type_bytes);

  field_name = "test";
  toJson.TypeBytes2String(field_name, type_bytes);
}

TEST_F(UtestTensorflowParser, tensorflow_Pb2Json_RepeatedMessage2Json_test)
{
  Pb2Json toJson;
  tensorflow::NodeDef *node_def = initNodeDef();
  ProtobufFieldDescriptor *field = new google::protobuf::FieldDescriptor();
  ProtobufReflection *reflection = nullptr;
  set<string> black_fields;
  black_fields.emplace("offset");
  Json json;
  bool enum2str = true;
  toJson.RepeatedMessage2Json((*node_def), field, reflection, black_fields, json, enum2str);
  delete field;
}

TEST_F(UtestTensorflowParser, tensorflow_Pb2Json_OneField2Json_test)
{
  Pb2Json toJson;
  tensorflow::NodeDef *node_def = initNodeDef();
  ProtobufFieldDescriptor *field = new google::protobuf::FieldDescriptor();
  ProtobufReflection *reflection = nullptr;
  set<string> black_fields;
  black_fields.emplace("offset");
  Json json;
  bool enum2str = true;

  Message2Operator mess2Op;
  int depth = 4;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("FusionCustom", "FusionCustom");
  ge::Operator ops = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  field->CppTypeName(google::protobuf::FieldDescriptor::CPPTYPE_ENUM);
  mess2Op.ParseField(reflection, node_def, field, depth, ops);
  toJson.OneField2Json((*node_def), field, reflection, black_fields, json, enum2str, 1);
  toJson.OneField2Json((*node_def), field, reflection, black_fields, json, enum2str, 5);
  delete field;
}

TEST_F(UtestTensorflowParser, input_proto_real_path_success) {
  const char *caffe_proto_path = "./caffe/caffe.proto";
  const char *custom_proto_path = "./caffe/custom.proto";
  ProtoFileParser proto_file_parser;
  string fusion_proto_file;
  auto ret = proto_file_parser.CombineProtoFile(caffe_proto_path, custom_proto_path, fusion_proto_file);
  EXPECT_EQ(ret, FAILED);

  ret = proto_file_parser.RecordProtoMessage(caffe_proto_path);
  EXPECT_EQ(ret, FAILED);

  ret = proto_file_parser.WriteProtoFile(caffe_proto_path, custom_proto_path);
  EXPECT_EQ(ret, FAILED);

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string proto_file = caseDir + "/tensorflow_model/caffe.proto";
  caffe_proto_path = proto_file.c_str();
  ret = proto_file_parser.CombineProtoFile(caffe_proto_path, caffe_proto_path, fusion_proto_file);
  EXPECT_EQ(ret, SUCCESS);

  ret = proto_file_parser.WriteProtoFile(caffe_proto_path, custom_proto_path);
  EXPECT_EQ(ret, FAILED);

  std::string dest_line = "test";
  ret = proto_file_parser.FindConflictLine(custom_proto_path, 0, dest_line);
  EXPECT_EQ(ret, FAILED);

  std::map<int, std::pair<string, string>> identifier_op_map;
  std::map<std::string, std::pair<int, string>> op_identifier_map;
  ret = proto_file_parser.ParseProtoFile(custom_proto_path, identifier_op_map, op_identifier_map);
  EXPECT_EQ(ret, FAILED);

  proto_file_parser.GetFusionProtoFile();

  std::ofstream write_tmp;
  ret = proto_file_parser.AddCustomAndConflictMessage(custom_proto_path, write_tmp);
  EXPECT_EQ(ret, FAILED);

  ret = proto_file_parser.AddCustomAndConflictLayer(custom_proto_path, write_tmp);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, all_success)
{
  PreChecker::OpId id1 = (void*)(intptr_t)1;
  PreChecker::OpId id2 = (void*)(intptr_t)2;
  PreChecker::OpId id3 = (void*)(intptr_t)3;
  PreChecker::OpId id4 = (void*)(intptr_t)4;
  PreChecker &checker = PreChecker::Instance();

  EXPECT_EQ(checker.AddOp(id1, "name1", "type1"), SUCCESS);
  EXPECT_EQ(checker.AddOp(id2, "name2", "type2"), SUCCESS);
  EXPECT_EQ(checker.AddOp(id3, "name1", "type3"), SUCCESS);
  EXPECT_EQ(checker.AddOp(id4, "name4", ge::parser::DETECTIONOUTPUT), SUCCESS);

  EXPECT_EQ(checker.CheckName(id1), SUCCESS);
  EXPECT_EQ(checker.CheckName(id2), SUCCESS);
  EXPECT_EQ(checker.CheckName(id3), SUCCESS);
  EXPECT_EQ(checker.CheckName(id4), SUCCESS);

  EXPECT_EQ(checker.CheckType(id1), SUCCESS);
  EXPECT_EQ(checker.CheckType(id2), SUCCESS);
  EXPECT_EQ(checker.CheckType(id3), SUCCESS);
  EXPECT_EQ(checker.CheckType(id4), SUCCESS);

  EXPECT_EQ(checker.AddCause(id1, PreChecker::ErrorCode::OK, "msg"), SUCCESS);
  EXPECT_EQ(checker.AddCause(id1, PreChecker::ErrorCode::PARAM_INVALID, "msg"), domi::SUCCESS);

  PreChecker::Cause cause;
  cause.code = PreChecker::ErrorCode::TYPE_AMBIGUOUS;
  cause.message = "msg";
  EXPECT_EQ(checker.AddCause(id1, cause), SUCCESS);
  EXPECT_EQ(checker.HasError(), true);
  EXPECT_EQ(checker.Save("check_result.json"), SUCCESS);

  std::string msg = "msg";
  Status ret = checker.Clear(id1, msg);
  EXPECT_EQ(ret, SUCCESS);

  checker.Clear();
  checker.RefreshErrorMessageByName("name1",PreChecker::ErrorCode::PARAM_INVALID,"node repeated in");
}

TEST_F(UtestTensorflowParser, tensorflow_tbe_tfplugin_loader_test)
{
  TBEPluginLoader pluginLoad;
  vector<string> fileList = {};
  string caffeParserPath = "";
  string full_name = "dabc";
  string caffe_parser_so_suff = "abc";
  pluginLoad.ProcessSoFullName(fileList, caffeParserPath, full_name, caffe_parser_so_suff);
  ASSERT_EQ(caffeParserPath, full_name);

  void *p = (void*)malloc(sizeof(int));
  pluginLoad.handles_vec_.push_back(p);
  pluginLoad.ClearHandles_();

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string proto_file = caseDir + "/tensorflow_model/";
  std::string path = proto_file;
  std::string caffe_parser_path = path;
  pluginLoad.FindParserSo(path, fileList, caffe_parser_path);

  setenv("ASCEND_OPP_PATH", "aaa", 1);
  std::string customop_path = "";
  pluginLoad.GetCustomOpPath(customop_path);
  ASSERT_EQ(customop_path, "aaa/framework/custom/:aaa/framework/built-in/tensorflow/");

  Status ret = pluginLoad.Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_data_op_parser_test)
{
  std::vector<int64_t> shape = {1, 1, 224, 224};
  ge::GeTensorDesc tensor_desc;
  DataOpParser opParser;
  Status ret = opParser.Init5DInputTensor(shape, tensor_desc);
  EXPECT_EQ(ret, SUCCESS);

  ret = opParser.Init5DOutputTensor(shape, tensor_desc);
  EXPECT_EQ(ret, SUCCESS);

  ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  ret = opParser.ParseShape(shape, op);
}

TEST_F(UtestTensorflowParser, read_proto_from_mem_test)
{
  tensorflow::NodeDef *node_def = initNodeDef();
  const char *data = nullptr;
  int size = 3;
  bool ret = parser::ReadProtoFromMem(data, size, node_def);
  EXPECT_EQ(false, ret);

  data = "not file";
  ret = parser::ReadProtoFromMem(data, size, node_def);
  EXPECT_EQ(false, ret);
}

TEST_F(UtestTensorflowParser, tensorflow_GetOriginalType_test)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("fusionCustom", parser::FRAMEWORKOP);
  ge::NodePtr node = std::make_shared<ge::Node>(op, graph);
  string type = parser::FRAMEWORKOP;
  Status ret = parser::GetOriginalType(node, type);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestTensorflowParser, tensorflow_realpath_test)
{
  char path[4096 + 1] = { 0 };
  memset(path, 'a', 4096);
  std::string realPath = parser::RealPath(path);
  EXPECT_EQ(realPath, "");

  const char *real_path = nullptr;
  realPath = parser::RealPath(real_path);
  EXPECT_EQ(realPath, "");
}

TEST_F(UtestTensorflowParser, tensorflow_AclGraphParserUtil_ParseAclInputFp16Nodes_test)
{
  AclGraphParserUtil parserUtil;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  std::string input_fp16_nodes = "Add";
  std::string is_input_adjust_hw_layout = "is_input_adjust_hw_layout";
  Status ret = parserUtil.ParseAclInputFp16Nodes(graph, input_fp16_nodes, is_input_adjust_hw_layout);
  EXPECT_EQ(ret, PARAM_INVALID);

  is_input_adjust_hw_layout = "true";
  ret = parserUtil.ParseAclInputFp16Nodes(graph, input_fp16_nodes, is_input_adjust_hw_layout);
  EXPECT_EQ(ret, PARAM_INVALID);

  vector<string> adjust_fp16_format_vec = {"true", "false"};
  uint32_t index = 1;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  parserUtil.AddAttrsForInputNodes(adjust_fp16_format_vec, input_fp16_nodes, index, op_desc);

  std::string is_output_fp16 = "is_output_fp16";
  ret = parserUtil.ParseAclOutputFp16NodesFormat(is_output_fp16);
  EXPECT_EQ(ret, PARAM_INVALID);

  is_output_fp16 = "false";
  ret = parserUtil.ParseAclOutputFp16NodesFormat(is_output_fp16);
  EXPECT_EQ(ret, SUCCESS);

  is_output_fp16 = "true";
  ret = parserUtil.ParseAclOutputFp16NodesFormat(is_output_fp16);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_ModelSaver_test)
{
  const char *file_path = nullptr;
  const Json model = {{"a", "b"}};
  Status ret = ge::parser::ModelSaver::SaveJsonToFile(file_path, model);
  EXPECT_EQ(ret, FAILED);

  file_path = "./tensorflow_model/";
  ret = ge::parser::ModelSaver::SaveJsonToFile(file_path, model);
  EXPECT_EQ(ret, FAILED);

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string proto_file = caseDir + "/tensorflow_model/caffe.proto";
  file_path = proto_file.c_str();
  ret = ge::parser::ModelSaver::SaveJsonToFile(file_path, model);

  char path[4096 + 1] = { 0 };
  memset(path, 'a', 4096);
  EXPECT_EQ(-1, ge::parser::ModelSaver::CreateDirectory(path));
  EXPECT_EQ(-1, ge::parser::ModelSaver::CheckPath(path));
}

TEST_F(UtestTensorflowParser, create_weights_parser_failed)
{
  WeightsParserFactory* factory = WeightsParserFactory::Instance();
  shared_ptr<WeightsParser> weight_parser = factory->CreateWeightsParser(FRAMEWORK_RESERVED);
  ASSERT_TRUE(NULL == weight_parser);

  ModelParserFactory *modelFactory = ModelParserFactory::Instance();
  shared_ptr<ModelParser> model_parser = modelFactory->CreateModelParser(FRAMEWORK_RESERVED);
  ASSERT_TRUE(NULL == model_parser);

  std::shared_ptr<OpParserFactory> parserFactory = OpParserFactory::Instance(domi::FrameworkType::CAFFE);
  std::shared_ptr<OpParser> fusion_op_parser = parserFactory->CreateFusionOpParser(ge::parser::DATA);
  ASSERT_TRUE(NULL == fusion_op_parser);

  std::shared_ptr<OpParser> op_parser = parserFactory->CreateOpParser("10");
  ASSERT_TRUE(NULL == op_parser);
}

TEST_F(UtestTensorflowParser, custom_parser_adapter_register)
{
  using PARSER_CREATOR_FN = std::function<std::shared_ptr<OpParser>(void)>;
  PARSER_CREATOR_FN func = CustomParserAdapterRegistry::Instance()->GetCreateFunc(domi::TENSORFLOW);
  CustomParserAdapterRegistry::Instance()->Register(domi::TENSORFLOW, func);
  CustomParserAdapterRegistry::Instance()->Register(domi::TENSORFLOW, func);

  func = CustomParserAdapterRegistry::Instance()->GetCreateFunc(domi::FRAMEWORK_RESERVED);
  ASSERT_EQ(nullptr, func);
}

static Status ParseParamsStub1(const google::protobuf::Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

TEST_F(UtestTensorflowParser, tensorflow_parser_api_test)
{

  REGISTER_CUSTOM_OP("Add11")
  .FrameworkType(domi::TENSORFLOW)
  .OriginOpType("Add11")
  .ParseParamsFn(ParseParamsStub1);
  std::map<std::string, std::string> options = {{"ge.runFlag", "1"}};
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::TENSORFLOW)));
  Status ret = ParserInitialize(options);
  EXPECT_EQ(ret, SUCCESS);

  ret = ParserInitialize(options);
  EXPECT_EQ(ret, SUCCESS);

  ret = ParserFinalize();
  EXPECT_EQ(ret, SUCCESS);

  ret = ParserFinalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_parser_api_test_cafee)
{
  std::map<std::string, std::string> options = {{"ge.runFlag", "1"}};
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::CAFFE)));
  Status ret = ParserInitialize(options);
  EXPECT_EQ(ret, SUCCESS);
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::CAFFE)));

  ret = ParserInitialize(options);
  EXPECT_EQ(ret, SUCCESS);

  ret = ParserFinalize();
  EXPECT_EQ(ret, SUCCESS);

  ret = ParserFinalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_FP16_parser_test)
{
  parser::fp16_t fp16;
  fp16.ToDouble();
  fp16.ToInt8();
  fp16.ToUInt8();
  fp16.ToInt16();
  fp16.ToUInt16();
  fp16.ToInt32();
  fp16.ToUInt32();
  fp16.IsInf();
  fp16.operator+(fp16);
  fp16.operator-(fp16);
  fp16.operator*(fp16);
  fp16.operator/(fp16);
  fp16.operator+=(fp16);
  fp16.operator-=(fp16);
  fp16.operator*=(fp16);
  fp16.operator/=(fp16);
  fp16.operator==(fp16);
  fp16.operator!=(fp16);
  fp16.operator>(fp16);
  fp16.operator>=(fp16);
  fp16.operator<(fp16);
  fp16.operator<=(fp16);
  fp16.operator=(fp16);

  float f_val = 0.1;
  fp16.operator=(f_val);
  f_val = 1000000.5;
  fp16.operator=(f_val);
  f_val = 0.00001;
  fp16.operator=(f_val);

  double d_val = 0.2;
  fp16.operator=(d_val);
  d_val = 200000.2;
  fp16.operator=(d_val);
  d_val = 0.00002;
  fp16.operator=(d_val);

  int8_t i_val = 1;
  fp16.operator=(i_val);

  uint8_t ui_val = 2;
  fp16.operator=(ui_val);

  int16_t i_vals = 1;
  fp16.operator=(i_vals);
  i_vals = 5000;
  fp16.operator=(i_vals);
  i_vals = 0;
  fp16.operator=(i_vals);

  uint16_t ui16_val = 1;
  fp16.operator=(ui16_val);
  ui16_val = 0;
  fp16.operator=(ui16_val);
  ui16_val = 1;
  fp16.operator=(ui16_val);
  ui16_val = 5000;
  fp16.operator=(ui16_val);

  int32_t i32_val = 0;
  fp16.operator=(i32_val);
  i32_val = 1;
  fp16.operator=(i32_val);
  i32_val = 5000;
  fp16.operator=(i32_val);

  uint32_t ui32_val = 0;
  fp16.operator=(ui32_val);
  ui32_val = 1;
  fp16.operator=(ui32_val);
  ui32_val = 5000;
  fp16.operator=(ui32_val);
}

TEST_F(UtestTensorflowParser, tensorflow_AclParserInitialize_test)
{
  AclGraphParserUtil parseUtil;
  std::map<std::string, std::string> options;
  Status ret = parseUtil.AclParserInitialize(options);
  EXPECT_EQ(ret, FAILED);

  options = {{ge::FRAMEWORK_TYPE, "2"}};
  ret = parseUtil.AclParserInitialize(options);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_GetOutputLeaf_test)
{
  AclGraphParserUtil parseUtil;
  ge::ComputeGraphPtr compute_graph = build_graph(true);
  ge::NodePtr output_nodes_info = compute_graph->FindNode("Relu3");
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes = {{output_nodes_info,0}};
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = AddNode(compute_graph, "K", parser::NETOUTPUT,1,1);
  Status ret = parseUtil.GetOutputLeaf(node, output_nodes);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, graph_pass_error)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ErrorGraphPass pass;
  ge::parser::PassManager passManager;
  std::vector<std::pair<string, GraphPass*>> passes;
  passes.emplace_back("", &pass);
  Status status = passManager.Run(graph, passes);
  EXPECT_EQ(domi::FAILED, status);
}

TEST_F(UtestTensorflowParser, parser_FindFmkNodeCluser_success)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("FrameworkOp");
  ParserGraphOptimizer graphOptimizer(graph, domi::TENSORFLOW);
  ge::NodePtr node = AddNode(graph, "K", parser::FRAMEWORK_OP_TYPE, 1, 1);
  ge::NodePtr output_nodes_info = graph->FindNode("Relu3");
  std::unordered_map<string, vector<NodePtr>> node_cluser_Map({
    {"x", {node, output_nodes_info}},
  });
  Status ret = graphOptimizer.FindFmkNodeCluser(node_cluser_Map);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, parser_RebuildOutputAnchors_test)
{
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  ParserGraphOptimizer graphOptimizer(subGraph, domi::TENSORFLOW);
  string inputNodeType = "DATA";
  MakeDagGraph(subGraph, inputNodeType);

  vector<ge::InDataAnchorPtr> in_anchor;
  vector<ge::OutDataAnchorPtr> out_anchor;
  for(ge::NodePtr node : subGraph->GetAllNodes()) {
    for(auto out : node->GetAllOutDataAnchors()) {
      for(auto in : node->GetAllInDataAnchors()) {
        if(in->GetPeerOutAnchor() != nullptr && in->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType() == parser::DATA) {
            in_anchor.push_back(in);
        }
      }
      for(auto i : out->GetPeerInDataAnchors()) {
        if(i->GetOwnerNode()->GetOpDesc()->GetType() == parser::NETOUTPUT) {
            out_anchor.push_back(out);
        }
      }
    }
  }
  OpDescPtr fusion_op_desc = make_shared<ge::OpDesc>("FusionCustom", ge::parser::CONSTANT);
  Status ret = graphOptimizer.RebuildOutputAnchors(out_anchor, fusion_op_desc);
  EXPECT_EQ(domi::SUCCESS, ret);

  ret = graphOptimizer.RebuildInputAnchors(in_anchor, fusion_op_desc);
  EXPECT_EQ(domi::SUCCESS, ret);
}

TEST_F(UtestTensorflowParser, parser_LinkInnerAnchor_test)
{
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node_a = AddNode(subGraph, "A", parser::NETOUTPUT, 1, 1);
  NodePtr node_b = AddNode(subGraph, "B", parser::NETOUTPUT, 1, 1);
  unordered_map<string, ge::NodePtr> node_map;
  node_map.insert(pair<string, ge::NodePtr>("A", node_a));
  node_map.insert(pair<string, ge::NodePtr>("B", node_b));

  ParserGraphOptimizer graphOptimizer(subGraph, domi::TENSORFLOW);
  graphOptimizer.LinkInnerAnchor(node_map);
}

TEST_F(UtestTensorflowParser, parser_MarkForFusion_test)
{
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  ParserGraphOptimizer graphOptimizer(subGraph, domi::TENSORFLOW);
  ge::NodePtr node = AddNode(subGraph, "K", parser::FRAMEWORK_OP_TYPE, 1, 1);
  ge::NodePtr output_nodes_info = subGraph->FindNode("Relu3");
  std::unordered_map<string, vector<NodePtr>> node_cluser_Map({
    {"x", {node, output_nodes_info}},
  });
  Status ret = graphOptimizer.MarkForFusion(node_cluser_Map);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestTensorflowParser, parser_UpdateGraph_test)
{
  std::vector<NodePtr> nodes;
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  ParserGraphOptimizer graphOptimizer(subGraph, domi::TENSORFLOW);
  NodePtr node_a = AddNode(subGraph, "A", parser::NETOUTPUT, 1, 1);
  NodePtr node_b = AddNode(subGraph, "B", parser::NETOUTPUT, 1, 1);
  nodes.emplace_back(node_a);
  nodes.emplace_back(node_b);
  Status ret = graphOptimizer.UpdateGraph(nodes);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestTensorflowParser, tensorflow_optimizer_fmk_fusion_op_) {
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  const std::string root_proto = caseDir + "/origin_models/getnext_dynamic_fusion.pbtxt";
  domi::tensorflow::GraphDef graphDef;

  bool protoRet = parser::ReadProtoFromText(root_proto.c_str(), &graphDef);
  ASSERT_EQ(protoRet, true);

  TensorFlowModelParser tensorflow_parser;
  ge::ComputeGraphPtr root_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmp_graph");
  Status ret = tensorflow_parser.ParseProto(reinterpret_cast<google::protobuf::Message *>(&graphDef), root_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(root_graph->GetDirectNode().size(), 3);
}



TEST_F(UtestTensorflowParser, parser_UpdateGraph_node_0)
{
  std::vector<NodePtr> nodes;
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  ParserGraphOptimizer graphOptimizer(subGraph, domi::TENSORFLOW);
  Status ret = graphOptimizer.UpdateGraph(nodes);
  EXPECT_EQ(ret, PARAM_INVALID);
}



TEST_F(UtestTensorflowParser, parser_RebuildFusionNode_test)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
  ParserGraphOptimizer graphOptimizer(graph, domi::TENSORFLOW);
  string inputNodeType = "DATA";
  MakeDagGraph(graph, inputNodeType);
  vector<ge::InDataAnchorPtr> input_anchors;
  vector<ge::OutDataAnchorPtr> output_anchors;
  for(ge::NodePtr node : graph->GetAllNodes()) {
    for(auto out : node->GetAllOutDataAnchors()) {
      for(auto in : node->GetAllInDataAnchors()) {
        if(in->GetPeerOutAnchor() != nullptr && in->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType() == parser::DATA) {
            input_anchors.push_back(in);
        }
      }
      for(auto i : out->GetPeerInDataAnchors()) {
        if(i->GetOwnerNode()->GetOpDesc()->GetType() == parser::NETOUTPUT) {
            output_anchors.push_back(out);
        }
      }
    }
  }
  map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> output_in_map;
  vector<ge::InControlAnchorPtr> input_control_anchors;
  vector<ge::OutControlAnchorPtr> output_control_anchors;

  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("dpop_123", "FrameworkOp");
  ge::NodePtr fusion_node = std::make_shared<ge::Node>(op, graph);
  Status ret = graphOptimizer.RebuildFusionNode(input_anchors, output_anchors, output_in_map, input_control_anchors, output_control_anchors, fusion_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestTensorflowParser, parser_InsertNode_test)
{
  std::vector<NodePtr> nodes;
  ge::ComputeGraphPtr subGraph = std::make_shared<ge::ComputeGraph>("default");
  ParserGraphOptimizer graphOptimizer(subGraph, domi::TENSORFLOW);
  auto merge_node = AddNode(subGraph, "Merge", parser::MERGE, 1, 2);
  auto node1 = AddNode(subGraph, "Op1", parser::RELU, 1, 1);
  auto node2 = AddNode(subGraph, "Op2", parser::CONVOLUTION, 1, 1);
  auto node3 = AddNode(subGraph, "Op3", parser::CONVOLUTION, 1, 1);
  nodes.emplace_back(merge_node);
  nodes.emplace_back(node1);
  nodes.emplace_back(node2);
  nodes.emplace_back(node3);
  vector<ge::InDataAnchorPtr> in_anchor;
  vector<ge::OutDataAnchorPtr> out_anchor;
  map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> output_in_map;
  vector<ge::InControlAnchorPtr> input_control_anchors;
  vector<ge::OutControlAnchorPtr> output_control_anchors;
  unordered_map<string, ge::NodePtr> node_map;
  node_map.insert(pair<string, ge::NodePtr>("A", merge_node));
  node_map.insert(pair<string, ge::NodePtr>("B", node1));
  node_map.insert(pair<string, ge::NodePtr>("C", node2));
  node_map.insert(pair<string, ge::NodePtr>("D", node3));

  Status ret = graphOptimizer.InsertNode(subGraph, nodes, in_anchor, out_anchor, output_in_map, input_control_anchors, output_control_anchors, node_map);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestTensorflowParser, parser_GeStoi_test)
{
  TensorFlowModelParser model_parser;
  string input_node_name = "dynamic_rnn_node1";
  string index_str = "dynamic_rnn";
  int32_t index = 0;

  Status ret = model_parser.GeStoi(input_node_name, index_str, &index);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestTensorflowParser, parser_ConstOpNeedUpdate_test)
{
  ge::TensorFlowModelParser tensorflow_parser;
  NodeDef *op_node_def = new NodeDef();
  op_node_def->set_name("OP");
  op_node_def->add_input("OP/Input_1");
  op_node_def->set_op(TENSORFLOWF_NODE_OP_CONST);

  NodeDef *input_node = new NodeDef();
  input_node->set_op(TENSORFLOWF_NODE_OP_IDENTITY);
  input_node->add_input("OP/Input_1/Input_2");

  NodeDef *input_2 = new NodeDef();
  input_2->set_op(TENSORFLOWF_NODE_OP_IDENTITY);

  tensorflow_parser.nodedef_map_["OP"] = op_node_def;
  tensorflow_parser.nodedef_map_["OP/Input_1"] = input_node;
  tensorflow_parser.nodedef_map_["OP/Input_1/Input_2"] = input_2;

  std::string op_name = "OP/Input_1/Input_2";
  Status ret = tensorflow_parser.ConstOpNeedUpdate(op_name);
  EXPECT_EQ(ret, true);

  op_name = "OP";
  ret = tensorflow_parser.ConstOpNeedUpdate(op_name);
  EXPECT_EQ(ret, true);

  delete op_node_def;
  delete input_node;
  delete input_2;
}

TEST_F(UtestTensorflowParser, parser_UppdateInputMap_test)
{
  ge::TensorFlowModelParser tensorflow_parser;
  ScopeFusionOpInfo info;
  ge::OpNodeContext normal_op_node_context;
  ge::OpNodeContext fusion_op_node_context;

  string fusion_op_name = "dropout";
  normal_op_node_context.input_map["dropout"].push_back({0, 0});
  normal_op_node_context.input_map["conv_conv5/BatchNorm/moving_variance"].push_back({0, 1});
  normal_op_node_context.output_map["dropout"].push_back({1, 0});
  normal_op_node_context.output_map["conv_conv5/BatchNorm/batchnorm/add/y"].push_back({-1, -1});

  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  ScopePassManager passmanager;
  shared_ptr<ScopeGraph> scope_graph = passmanager.BuildScopeGraph(graph);
  NodeDef *node1 = graph->add_node();
  node1->set_name("dropout");
  node1->set_op(TENSORFLOWF_NODE_OP_IDENTITY);
  node1->add_input("conv_conv5/BatchNorm/moving_variance");
  node1->add_input("conv_conv5/BatchNorm/batchnorm/add/y");

  NodeDef *node2 = graph->add_node();
  node2->set_name("conv_conv5/BatchNorm/moving_variance");
  node2->set_op(TENSORFLOWF_NODE_OP_IDENTITY);

  NodeDef *node3 = graph->add_node();
  node3->set_name("conv_conv5/BatchNorm/batchnorm/add/y");
  node3->set_op(TENSORFLOWF_NODE_OP_IDENTITY);

  info.fusion_node_name = "conv_conv5/BatchNorm/batchnorm";
  info.fusion_op_type = parser::FUSIONBATCHNORM;
  info.node_name = "conv_conv5/BatchNorm/batchnorm/add";
  info.description = "";
  info.scope_pass = true;

  tensorflow_parser.nodedef_map_["dropout"] = node1;
  tensorflow_parser.nodedef_map_["conv_conv5/BatchNorm/moving_variance"] = node2;
  tensorflow_parser.nodedef_map_["conv_conv5/BatchNorm/batchnorm/add/y"] = node3;

  Status ret = tensorflow_parser.UppdateInputMap(scope_graph, info, fusion_op_node_context, normal_op_node_context);
  EXPECT_EQ(ret, domi::SUCCESS);

  ret = tensorflow_parser.UppdateOutputMap(scope_graph, info, fusion_op_node_context, normal_op_node_context);

  TensorFlowWeightsParser weights_parser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string proto_file = caseDir + "/ /tf_add.pb";
  const char *file = proto_file.c_str();
  ge::Graph graphs;
  Status weightsRet = weights_parser.Parse(file, graphs);
  EXPECT_EQ(weightsRet, SUCCESS);
  delete graph;
}

TEST_F(UtestTensorflowParser, tensorflow_ConstOpNeedUpdate)
{
  NodeDef *transpose_node = initNodeDef();
  TensorFlowModelParser modelParser;
  modelParser.nodedef_map_["arg1"] = transpose_node;
  bool ret = modelParser.ConstOpNeedUpdate("arg1");
  ASSERT_EQ(ret, true);

  ge::OpNodeContext op_node_context;
  op_node_context.input_map["pre_node_a"].push_back({0, 0});
  op_node_context.input_map["pre_node_ctrl_in"].push_back({-1, -1}); // ctrl edges
  op_node_context.output_map["post_node_b"].push_back({0, 0});
  op_node_context.output_map["post_node_c"].push_back({1, 0});
  op_node_context.output_map["post_node_d"].push_back({-1, -1});
  op_node_context.output_map["_Retval"].push_back({0, 1});
  modelParser.op_node_context_map_["arg1"] = op_node_context;
  ret = modelParser.ConstOpNeedUpdate("arg1");
  ASSERT_EQ(ret, true);

  transpose_node->set_op("NULL");
  modelParser.nodedef_map_["arg2"] = transpose_node;
  ret = modelParser.ConstOpNeedUpdate("arg2");
  ASSERT_EQ(ret, true);
  delete transpose_node;
}

TEST_F(UtestTensorflowParser, tensorflow_IsFusionOpChild)
{
  TensorFlowModelParser modelParser;
  ge::ScopeFusionOpInfo info;
  info.node_name = "node_name";
  info.fusion_node_name = "fusion_node_name";
  info.fusion_op_type = "fusion_op_type";
  info.description = "description";
  info.scope_pass = "scope_pass";

  modelParser.fusion_op_children_["argv1"] = info;
  bool ret = modelParser.IsFusionOpChild("argv1", &info);
  ASSERT_EQ(ret, true);
}

TEST_F(UtestTensorflowParser, tensorflow_UpdateAllNodeOpContext)
{
  TensorFlowModelParser modelParser;
  Status ret;
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }
  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  ge::ScopeFusionOpInfo info;
  info.node_name = "node_name";
  info.fusion_node_name = "fusion_node_name";
  info.fusion_op_type = "fusion_op_type";
  info.description = "description";
  info.scope_pass = "scope_pass";
  modelParser.fusion_op_children_["Const"] = info;
  ge::OpNodeContext op_node_context;
  op_node_context.input_map["pre_node_a"].push_back({0, 0});
  op_node_context.input_map["pre_node_ctrl_in"].push_back({-1, -1}); // ctrl edges
  op_node_context.output_map["post_node_b"].push_back({0, 0});
  op_node_context.output_map["post_node_c"].push_back({1, 0});
  op_node_context.output_map["post_node_d"].push_back({-1, -1});
  op_node_context.output_map["_Retval"].push_back({0, 1});
  modelParser.op_node_context_map_["Const"] = op_node_context;
  NodeDef *node = initNodeDef();
  node->set_op("NULL");
  modelParser.nodedef_map_["Const"] = node;
  std::vector<std::string> op_node_name_list = {"Const"};

  ret = modelParser.UpdateAllNodeOpContext(scope_graph, op_node_name_list);
  EXPECT_EQ(ret, SUCCESS);

  delete node;
}

TEST_F(UtestTensorflowParser, tensorflow_UppdateInputMap)
{
  TensorFlowModelParser modelParser;
  Status ret;
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }
  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  ge::ScopeFusionOpInfo info;
  ge::OpNodeContext fusion_op_node_context, normal_op_node_context;
  info.node_name = "node_name";
  info.fusion_node_name = "fusion_node_name";
  info.fusion_op_type = "fusion_op_type";
  info.description = "description";
  info.scope_pass = "scope_pass";
  modelParser.fusion_op_children_["Const"] = info;
  normal_op_node_context.input_map["Const"].push_back({0, 1});
  normal_op_node_context.output_map["Const"].push_back({0, 1});
  fusion_op_node_context.output_map["Const"].push_back({0, 1});

  NodeDef *node = initNodeDef();
  node->set_op("NULL");
  modelParser.nodedef_map_["Const"] = node;
  ret = modelParser.UppdateInputMap(scope_graph, info, fusion_op_node_context, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);

  ge::ScopeFusionOpInfo info1;
  info1.fusion_node_name = "no fusion_node_name";
  ret = modelParser.UppdateInputMap(scope_graph, info1, fusion_op_node_context, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);

  delete node;
}

TEST_F(UtestTensorflowParser, tensorflow_UppdateOutputMap)
{
  TensorFlowModelParser modelParser;
  Status ret;
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }
  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  ge::ScopeFusionOpInfo info;
  ge::OpNodeContext fusion_op_node_context, normal_op_node_context;
  info.node_name = "node_name";
  info.fusion_node_name = "fusion_node_name";
  info.fusion_op_type = "fusion_op_type";
  info.description = "description";
  info.scope_pass = "scope_pass";
  modelParser.fusion_op_children_["Const"] = info;
  normal_op_node_context.output_map["Const"].push_back({0, 1});
  ret = modelParser.UppdateOutputMap(scope_graph, info, fusion_op_node_context, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);

  ge::ScopeFusionOpInfo info1;
  info1.fusion_node_name = "no fusion_node_name";
  ret = modelParser.UppdateOutputMap(scope_graph, info1, fusion_op_node_context, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestTensorflowParser, tensorflow_EraseNormalOpOutputIfChild)
{
  Status ret;
  TensorFlowModelParser modelParser;
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }
  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  const string op_node_name = "Const";
  OpNodeContext normal_op_node_context;
  normal_op_node_context.input_map["pre_node_a"].push_back({0, 0});
  normal_op_node_context.output_map[op_node_name].push_back({0, 0});

  ge::ScopeFusionOpInfo info;
  info.node_name = "node_name";
  info.fusion_node_name = "fusion_node_name";
  info.fusion_op_type = "fusion_op_type";
  info.description = "description";
  info.scope_pass = "scope_pass";
  modelParser.fusion_op_children_["Const"] = info;

  NodeDef *node = initNodeDef();
  node->set_op("NULL");
  modelParser.nodedef_map_["Const"] = node;

  ret = modelParser.EraseNormalOpOutputIfChild(scope_graph, op_node_name, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);

  delete node;
}

TEST_F(UtestTensorflowParser, tensorflow_UpdateNormalOpContext)
{
  Status ret;
  TensorFlowModelParser modelParser;
  auto scope_graph = ge::parser::MakeShared<ge::ScopeGraph>();
  if (scope_graph == nullptr) {
    GELOGE(FAILED, "Scope graph make shared failed.");
    return;
  }
  if (scope_graph->Init() != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return;
  }

  const string op_node_name = "Const";
  OpNodeContext normal_op_node_context;
  normal_op_node_context.input_map[op_node_name].push_back({0, 0});

  ge::ScopeFusionOpInfo info;
  info.node_name = "node_name";
  info.fusion_node_name = "fusion_node_name";
  info.fusion_op_type = "fusion_op_type";
  info.description = "description";
  info.scope_pass = "scope_pass";
  modelParser.fusion_op_children_["Const"] = info;

  NodeDef *node = initNodeDef();
  node->set_op("NULL");
  modelParser.nodedef_map_["Const"] = node;

  ret = modelParser.UpdateNormalOpContext(scope_graph, op_node_name, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);

  node->set_op("Const");
  modelParser.nodedef_map_["Const"] = node;

  ret = modelParser.UpdateNormalOpContext(scope_graph, op_node_name, normal_op_node_context);
  EXPECT_EQ(ret, SUCCESS);

  delete node;
}

TEST_F(UtestTensorflowParser, tensorflow_OptimizeTranspose)
{
  TensorFlowModelParser modelParser;
  DelTransposeInfo info;
  info.node_def = new NodeDef();
  info.nextNodeDef = new NodeDef();
  info.node_def->add_input("ge");
  info.nextNodeDef->add_input("ge");
  info.inputIdx = 0;
  std::map<std::string, DelTransposeInfo> transposeInfo = {{"ge", info}};
  modelParser.OptimizeTranspose(transposeInfo);

  delete info.node_def;
  delete info.nextNodeDef;
}

TEST_F(UtestTensorflowParser, tensorflow_SoftmaxAddAttr)
{
  TensorFlowModelParser modelParser;
  domi::tensorflow::GraphDef graph_def;
  graph_def.add_node();
  modelParser.SoftmaxAddAttr(&graph_def);
}

TEST_F(UtestTensorflowParser, tensorflow_InferInputFormats)
{
  domiTensorFormat_t ret2;
  TensorFlowModelParser modelParser;

  GetParserContext().format = DOMI_TENSOR_RESERVED;
  
  NodeDef *node = MallocNodeDef("node", "DATA");
  modelParser.nodedef_map_["node"] = node;
  tensorflow_op_map["DATA"] = "node";
  ret2 = modelParser.InferInputFormats();
  EXPECT_EQ(ret2, domi::DOMI_TENSOR_NHWC);
  delete node;
  
  NodeDef* node1 = nullptr;
  modelParser.nodedef_map_["node"] = node1;

  ret2 = modelParser.InferInputFormats();
  EXPECT_EQ(ret2, domi::DOMI_TENSOR_RESERVED);

  char *data = nullptr;
  uint32_t size = 0;
  ge::Graph graph;
  Status ret = modelParser.ParseFromMemory(data, size, graph);
  EXPECT_EQ(ret, SUCCESS);

  string file = "./";
  ret = modelParser.Save(file);
  EXPECT_NE(ret, SUCCESS);

  bool ret1 = modelParser.HasError();
  EXPECT_EQ(ret1, SUCCESS);
  modelParser.Clear();

  TensorFlowWeightsParser tensorflow_weights_parser;
  string file_path = "./";
  ret = tensorflow_weights_parser.Save(file_path);
  EXPECT_NE(ret, SUCCESS);

  ret1 = tensorflow_weights_parser.HasError();
  EXPECT_EQ(ret1, SUCCESS);
  tensorflow_weights_parser.Clear();
}

TEST_F(UtestTensorflowParser, tensorflow_GetTransposeInfo)
{
  Status ret;
  DelTransposeInfo info;
  tensorflow::GraphDef *graph = new tensorflow::GraphDef();
  std::map<std::string, std::string> softmaxInfo = {{"ge", "ge"}};

  info.node_def = new NodeDef();
  info.nextNodeDef = new NodeDef();
  info.node_def->add_input("ge");
  info.nextNodeDef->add_input("ge");
  info.inputIdx = 0;

  NodeDef *node = graph->add_node();
  node->set_op("Transpose");

  std::map<std::string, DelTransposeInfo> transposeInfo = {{"Softmax", info}};
  ret = ge::GetTransposeInfo(graph, softmaxInfo, transposeInfo);
  EXPECT_EQ(ret, SUCCESS);

  node->set_op("Softmax");
  node->set_name("Softmax");
  node->add_input("Softmax");
  ret = ge::GetTransposeInfo(graph, softmaxInfo, transposeInfo);
  EXPECT_EQ(ret, SUCCESS);

  delete info.node_def;
  delete info.nextNodeDef;
  delete graph;
}

TEST_F(UtestTensorflowParser, tensorflow_EraseTransposeNode)
{
  Status ret;
  DelTransposeInfo info;
  std::map<std::string, std::string> softmaxInfo = {{"Softmax", "Softmax"}};

  info.node_def = new NodeDef();
  info.nextNodeDef = new NodeDef();
  info.node_def->add_input("ge");
  info.nextNodeDef->add_input("ge");
  info.nextNodeDef->set_name("ge");
  info.inputIdx = 0;

  std::map<std::string, DelTransposeInfo> transposeInfo = {{"Softmax", info}};

  ret = EraseTransposeNode(softmaxInfo, transposeInfo);
  EXPECT_EQ(ret, FAILED);

  delete info.node_def;
  delete info.nextNodeDef;
}

TEST_F(UtestTensorflowParser, tensorflow_GetUniqueName)
{
  string name_ge = "ge", name_ge_1 = "ge_0", name_ge_2 = "ge_1";
  NameMapHelper helper;
  helper.used_names_.insert(name_ge);
  helper.used_names_.insert(name_ge_1);
  string ret = helper.GetUniqueName(name_ge);
  EXPECT_EQ(ret, name_ge_2);
}

TEST_F(UtestTensorflowParser, tensorflow_UniqueInputOrOutputName)
{
  string name;
  NameMapHelper helper;
  string ret = helper.UniqueInputOrOutputName(name);
  EXPECT_EQ(ret, "unknown");
}

TEST_F(UtestTensorflowParser, tensorflow_Renormalize)
{
  string name = "ge";
  NameMapHelper helper;
  helper.name_mapping_.insert(std::make_pair("ge", "ge"));
  string ret = helper.Renormalize(name);
  EXPECT_EQ(ret, "ge");
}

TEST_F(UtestTensorflowParser, tensorflow_ComputeArgRange)
{
  domi::Status ret;
  domi::tensorflow::NodeDef node_def;
  domi::tensorflow::OpDef::ArgDef arg_def;
  int num;
  ret = ComputeArgRange(node_def, arg_def, &num);
  EXPECT_EQ(ret, domi::INTERNAL_ERROR);
}

TEST_F(UtestTensorflowParser, AddDumpOriginName_test)
{
  GeTensorDesc scalar_tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::ComputeGraphPtr parent_graph = std::make_shared<ge::ComputeGraph>("parent_graph");
  ge::OpDescPtr parent = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(parent, "Foo");
  parent->SetName("foo");
  ge::NodePtr foo = parent_graph->AddNode(parent);


  ge::ComputeGraphPtr sub_graph = std::make_shared<ge::ComputeGraph>("sub_graph");
  auto child = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(child, "Bar");
  child->SetName("bar");
  ge::NodePtr bar = sub_graph->AddNode(child);

  AddDumpOriginName(foo, "f", sub_graph);

  std::vector<std::string> original_names;
  (void)ge::AttrUtils::GetListStr(bar->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  EXPECT_EQ(original_names.size(), 1U);
  EXPECT_EQ(original_names[0], "foo/f/bar");

  (void)ge::AttrUtils::SetListStr(foo->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  AddDumpOriginName(foo, "f", sub_graph);

  original_names.clear();
  (void)ge::AttrUtils::GetListStr(bar->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  EXPECT_EQ(original_names.size(), 1U);
  EXPECT_EQ(original_names[0], "foo/f/bar/f/bar");

  original_names.push_back("abc");
  (void)ge::AttrUtils::SetListStr(foo->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  AddDumpOriginName(foo, "f", sub_graph);

  original_names.clear();
  (void)ge::AttrUtils::GetListStr(bar->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  EXPECT_EQ(original_names.size(), 2U);
  EXPECT_EQ(original_names[0], "foo/f/bar/f/bar/f/bar");
  EXPECT_EQ(original_names[1], "abc");
}

TEST_F(UtestTensorflowParser, test_plugin_manager_getopp_plugin_vendors_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = TBEPluginLoader::GetOppPluginVendors(path_config, vendors);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(vendors[0], "customize");
  EXPECT_EQ(vendors[1], "mdc");
  EXPECT_EQ(vendors[2], "lhisi");
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_getopp_plugin_vendors_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo '' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = TBEPluginLoader::GetOppPluginVendors(path_config, vendors);
  EXPECT_NE(ret, SUCCESS);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_getopp_plugin_vendors_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = TBEPluginLoader::GetOppPluginVendors(path_config, vendors);
  EXPECT_NE(ret, SUCCESS);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_getopp_plugin_vendors_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("rm -rf " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = TBEPluginLoader::GetOppPluginVendors(path_config, vendors);
  EXPECT_NE(ret, SUCCESS);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_getopp_plugin_vendors_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo ' load_priority = customize , mdc , lhisi ' > " + path_config).c_str());

  std::vector<std::string> vendors;
  Status ret = TBEPluginLoader::GetOppPluginVendors(path_config, vendors);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(vendors[0], "customize");
  EXPECT_EQ(vendors[1], "mdc");
  EXPECT_EQ(vendors[2], "lhisi");
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetOpsProtoPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("rm -rf " + path_builtin).c_str());

  std::string opsproto_path;
  Status ret = TBEPluginLoader::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
      opp_path + "op_proto/custom/:" + opp_path + "op_proto/built-in/"
  );
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetOpsProtoPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::string opsproto_path;
  Status ret = TBEPluginLoader::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetOpsProtoPath_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority' > " + path_config).c_str());

  std::string opsproto_path;
  Status ret = TBEPluginLoader::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    opp_path + "op_proto/custom/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetOpsProtoPath_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  setenv("ASCEND_CUSTOM_OPP_PATH", custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = TBEPluginLoader::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path + "/op_proto/:" +
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetOpsProtoPath_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  setenv("ASCEND_CUSTOM_OPP_PATH", "", 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = TBEPluginLoader::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetOpsProtoPath_06) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  setenv("ASCEND_CUSTOM_OPP_PATH", (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/op_proto").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/op_proto").c_str());

  std::string opsproto_path;
  Status ret = TBEPluginLoader::GetOpsProtoPath(opsproto_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(opsproto_path,
    custom_opp_path_01 + "/op_proto/:" +
    custom_opp_path_02 + "/op_proto/:" +
    path_vendors + "/customize/op_proto/:" +
    path_vendors + "/mdc/op_proto/:" +
    path_vendors + "/lhisi/op_proto/:" +
    opp_path + "built-in/op_proto/"
  );
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetCustomOpPath_01) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  system(("rm -rf " + path_builtin).c_str());

  std::string customop_path;
  TBEPluginLoader::GetCustomOpPath(customop_path);
  EXPECT_EQ(customop_path.find(opp_path + "framework/custom/:" + opp_path + "framework/built-in/"), 0);
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetCustomOpPath_02) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());

  std::string customop_path;
  TBEPluginLoader::GetCustomOpPath(customop_path);
  EXPECT_EQ(customop_path.find(
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/"), 0);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetCustomOpPath_03) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority' > " + path_config).c_str());

  std::string customop_path;
  TBEPluginLoader::GetCustomOpPath(customop_path);
  EXPECT_EQ(customop_path.find(
    opp_path + "framework/custom/:" +
    opp_path + "built-in/framework/"), 0);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetCustomOpPath_04) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  setenv("ASCEND_CUSTOM_OPP_PATH", custom_opp_path.c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework").c_str());

  std::string customop_path;
  TBEPluginLoader::GetCustomOpPath(customop_path);
  EXPECT_EQ(customop_path.find(
    custom_opp_path + "/framework/:" +
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/"), 0);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetCustomOpPath_05) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path = opp_path + "custom_opp_path";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  setenv("ASCEND_CUSTOM_OPP_PATH", "", 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path + "/framework").c_str());

  std::string customop_path;
  TBEPluginLoader::GetCustomOpPath(customop_path);
  EXPECT_EQ(customop_path.find(
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/"), 0);
  system(("rm -rf " + opp_path).c_str());
}

TEST_F(UtestTensorflowParser, test_plugin_manager_GetCustomOpPath_06) {
  std::string opp_path = __FILE__;
  opp_path = opp_path.substr(0, opp_path.rfind("/") + 1) + "opp_path/";
  std::string custom_opp_path_01 = opp_path + "custom_opp_path_01";
  std::string custom_opp_path_02 = opp_path + "custom_opp_path_02";
  setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
  setenv("ASCEND_CUSTOM_OPP_PATH", (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1);

  std::string path_builtin = opp_path + "built-in";
  std::string path_vendors = opp_path + "vendors";
  std::string path_config = path_vendors + "/config.ini";
  system(("mkdir -p " + path_builtin).c_str());
  system(("mkdir -p " + path_vendors).c_str());
  system(("echo 'load_priority=customize,mdc,lhisi' > " + path_config).c_str());
  system(("mkdir -p " + custom_opp_path_01 + "/framework").c_str());
  system(("mkdir -p " + custom_opp_path_02 + "/framework").c_str());

  std::string customop_path;
  TBEPluginLoader::GetCustomOpPath(customop_path);
  EXPECT_EQ(customop_path.find(
    custom_opp_path_01 + "/framework/:" +
    custom_opp_path_02 + "/framework/:" +
    path_vendors + "/customize/framework/:" +
    path_vendors + "/mdc/framework/:" +
    path_vendors + "/lhisi/framework/:" +
    opp_path + "built-in/framework/"), 0);
  system(("rm -rf " + opp_path).c_str());
}
} // namespace ge
