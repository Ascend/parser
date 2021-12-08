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
#include "parser/tensorflow/tensorflow_fusion_op_parser.h"
#include "parser/tensorflow/tensorflow_auto_mapping_parser_adapter.h"
#include "parser/common/op_def/arg_op.h"
#include "parser/tensorflow/tensorflow_fusion_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_reshape_parser.h"
#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_squeeze_parser.h"
#include "parser/tensorflow/graph_functiondef.h"
#include "parser/tensorflow/graph_optimizer.h"
#include "cce/dnn_base_def.hpp"
#include "parser/tensorflow/scope/scope_pass_manager.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "compute_graph_impl.h"
#include "parser/tensorflow/tensorflow_enter_parser.h"
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
class STestTensorflowParser : public testing::Test {
 protected:
  void SetUp() {
    ParerSTestsUtils::ClearParserInnerCtx();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
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

    //设置data_format属性
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

    //设置data_format属性
    domi::tensorflow::AttrValue df_attr_value;
    domi::tensorflow::AttrValue df_attr_value2;
    df_attr_value2.set_s(TENSORFLOWF_TENSOR_NHWC);

    df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);
    (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value2;
    //设置padding属性
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

    //设置dtype属性
    domi::tensorflow::AttrValue type_attr;
    type_attr.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[VAR_ATTR_DTYPE] = type_attr;

    //设置var_name属性
    domi::tensorflow::AttrValue var_name_attr_value;
    var_name_attr_value.set_s("temporary_variable_name");
    (*node_attr_map)[ge::VAR_ATTR_NAME] = var_name_attr_value;

    //设置shape属性
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

    //设置data_format属性
    domi::tensorflow::AttrValue df_attr_value2;
    df_attr_value2.set_s(TENSORFLOWF_TENSOR_NHWC);
    (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value2;
    domi::tensorflow::AttrValue df_attr_value;
    df_attr_value.set_i((int64_t)ccTensorFormat_t::CC_TENSOR_NHWC);

    //设置padding属性
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
    NodeDef * nodeDef = new NodeDef();
    ::google::protobuf::Map< ::std::string, ::tensorflow::AttrValue >* node_attr_map = nodeDef->mutable_attr();

    //设置 type属性
    domi::tensorflow::AttrValue dtype_attr_value ;

    if (index == 0) {
        dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
    } else if (index == 1) {
        dtype_attr_value.set_type(domi::tensorflow::DT_INT32);
    } else if (index == 2) {
        dtype_attr_value.set_type(tensorflow::DT_HALF);
    }
    (*node_attr_map)[ge::TENSORFLOW_ATTR_DTYPE] = dtype_attr_value;
    //设置data_format属性
    domi::tensorflow::AttrValue df_attr_value;
    df_attr_value.set_s(TENSORFLOWF_TENSOR_NCHW);
    (*node_attr_map)[TENSORFLOW_ATTR_DATA_FORMAT] = df_attr_value;

    // 设置 tensor 属性
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
      OpRegistrationTbe::Instance()->Finalize(reg_data);
      OpRegistry::Instance()->Register(reg_data);
    }
    OpRegistry::Instance()->registrationDatas.clear();
  }

  NodeDef *initNodeDef_axis_dims() {
    NodeDef *nodeDef = new NodeDef();
    google::protobuf::Map<std::string, tensorflow::AttrValue> *node_attr_map = nodeDef->mutable_attr();

    //设置T属性
    domi::tensorflow::AttrValue dtype_attr_value ;
    dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

    //设置strides属性
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

    //设置T属性
    domi::tensorflow::AttrValue dtype_attr_value ;
    dtype_attr_value.set_type(domi::tensorflow::DT_FLOAT);
    (*node_attr_map)[TENSORFLOW_ATTR_T] = dtype_attr_value;

    //设置strides属性
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
    opDef->SetType(_type);
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

  std::string serialized_proto = "";
  ret = tensorflow_parser.ParseProto(serialized_proto, root_graph);
  ASSERT_EQ(ret, FAILED);
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
  tensorflow::NodeDef *node_def = initNodeDef();
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
  EXPECT_EQ(status, FAILED);
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

TEST_F(STestTensorflowParser, tensorflow_squeeze_test) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_sequeeze.pb";
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
  //设置strides属性
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

TEST_F(STestTensorflowParser, tensorflow_VariableV2_test) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/test_VariableV2.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_fusion_op_parser_test)
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

TEST_F(STestTensorflowParser, tensorflow_auto_mapping_parser_adapter_test)
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
  op_dest->SetType(ge::parser::EMPTY);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  op_dest->SetType(ge::parser::IDENTITYN);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  op_dest->SetType(ge::parser::SIZE);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);

  op_dest->SetType(ge::parser::SHAPE);
  ret = autoMappingParser.ParseParams(node_def, op_dest);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_fusion_custom_parser_adapter_test)
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

TEST_F(STestTensorflowParser, tensorflow_custom_parser_adapter_test)
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

TEST_F(STestTensorflowParser, tensorflow_graph_functiondef_FindAttrValue_test)
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

TEST_F(STestTensorflowParser, tensorflow_graph_functiondef_BuildFunctionDef_test)
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

TEST_F(STestTensorflowParser, tensorflow_CheckOpShapeDim_test)
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

TEST_F(STestTensorflowParser, tensorflow_Scope_pass_test)
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

TEST_F(STestTensorflowParser, tensorflow_variable_v2_parser_test)
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

TEST_F(STestTensorflowParser, tensorflow_var_is_initialized_op_test)
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

TEST_F(STestTensorflowParser, tensorflow_arg_parser_test)
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

TEST_F(STestTensorflowParser, tensorflow_frameworkop_parser_test)
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

TEST_F(STestTensorflowParser, tensorflow_reshape_parser_test)
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

  //设置padding属性
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

TEST_F(STestTensorflowParser, tensorflow_DefunToPartitionedCall_parser_test)
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

TEST_F(STestTensorflowParser, tensorflow_TransNodeToOpDesc_parser_test)
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

TEST_F(STestTensorflowParser, Fusion_node_parse_params_success) {
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

TEST_F(STestTensorflowParser, Tensorflow_recordFusionResult_parser_test)
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

TEST_F(STestTensorflowParser, Tensorflow_UpdateFusionOpContext_test)
{
  ModelParserFactory* factory = ModelParserFactory::Instance();
  shared_ptr<domi::ModelParser> model_parser = factory->CreateModelParser(domi::TENSORFLOW);
  TensorFlowModelParser tensorflow_parser;
  ScopeFusionOpInfo info;
  ge::OpNodeContext normal_op_node_context;
  ge::OpNodeContext fusion_op_node_context;

  /* 1.预置条件 */
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

TEST_F(STestTensorflowParser, Tensorflow_GetInOutPutIndex_scope_pass)
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
  delete graph;
}

TEST_F(STestTensorflowParser, Tensorflow_AddFusionNodeDef_add_fusion_op_succ)
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

TEST_F(STestTensorflowParser, remain_dpop_node)
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

TEST_F(STestTensorflowParser, tensorflow_UpdateEdgesControlInfo_test)
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

TEST_F(STestTensorflowParser, tensorflow_OptimizeIdentityByOutput_test)
{
  TensorFlowModelParser model_parser;
  NodeDef *node_def = new NodeDef();
  node_def->set_name("Placeholder");
  node_def->set_op("Placeholder_0");
  std::map<string, NodeDef *> nodedef_map;
  nodedef_map.emplace("Placeholder", node_def);
  std::string curr_node_name = "Placeholder";
  bool clear_input_flag = true;
  Status ret = model_parser.OptimizeIdentityByOutput(nodedef_map, curr_node_name, clear_input_flag);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  GraphDef graph;
  curr_node_name = "pre_node_a";
  nodedef_map.emplace("pre_node_a", node_def);
  node_def->set_op("pre_node_a");
  GenOriginContext(&model_parser, curr_node_name);
  ret = model_parser.OptimizeIdentityByOutput(nodedef_map, curr_node_name, clear_input_flag);
  EXPECT_EQ(ret, SUCCESS);
  delete node_def;
}

TEST_F(STestTensorflowParser, tensorflow_OptimizeSnapShot_test)
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

TEST_F(STestTensorflowParser, tensorflow_GraphDefOptimizeSnapShot_test)
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

TEST_F(STestTensorflowParser, tensorflow_SetDestNodeName_test)
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

TEST_F(STestTensorflowParser, tensorflow_OptimizeDestroyTemporaryVariable_test)
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

TEST_F(STestTensorflowParser, tensorflow_OptimizeDestroyTemporaryVariable_test2)
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

TEST_F(STestTensorflowParser, tensorflow_AddControlEdgeAfterRemoveInputs_test)
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
}

TEST_F(STestTensorflowParser, tensorflow_GraphDefOptimizeIdentity_test)
{
  tensorflow::GraphDef graph_def;
  TensorFlowModelParser tensorflow_parser;
  tensorflow::NodeDef *node_def = initNodeDef();
  node_def->set_name("post_node_d");

  std::map<string, NodeDef *> nodedef_map;
  nodedef_map.emplace("post_node_d", node_def);
  nodedef_map.emplace("post_node_a", node_def);
  nodedef_map.emplace("post_node_b", node_def);
  std::vector<NodeDef *> nodedef_to_optimize;
  nodedef_to_optimize.emplace_back(node_def);

  std::string curr_node_name = "post_node_b";
  GenOriginContext(&tensorflow_parser, curr_node_name);
  Status ret = tensorflow_parser.GraphDefOptimizeIdentity(&graph_def, nodedef_map, nodedef_to_optimize);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(STestTensorflowParser, tensorflow_RemoveInputs_test)
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

TEST_F(STestTensorflowParser, tensorflow_UpdateInnerNodeContext_test)
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

TEST_F(STestTensorflowParser, tensorflow_UpdateInnerInputMap_test)
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

TEST_F(STestTensorflowParser, tensorflow_UpdateInnerOutputMap_test)
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

TEST_F(STestTensorflowParser, tensorflow_ScopePassManager_AddPass_test)
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

TEST_F(STestTensorflowParser, tensorflow_CheckAttrHasType_test1)
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

TEST_F(STestTensorflowParser, tensorflow_CheckAttrHasType_test2)
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

TEST_F(STestTensorflowParser, tensorflow_TransTensorDescriptor_test)
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

TEST_F(STestTensorflowParser, tensorflow_GraphDefOptimizeDestroyTemporaryVariable_test)
{
  tensorflow::GraphDef *graph_def = nullptr;
  tensorflow::NodeDef *nodeCurrent = initNodeDef();
  TensorFlowModelParser model_parser;
  Status ret = model_parser.GraphDefOptimizeDestroyTemporaryVariable(graph_def, nodeCurrent);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestTensorflowParser, tensorflow_GetFunctionProto_test)
{
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string file = caseDir + "/origin_models/test_enter.pb";
  domi::tensorflow::GraphDefLibrary graph_def_library;
  TensorFlowModelParser model_parser;
  Status ret = model_parser.GetFunctionProto(file, graph_def_library);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestTensorflowParser, tensorflow_GetNodeFormat_test)
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

TEST_F(STestTensorflowParser, tensorflow_GetFormatTranspose_test)
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

TEST_F(STestTensorflowParser, tensorflow_GetTensorflowGraphInOutMap_test)
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

TEST_F(STestTensorflowParser, tensorflow_RemoveIsolateNode_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::GraphDef graph;
  CreateGraphDef(graph);
  Status ret = model_parser.RemoveIsolateNode(&graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestTensorflowParser, tensorflow_AddNodeToGraphAndMarkFormat_test)
{
  TensorFlowModelParser model_parser;
  ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("default");
  std::vector<std::string> op_node_name_list = {"Const", "placeholder0"};
  GenOriginNodeDef(&model_parser, op_node_name_list);
  Status ret = model_parser.AddNodeToGraphAndMarkFormat(graph, op_node_name_list);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(STestTensorflowParser, tensorflow_ParserNodeDef1_test)
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

TEST_F(STestTensorflowParser, tensorflow_ParserNodeDef2_test)
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

TEST_F(STestTensorflowParser, tensorflow_AddExternalGraph_test)
{
  TensorFlowModelParser modelParser;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/origin_models/tf_add.pb";
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parser_params = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:0")}};
  auto ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parser_params, graph);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  ret = modelParser.AddExternalGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_AddFmkNode_test)
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

TEST_F(STestTensorflowParser, tensorflow_OptimizeConstNodes4CustomOp_test)
{
  TensorFlowModelParser model_parser;
  tensorflow::GraphDef graph_def;
  CreateGraphDef(graph_def);
  Status ret = model_parser.OptimizeConstNodes4CustomOp(&graph_def);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STestTensorflowParser, tensorflow_ParseOpParams_test)
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

} // namespace ge
