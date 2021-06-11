/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "parser/tensorflow/tensorflow_parser.h"
#include <algorithm>
#include <iostream>
#include "ge/ge_api_types.h"
#include "parser/common/convert/pb2json.h"
#include "parser/common/acl_graph_parser_util.h"
#include "common/util/error_manager/error_manager.h"
#include "external/graph/operator_factory.h"
#include "external/parser/tensorflow_parser.h"
#include "external/register/scope/scope_fusion_pass_register.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "iterator_fusion_pass.h"
#include "omg/parser/op_parser.h"
#include "omg/parser/parser_factory.h"
#include "parser/common/acl_graph_parser_util.h"
#include "parser/common/model_saver.h"
#include "parser/common/op_map.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/parser_fp16_t.h"
#include "parser/common/pass_manager.h"
#include "parser/common/pre_checker.h"
#include "parser/common/prototype_pass_manager.h"
#include "parser/common/thread_pool.h"
#include "parser/common/parser_utils.h"
#include "parser/common/util.h"
#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_fusion_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_fusion_op_parser.h"
#include "parser/tensorflow/tensorflow_fusionop_util.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "register/op_registry.h"
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pass_registry_impl.h"

using ge::const_op_update_vec;
using ge::OpParserFactory;
using ge::Pb2Json;
using ge::PreChecker;
using ge::TENSORFLOW_ATTR_DATA_FORMAT;
using ge::TENSORFLOW_ATTR_DTYPE;
using ge::TENSORFLOW_ATTR_SHAPE;
using ge::TENSORFLOW_ATTR_T;
using ge::TENSORFLOW_ATTR_TYPE_STRING;
using ge::TENSORFLOW_ATTR_TYPE_TENSOR;
using ge::TENSORFLOW_ATTR_TYPE_TYPE;
using ge::TENSORFLOW_ATTR_VALUE;
using ge::TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG;
using ge::TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG;
using ge::tensorflow_op_map;
using ge::tensorflow_train_op_map;
using ge::TENSORFLOWF_NODE_OP_CONST;
using ge::TENSORFLOWF_NODE_OP_IDENTITY;
using ge::TENSORFLOWF_NODE_OP_MERGE;
using ge::TENSORFLOWF_NODE_OP_PLACEHOLDER;
using ge::TENSORFLOWF_NODE_OP_SWITCH;
using ge::TENSORFLOWF_NODE_OP_TRANSPOSE;
using ge::TENSORFLOWF_TENSOR_NCHW;
using ge::TENSORFLOWF_TENSOR_NHWC;
using ge::TensorFlowFunsionOPUtil;
using ge::TensorFlowFusionCustomParserAdapter;
using ge::TensorFlowFusionOpParser;
using ge::TensorFlowOpParser;
using ge::ThreadPool;
using ge::parser::fp16_t;
using ge::parser::ModelSaver;

namespace ge {
graphStatus aclgrphParseTensorFlow(const char *model_file, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(model_file);
  GetParserContext().type = domi::TENSORFLOW;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::TENSORFLOW)));

  // load custom plugin so and proto
  AclGrphParseUtil acl_graph_parse_util;
  domi::Status status = acl_graph_parse_util.AclParserInitialize(options);
  if (status != domi::SUCCESS) {
    GELOGE(GRAPH_FAILED, "Parser Initialize failed.");
    return GRAPH_FAILED;
  }

  // Create an empty computegraph
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  if (compute_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "New ComputeGraph failed");
    GELOGE(FAILED, "Create ComputeGraph fail.");
    return FAILED;
  }

  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::TENSORFLOW);
  if (model_parser == nullptr) {
    REPORT_CALL_ERROR("E19999", "No Model Parser for tensorflow, check invalid");
    GELOGE(GRAPH_FAILED, "No Model Parser for tensorflow, check invalid");
    return FAILED;
  }

  // parse tensorflow model_file to GE graph
  ge::graphStatus ret = model_parser->Parse(model_file, graph);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "Parser graph %s failed.", graph.GetName().c_str());
    return ge::FAILED;
  }

  std::map<AscendString, AscendString> parser_params;
  if (acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params) != ge::SUCCESS) {
    GELOGE(ret, "Set graph %s default output node failed.", graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("Parser graph %s success.", graph.GetName().c_str());
  return ge::SUCCESS;
}

graphStatus aclgrphParseTensorFlow(const char *model_file, const std::map<AscendString, AscendString> &parser_params,
                                   ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(model_file);
  GetParserContext().type = domi::TENSORFLOW;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::TENSORFLOW)));

  // load custom plugin so and proto
  AclGrphParseUtil acl_graph_parse_util;
  domi::Status status = acl_graph_parse_util.AclParserInitialize(options);
  if (status != domi::SUCCESS) {
    GELOGE(GRAPH_FAILED, "Parser Initialize failed.");
    return GRAPH_FAILED;
  }

  string output_name;
  if (acl_graph_parse_util.ParseParamsBeforeGraph(parser_params, output_name) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Parser params before graph failed.");
    return ge::FAILED;
  }
  // Create an empty computegraph
  string graph_name = output_name.empty() ? "tmpGraph" : output_name;
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>(graph_name);
  if (compute_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "New ComputeGraph failed");
    GELOGE(FAILED, "Create ComputeGraph fail.");
    return FAILED;
  }

  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::TENSORFLOW);
  if (model_parser == nullptr) {
    REPORT_CALL_ERROR("E19999", "No Model Parser for tensorflow, check invalid");
    GELOGE(GRAPH_FAILED, "No Model Parser for tensorflow, check invalid");
    return FAILED;
  }

  // parse tensorflow model_file to GE graph
  ge::graphStatus ret = model_parser->Parse(model_file, graph);
  if (ret != ge::SUCCESS) {
    GELOGE(ret, "Parser graph %s failed.", graph.GetName().c_str());
    return ge::FAILED;
  }

  if (acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Parser params after graph failed.");
    return ge::FAILED;
  }

  if (acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Set graph %s default output node failed.", graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("AclgrphParse graph %s success.", graph.GetName().c_str());
  return ge::SUCCESS;
}
}  // namespace ge

namespace ge {
namespace {
const int kTransposeInputIdx = 0;
const uint32_t kThreadNum = 16;
const size_t kInputNumUint = 2;
const int kInputNumInt = 2;
const int32_t kControlSlot = -1;
const size_t kSoftmaxMultiple = 2;
const set<string> kTfBlackFields = {"tensor_content"};
const std::vector<std::string> kSkipCheckoutInputSizeNodes = {ge::parser::DATA, ge::parser::VARIABLE,
                                                              ge::parser::FRAMEWORKOP, ge::parser::LAYERNORM};
const std::vector<std::string> kMakeOperatorNotByIr = {ge::parser::ARG, ge::parser::VARIABLE, ge::parser::VARHANDLEOP,
                                                       ge::parser::FRAMEWORKOP, ge::parser::DATA};
const char *const kDpop = "DPOP";
const char *const kFuncDefLibraryFilePath = "graph_def_library.pbtxt";
const char *const kAttrNameIsScopeInnerNode = "_is_scope_inner_node";
struct ParseArg {
  const google::protobuf::Message *proto;
  std::string function_name;
  ge::NodePtr parent_node;
  std::string subgraph_name;
  ge::ComputeGraphPtr graph;
};

Status GenSubgraphParseTasks(const ge::ComputeGraphPtr &parent_graph, std::deque<ParseArg> &args) {
  GELOGI("Gen subgraph parse tasks start");
  for (auto &node : parent_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto subgraph_name_to_index : op_desc->GetSubgraphNameIndexes()) {
      auto i = subgraph_name_to_index.second;
      auto subgraph_iname = op_desc->GetSubgraphInstanceName(i);
      if (subgraph_iname.empty()) {
        GELOGW("The subgraph index %u of node %s is empty", i, node->GetName().c_str());
        continue;
      }

      // A function may be referenced multiple times in TF, change the graph name to ensure it is unique in GE
      auto unique_name = node->GetName() + std::to_string(i) + subgraph_iname;
      auto subgraph = ge::parser::MakeShared<ge::ComputeGraph>(unique_name);
      if (subgraph == nullptr) {
        REPORT_CALL_ERROR("E19999", "New ComputeGraph failed when create subgraph:%s", subgraph_iname.c_str());
        GELOGE(OUT_OF_MEMORY, "Failed to alloc subgraph %s", subgraph_iname.c_str());
        return OUT_OF_MEMORY;
      }
      auto ret = ge::NodeUtils::SetSubgraph(*node, i, subgraph);
      if (ret != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Set subgraph:%s to node:%s(%s) failed, index:%u",
                          subgraph_iname.c_str(), node->GetName().c_str(), node->GetType().c_str(), i);
        GELOGE(ret, "Failed to set subgraph %s to node %s index %u", subgraph_iname.c_str(), node->GetName().c_str(),
               i);
        return ret;
      }

      GELOGD("Add subgraph parse task to the queue, node %s, index %u, subgraph instance name %s",
             node->GetName().c_str(), i, subgraph_iname.c_str());
      args.push_back({nullptr, subgraph_iname, node, subgraph_name_to_index.first, subgraph});
    }
  }
  GELOGI("Gen subgraph parse tasks end");
  return SUCCESS;
}

Status PostOpProcessForSubgraph(const ParseArg &arg) {
  if (arg.parent_node == nullptr) {
    return SUCCESS;
  }
  std::string op_type = arg.parent_node->GetType();
  std::string op_name = arg.parent_node->GetName();
  domi::ParseSubgraphFuncV2 parse_func_v2 = nullptr;
  auto post_func = domi::OpRegistry::Instance()->GetParseSubgraphPostFunc(op_type);
  if (post_func == nullptr) {
    GELOGW("The subgraph post func for node %s type %s is null", op_name.c_str(), op_type.c_str());
    if (domi::OpRegistry::Instance()->GetParseSubgraphPostFunc(op_type, parse_func_v2) != SUCCESS ||
        parse_func_v2 == nullptr) {
      GELOGW("The subgraph post func v2 for node %s type %s is null", op_name.c_str(), op_type.c_str());
      return SUCCESS;
    }
  }

  GELOGD("Post process for subgraph %s node %s type %s subgraph name %s", arg.function_name.c_str(),
         arg.parent_node->GetName().c_str(), arg.parent_node->GetType().c_str(), arg.subgraph_name.c_str());

  // refresh node_name in subgraph
  for (const ge::NodePtr &node : arg.graph->GetDirectNode()) {
    if ((node->GetOpDesc() == nullptr) || (node->GetType() == "Variable") || (node->GetType() == "VariableV2")) {
      continue;
    }
    node->GetOpDesc()->SetName(node->GetOwnerComputeGraph()->GetName() + "/" + node->GetName());
  }

  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(arg.graph);
  Status ret = FAILED;
  if (post_func != nullptr) {
    ret = post_func(arg.subgraph_name, graph);
  } else if (parse_func_v2 != nullptr) {
    ret = parse_func_v2(arg.subgraph_name.c_str(), graph);
  }
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call ParseSubgraphPostFunc:%s failed, subgraph:%s, node:%s(%s), ret:0x%X",
                      arg.function_name.c_str(), arg.subgraph_name.c_str(),
                      arg.parent_node->GetName().c_str(), arg.parent_node->GetType().c_str(), ret);
    GELOGE(FAILED, "Failed to post-process subgraph %s on node %s type %s subgraph name %s", arg.function_name.c_str(),
           arg.parent_node->GetName().c_str(), arg.parent_node->GetType().c_str(), arg.subgraph_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace

/**
 * @ingroup domi_omg
 * @brief Trans common decorate function to PartitionedCall.
 * @param [in] node_def:  Node of common function.
 * @param [out] op: result of PartitionedCall OpDesc.
 * @return 0: SUCCESS / Others: FAILED
 */
Status TensorFlowModelParser::DefunToPartitionedCall(const domi::tensorflow::NodeDef *node_def, ge::OpDescPtr &op) {
  const string op_name = node_def->name();
  domi::tensorflow::AttrValue attr_call_inference;
  if (!ge::TensorFlowUtil::FindAttrValue(node_def, "_disable_call_shape_inference", attr_call_inference)) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E19014", {"opname", "value", "reason"},
        {node_def->name(), "attr [_disable_call_shape_inference]",
         "may has no ir definition, if it is not a common decorate function operator"});
    GELOGE(FAILED,
           "Op %s has no ir definition, or has no attr [_disable_call_shape_inference] "
           "if it is a common decorate function operator.", op_name.c_str());
    return FAILED;
  }

  op = ge::parser::MakeShared<ge::OpDesc>(op_name, ge::parser::PARTITIONEDCALL);
  GE_CHECK_NOTNULL(op);

  size_t input_tensor_num = 0;
  size_t output_tensor_num = 0;
  GetInputOutputTensorNum(op, input_tensor_num, output_tensor_num);

  for (size_t i = 0; i < input_tensor_num; ++i) {
    ge::GeTensorDesc input_tensor;
    if (op->AddInputDesc(input_tensor) != ge::GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                        op->GetName().c_str(), op->GetType().c_str());
      GELOGE(FAILED, "op [%s] type[%s] add input(%zu) tensor failed.", op_name.c_str(), op->GetType().c_str(), i);
      return FAILED;
    }
  }

  for (size_t i = 0; i < output_tensor_num; ++i) {
    ge::GeTensorDesc output_tensor;
    if (op->AddOutputDesc(output_tensor) != ge::GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                        op->GetName().c_str(), op->GetType().c_str());
      GELOGE(FAILED, "op [%s] type[%s] add output(%zu) tensor failed.", op_name.c_str(), op->GetType().c_str(), i);
      return FAILED;
    }
  }

  GELOGI("After AddTensorDescToOpDesc op[%s]: type[%s] have input size: %zu, output size: %zu, disable inference: %d",
         op_name.c_str(), op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize(), attr_call_inference.b());

  (void)op->AddSubgraphName("f");
  (void)op->SetSubgraphInstanceName(0, op_name);
  return SUCCESS;
}

Status TensorFlowModelParser::TransNodeToOpDesc(const domi::tensorflow::NodeDef *node_def, ge::OpDescPtr &op,
                                                const string &op_type) {
  GE_CHECK_NOTNULL(node_def);
  string node_name = node_def->name();
  ge::Operator op_factory = ge::OperatorFactory::CreateOperator(node_name, op_type);
  if (op_factory.GetName() != node_name || op_type == ge::parser::DATA) {
    if (std::find(kMakeOperatorNotByIr.begin(), kMakeOperatorNotByIr.end(), op_type) != kMakeOperatorNotByIr.end()) {
      op = ge::parser::MakeShared<ge::OpDesc>(node_name, op_type);
      GE_CHECK_NOTNULL(op);
    } else if (node_name == op_type) {
      // Trans @tensorflow.python.framework.Defun(...) to PartitionedCall.
      GE_RETURN_IF_ERROR(DefunToPartitionedCall(node_def, op));
      GE_CHECK_NOTNULL(op);
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage("E10501", {"opname", "optype"}, {node_name, op_type});
      GELOGE(INTERNAL_ERROR, "IR for op[%s] optype[%s] is not registered.", node_name.c_str(), op_type.c_str());
      return FAILED;
    }
  } else {
    op = ge::OpDescUtils::GetOpDescFromOperator(op_factory);
    GE_CHECK_NOTNULL(op);
    GELOGI("After GetOpDescFromOperator op[%s]: type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
           op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());

    GE_RETURN_IF_ERROR(AddTensorDescToOpDesc(op, node_def));
    GELOGI("After AddTensorDescToOpDesc op[%s]: type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
           op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());
  }
  op_factory.BreakConnect();
  return SUCCESS;
}

Status TensorFlowModelParser::ParseOpParams(const domi::tensorflow::NodeDef *node_def, ge::OpDescPtr &op,
                                            shared_ptr<OpParser> &op_parser) {
  GE_CHECK_NOTNULL(node_def);
  GE_CHECK_NOTNULL(op);
  GE_CHECK_NOTNULL(op_parser);

  string node_name = node_def->name();
  string node_op = node_def->op();

  Status status = FAILED;
  domi::ParseParamByOpFunc parse_param_by_op_fn = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(node_op);
  if (parse_param_by_op_fn == nullptr) {
    shared_ptr<TensorFlowOpParser> tensorflow_op_parser = std::dynamic_pointer_cast<TensorFlowOpParser>(op_parser);
    GE_CHECK_NOTNULL(tensorflow_op_parser);
    status = tensorflow_op_parser->ParseParams(node_def, op);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for node[%s] failed", node_name.c_str());
      return status;
    }
  } else {
    ge::Operator op_src(node_def->name(), node_def->op());
    status = domi::AutoMappingFn(node_def, op_src);
    if (status != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Auto mapping node_def:%s(%s) to operator failed",
                        node_def->name().c_str(), node_def->op().c_str());
      GELOGE(status, "Node[%s] auto mapping failed.", node_name.c_str());
      return status;
    }
    std::shared_ptr<ge::TensorFlowCustomParserAdapter> tf_custom_op_parser =
        std::dynamic_pointer_cast<ge::TensorFlowCustomParserAdapter>(op_parser);
    GE_CHECK_NOTNULL(tf_custom_op_parser);
    status = tf_custom_op_parser->ParseParams(op_src, op);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for node[%s] failed", op_src.GetName().c_str());
      return status;
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::AddNode(const domi::tensorflow::NodeDef *node_def, ge::ComputeGraphPtr &graph,
                                      shared_ptr<ge::ScopeGraph> &scope_graph) {
  GE_CHECK_NOTNULL(node_def);
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(scope_graph);
  domi::tensorflow::AttrValue attr_value;
  if (ge::TensorFlowUtil::FindAttrValue(node_def, kAttrNameIsScopeInnerNode, attr_value) && attr_value.b()) {
    std::mutex graph_mutex;
    return AddScopeInnerNode(this, graph, &graph_mutex, node_def);
  }
  // node is released in destructor
  string node_name = node_def->name();
  string node_op = node_def->op();
  auto type_it = tensorflow_op_map.find(node_op);
  if (type_it == tensorflow_op_map.end()) {
    GELOGI("Can not find,maybe this node has no plugin node_name is %s, node_op is %s ", node_name.c_str(),
           node_op.c_str());
    ge::OpDescPtr op_desc;
    GE_RETURN_IF_ERROR(TransNodeToOpDesc(node_def, op_desc, node_op));

    ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    GE_CHK_STATUS(domi::AutoMappingFn(node_def, op));
    op.BreakConnect();

    ge::NodePtr node = nullptr;
    node = graph->AddNode(op_desc);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((node == nullptr), DeleteFuisonNodeDef(); return INTERNAL_ERROR, "add node failed.");

    node_map_[node_name] = node;

    return SUCCESS;
  }

  string op_type = type_it->second;

  // The type value is obtained from the definition map set of DaVinci.
  ge::OpDescPtr op;
  GE_RETURN_IF_ERROR(TransNodeToOpDesc(node_def, op, op_type));

  bool needFusion = IsFusionOp(scope_graph, node_def);
  // The number of inputs and outputs of each operator can be determined after the new IR design model is resolved.
  // Add tensordesc to the opdesc object of the operator
  // Process change of tensordesc initialization of opdesc,
  // Previous process: Tensordesc is constructed according to graph structure in builder stage
  // Current process: Tensordesc is determined before the opdesc of the operator is added to the graph
  Status status = FAILED;
  // create OpParser
  shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  GE_CHECK_NOTNULL(factory);
  if (!needFusion) {
    shared_ptr<OpParser> op_parser = factory->CreateOpParser(op_type);
    // parse op param
    status = ParseOpParams(node_def, op, op_parser);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for node[%s] failed", node_name.c_str());
      return status;
    }
  }
  GELOGI("After op parser op[%s] type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
         op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());
  // checkout op input number with IR
  GE_RETURN_IF_ERROR(CheckoutInputNum(op, node_def));
  ge::NodePtr node = graph->AddNode(op);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((node == nullptr), DeleteFuisonNodeDef(); return INTERNAL_ERROR, "add node failed.");

  node_map_[node_name] = node;

  if (needFusion) {
    shared_ptr<OpParser> fusion_op_parser = factory->CreateFusionOpParser(op_type);
    GE_CHECK_NOTNULL(fusion_op_parser);
    // Find all children of the fusion operator
    auto iter = fusion_op_nodedef_map_.find(node_def->name());
    if (iter == fusion_op_nodedef_map_.end()) {
      REPORT_INNER_ERROR("E19999", "FusionOp node %s has no children node, check invalid", node_name.c_str());
      GELOGE(FAILED, "FusionOp node %s has no children node!", node_name.c_str());
      return INTERNAL_ERROR;
    }
    vector<const domi::tensorflow::NodeDef *> node_def_v = iter->second;
    // parse fusion node param
    status = FusionNodeParseParams(fusion_op_parser, node_def, node);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for fusion node[%s] failed", node_name.c_str());
      return status;
    }
    // record original op names
    std::vector<std::string> namesTmp;
    for (auto &node_def_iter : node_def_v) {
      GE_CHECK_NOTNULL(node_def_iter);
      std::string nodeName = node_def_iter->name();
      namesTmp.push_back(nodeName);
    }

    ge::GraphUtils::RecordOriginalNames(namesTmp, node);
    status = RecordFusionResult(scope_graph, node_def, op);
    if (status != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Record fusion result for fusion op: %s failed", op->GetName().c_str());
      DeleteFuisonNodeDef();
      return status;
    }
  }
  return SUCCESS;
}

void TensorFlowModelParser::GetInputOutputTensorNum(ge::OpDescPtr &op_desc, size_t &input_tensor_num,
                                                    size_t &output_tensor_num) const {
  // The caller guarantees that the pointer is not null
  auto iter = op_node_context_map_.find(op_desc->GetName());
  if (iter == op_node_context_map_.end()) {
    return;
  }
  const OpNodeContext &op_context = iter->second;
  const std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> &dest_input_map = op_context.input_map;
  // input number
  input_tensor_num = 0;
  for (auto &input_vec : dest_input_map) {
    for (auto &input_v : input_vec.second) {
      if (input_v.second != kControlSlot) {
        input_tensor_num++;
      }
    }
  }

  // output number
  const std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> &src_output_map = op_context.output_map;
  int32_t max_anchor_index = 0;
  for (auto &src_output_iter : src_output_map) {
    for (auto &index_output_iter : src_output_iter.second) {
      if (index_output_iter.first > max_anchor_index) {
        max_anchor_index = index_output_iter.first;
      }
    }
  }
  output_tensor_num = max_anchor_index + 1;

  return;
}

Status TensorFlowModelParser::CheckoutInputNum(ge::OpDescPtr &op_desc, const domi::tensorflow::NodeDef *node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(op_desc);

  if (std::find(kSkipCheckoutInputSizeNodes.begin(), kSkipCheckoutInputSizeNodes.end(), op_desc->GetType()) !=
      kSkipCheckoutInputSizeNodes.end()) {
    return SUCCESS;
  }

  // get input and output tensor number
  size_t input_tensor_num = 0;
  size_t output_tensor_num = 0;
  GetInputOutputTensorNum(op_desc, input_tensor_num, output_tensor_num);

  // get input and output tensor number from op desc
  size_t factory_input_size = op_desc->GetInputsSize();
  if (input_tensor_num != factory_input_size) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E19014", {"opname", "value", "reason"},
        {op_desc->GetName(), "input number of tensorflow[" + std::to_string(input_tensor_num) + "]",
         "should be equal to factory size[" + std::to_string(factory_input_size) + "]"});
    GELOGE(FAILED, "op [%s], type[%s], The input number of tensorflow[%zu] should be equal to factory size[%zu]",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), input_tensor_num, factory_input_size);
    return FAILED;
  }
  return SUCCESS;
}

void TensorFlowModelParser::UpdateInputTensor(ge::OpDescPtr &op_desc, const std::vector<ge::GeTensorDesc> &input_desc,
                                              const size_t input_tensor_num) {
  // The caller guarantees that the pointer is not null
  for (size_t i = 0; i < input_tensor_num; ++i) {
    if (i < input_desc.size()) {
      // i is guaranteed to be valid, no check required.
      ge::graphStatus ret = op_desc->UpdateInputDesc(op_desc->GetInputNameByIndex(i), input_desc[i]);
      if (ret != ge::GRAPH_SUCCESS) {
        // UpdateInputDesc for dynamic intput will be failed, but it will be added in later op parser.
        GELOGI("op [%s], type[%s], input(%zu) with name %s is not updated", op_desc->GetName().c_str(),
               op_desc->GetType().c_str(), i, op_desc->GetInputNameByIndex(i).c_str());
      }
    } else {
      ge::GeTensorDesc input_tensor;
      // i is guaranteed to be valid, no check required.
      ge::graphStatus ret = op_desc->UpdateInputDesc(op_desc->GetInputNameByIndex(i), input_tensor);
      if (ret != ge::GRAPH_SUCCESS) {
        // UpdateInputDesc for dynamic intput will be failed, but it will be added in later op parser.
        GELOGI("op [%s], type[%s], input(%zu) with name %s is not updated", op_desc->GetName().c_str(),
               op_desc->GetType().c_str(), i, op_desc->GetInputNameByIndex(i).c_str());
      }
    }
  }
}

void TensorFlowModelParser::UpdateOutputTensor(ge::OpDescPtr &op_desc, const std::vector<ge::GeTensorDesc> &output_desc,
                                               size_t output_tensor_num) {
  // The caller guarantees that the pointer is not null
  for (size_t i = 0; i < output_tensor_num; ++i) {
    if (i < output_desc.size()) {
      // i is guaranteed to be valid, no check required.
      ge::graphStatus ret = op_desc->UpdateOutputDesc(op_desc->GetOutputNameByIndex(i), output_desc[i]);
      if (ret != ge::GRAPH_SUCCESS) {
        // UpdateOutputDesc for dynamic output will be failed, but it will be added in later op parser.
        GELOGI("op [%s], type[%s], output(%zu) with name %s is not updated", op_desc->GetName().c_str(),
               op_desc->GetType().c_str(), i, op_desc->GetInputNameByIndex(i).c_str());
      }
    } else {
      ge::GeTensorDesc output_tensor;
      // i is guaranteed to be valid, no check required.
      ge::graphStatus ret = op_desc->UpdateOutputDesc(op_desc->GetOutputNameByIndex(i), output_tensor);
      if (ret != ge::GRAPH_SUCCESS) {
        // UpdateOutputDesc for dynamic output will be failed, but it will be added in later op parser.
        GELOGI("op [%s], type[%s], output(%zu) with name %s is not updated", op_desc->GetName().c_str(),
               op_desc->GetType().c_str(), i, op_desc->GetInputNameByIndex(i).c_str());
      }
    }
  }
}

Status TensorFlowModelParser::AddTensorDescToOpDesc(ge::OpDescPtr &op_desc, const domi::tensorflow::NodeDef *node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(op_desc);
  // get input and output attr from tensorflow
  const string type = node->op();
  domi::tensorflow::AttrValue input_attr_value;
  domi::tensorflow::AttrValue output_attr_value;
  ParserOperator temp_op;
  if (ge::TensorFlowUtil::FindAttrValue(node, ge::parser::ATTR_NAME_INPUT_TENSOR_DESC, input_attr_value)) {
    GE_CHK_STATUS_RET(ge::TensorFlowUtil::TransTensorDescriptor(input_attr_value, &temp_op,
                                                                TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG, type),
                      "trans input_attr_value failed, op: %s", node->name().c_str());
  } else {
    GELOGD("Frameworkop has no input tensor desc, name:%s, type:%s.", node->name().c_str(), type.c_str());
  }
  if (ge::TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_attr_value)) {
    GE_CHK_STATUS_RET(ge::TensorFlowUtil::TransTensorDescriptor(output_attr_value, &temp_op,
                                                                TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG, type),
                      "trans output_attr_value failed, op: %s", node->name().c_str());
  } else {
    GELOGD("Frameworkop has no output tensor desc, name:%s, type:%s.", node->name().c_str(), type.c_str());
  }

  auto iter = op_node_context_map_.find(op_desc->GetName());
  if (iter == op_node_context_map_.end()) {
    return SUCCESS;
  }

  const std::vector<ge::GeTensorDesc> &input_desc = temp_op.GetInputTensorDesc();
  const std::vector<ge::GeTensorDesc> &output_desc = temp_op.GetOutputTensorDesc();

  // get input and output tensor number
  size_t input_tensor_num = 0;
  size_t output_tensor_num = 0;
  GetInputOutputTensorNum(op_desc, input_tensor_num, output_tensor_num);

  // update input
  UpdateInputTensor(op_desc, input_desc, input_tensor_num);

  // update output
  UpdateOutputTensor(op_desc, output_desc, output_tensor_num);

  return SUCCESS;
}

Status TensorFlowModelParser::AddEdges(ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &src_iter : op_node_context_map_) {
    string src_op_name = src_iter.first;
    OpNodeContext src_op_node_context = src_iter.second;
    std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> &src_output_map = src_op_node_context.output_map;
    // Traverse all output of the op_node
    for (auto &src_output_iter : src_output_map) {
      string dest_op_name = src_output_iter.first;
      auto dest_iter = op_node_context_map_.find(dest_op_name);
      if (dest_iter == op_node_context_map_.end()) {
        continue;
      }
      // Find that the output of the source node is equal to the destination node
      std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> &dest_input_map = dest_iter->second.input_map;
      auto input_iter = dest_input_map.find(src_op_name);
      // Find output and input
      if (input_iter == dest_input_map.end()) {
        continue;
      }
      auto iter = node_map_.find(src_op_name);
      if (iter == node_map_.end()) {
        continue;
      }
      ge::NodePtr src = iter->second;
      GE_CHECK_NOTNULL(src);
      auto iter1 = node_map_.find(dest_op_name);
      if (iter1 == node_map_.end()) {
        continue;
      }
      // Each pair builds an edge
      ge::NodePtr dest = iter1->second;
      GE_CHECK_NOTNULL(dest);
      if (src_output_iter.second.size() != input_iter->second.size()) {
        REPORT_INNER_ERROR("E19999", "Input size of op[%s]:%zu is not equal to Output size of op[%s]:%zu.",
                           src_op_name.c_str(), input_iter->second.size(),
                           dest_op_name.c_str(), src_output_iter.second.size());
        GELOGE(INTERNAL_ERROR, "Input size of op[%s]:%zu is not equal to Output size of op[%s]:%zu.",
               src_op_name.c_str(), input_iter->second.size(), dest_op_name.c_str(), src_output_iter.second.size());
        return INTERNAL_ERROR;
      }
      for (auto &outputpair : src_output_iter.second) {
        // Get control edge properties
        bool control = GetEdgesControlInfo(dest_op_name, outputpair.second);
        // Graph create new edge
        if (!control) {
          GELOGD("Start add edge: from %s:%d to %s:%d.", src->GetName().c_str(), outputpair.first,
                 dest->GetName().c_str(), outputpair.second);
          ge::OutDataAnchorPtr out_archor_ptr = src->GetOutDataAnchor(outputpair.first);
          GE_CHECK_NOTNULL(out_archor_ptr);
          ge::InDataAnchorPtr in_archor_ptr = dest->GetInDataAnchor(outputpair.second);
          GE_CHECK_NOTNULL(in_archor_ptr);
          GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ge::GraphUtils::AddEdge(out_archor_ptr, in_archor_ptr) != ge::GRAPH_SUCCESS,
                                         REPORT_INNER_ERROR("E19999", "Add link from op:%s to op:%s failed",
                                                            src->GetName().c_str(), dest->GetName().c_str());
                                         return INTERNAL_ERROR, "Add link failed from op[%s] to op[%s].",
                                                src->GetName().c_str(), dest->GetName().c_str());
        } else {
          GELOGD("Start add contorl edge: from %s to %s.", src->GetName().c_str(), dest->GetName().c_str());
          ge::InControlAnchorPtr in_archor_ptr = dest->GetInControlAnchor();
          GE_CHECK_NOTNULL(in_archor_ptr);
          ge::OutControlAnchorPtr out_archor_ptr = src->GetOutControlAnchor();
          GE_CHECK_NOTNULL(out_archor_ptr);
          GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
              ge::GraphUtils::AddEdge(out_archor_ptr, in_archor_ptr) != ge::GRAPH_SUCCESS,
              REPORT_INNER_ERROR("E19999", "Add link from op:%s to op:%s failed",
                                 src->GetName().c_str(), dest->GetName().c_str());
              return INTERNAL_ERROR, "Add link failed from op[%s] to op[%s].", src->GetName().c_str(),
                     dest->GetName().c_str()
          );
        }
      }
      dest_input_map.erase(input_iter);
    }
  }

  return SUCCESS;
}

Status TensorFlowModelParser::AddFmkNodeDefToMap(const domi::tensorflow::GraphDef &graph_def,
                                                 const domi::tensorflow::NodeDef *node_def,
                                                 vector<string> &op_node_name_list) {
  GE_CHECK_NOTNULL(node_def);
  const string &node_name = node_def->name();

  nodedef_map_[node_name] = node_def;

  OpNodeContext op_node_context;
  op_node_context_map_[node_name] = op_node_context;
  op_node_name_list.push_back(node_name);

  return SUCCESS;
}

Status TensorFlowModelParser::CheckOpShapeDim(const domi::tensorflow::NodeDef *node_def, const std::set<int> &dims,
                                              bool &valid) {
  GE_CHECK_NOTNULL(node_def);
  domi::tensorflow::AttrValue input_attr_value;
  bool is_attr_exist =
      ge::TensorFlowUtil::FindAttrValue(node_def, ge::parser::ATTR_NAME_INPUT_TENSOR_DESC, input_attr_value);
  GE_IF_BOOL_EXEC(!is_attr_exist, return SUCCESS);
  GE_CHK_BOOL_EXEC(input_attr_value.has_list(),
                   REPORT_INNER_ERROR("E19999", "Attr:%s of node_def:%s(%s) is empty, check invalid",
                                      ge::parser::ATTR_NAME_INPUT_TENSOR_DESC.c_str(),
                                      node_def->name().c_str(), node_def->op().c_str());
                   return PARAM_INVALID, "output attr value vector is empty");

  // list contain many TensorDescriptors
  domi::tensorflow::AttrValue_ListValue a_list = input_attr_value.list();
  for (int32_t i = 0; i < a_list.func_size(); i++) {
    ge::GeTensorDesc ge_desc;
    int32_t tf_datatype = 0;
    GE_CHK_BOOL_RET_STATUS(ge::TensorFlowUtil::ParseFromAttrValueList(ge_desc, a_list, i, tf_datatype), PARAM_INVALID,
                           "parse ge_desc failed.");

    for (uint32_t j = 0; j < ge_desc.GetShape().GetDimNum(); ++j) {
      int64_t temp_dim = ge_desc.GetShape().GetDim(j);
      GE_IF_BOOL_EXEC(dims.count(temp_dim) > 0, valid = false);
    }
  }

  return SUCCESS;
}

Status TensorFlowModelParser::CheckOpType(const domi::tensorflow::NodeDef *node_def, string &op_type) {
  GE_CHECK_NOTNULL(node_def);
  bool valid = true;
  string node_name = node_def->name();

  std::map<std::string, set<int>> check_dims = {
      {ge::parser::SPARSESOFTMAXCROSSENTROPYWITHLOGITS, {10}},
  };

  GE_IF_BOOL_EXEC(
      op_type == ge::parser::SPARSESOFTMAXCROSSENTROPYWITHLOGITS,
      GE_CHK_STATUS_RET(CheckOpShapeDim(node_def, check_dims[op_type], valid), "failed to check op shape");
      GE_IF_BOOL_EXEC(!valid, op_type = ge::parser::FRAMEWORKOP; GELOGI("Set op %s to frameworkop", node_name.c_str());
                      framework_ops_[node_name] = node_def;););

  GE_IF_BOOL_EXEC(
      op_type == ge::parser::ADD || op_type == ge::parser::MULTIPLY || op_type == ge::parser::MEAN,
      for (const string &input_name
           : node_def->input()) {
        string tmp_input_name;
        GE_RETURN_IF_ERROR(CheckInputNodeName(input_name, &tmp_input_name, nullptr, nullptr));
        GELOGD("Add or Mul op %s input name is %s", node_name.c_str(), input_name.c_str());
        GE_IF_BOOL_EXEC(framework_ops_.find(tmp_input_name) != framework_ops_.end(),
                        GELOGI("Set op %s to frameworkop", node_name.c_str());
                        op_type = ge::parser::FRAMEWORKOP;);
      });

  return SUCCESS;
}

/*
 * @ingroup domi_omg
 * @brief Mapping TF's datatype to GE's datatype
 * @param [in] type, datatype types of operators in TF networks
 * @return ge::DataType
 */
ge::DataType TensorFlowModelParser::ConvertToGeDataType(const uint32_t type) {
  ErrorManager::GetInstance().GenWorkStreamIdDefault();

  ge::DataType data_type = domi::TensorAssign::ConvertTensorflowDataType(type);
  return data_type;
}

Status TensorFlowModelParser::ParseNodeDef(TensorFlowModelParser *parser, ge::ComputeGraphPtr &graph,
                                           std::mutex *graphMutex, shared_ptr<ge::ScopeGraph> &scope_graph,
                                           const domi::tensorflow::NodeDef *node_def,
                                           error_message::Context error_context) {
  ErrorManager::GetInstance().SetErrorContext(error_context);
  // The caller guarantees that the pointer is not null
  string node_name = node_def->name();
  string node_op = node_def->op();
  GELOGD("TF op node name = %s, op type= %s", node_name.c_str(), node_op.c_str());
  domi::tensorflow::AttrValue attr_value;
  if (ge::TensorFlowUtil::FindAttrValue(node_def, kAttrNameIsScopeInnerNode, attr_value) && attr_value.b()) {
    return AddScopeInnerNode(parser, graph, graphMutex, node_def);
  }

  auto iterator = parser->adaptedOpTypeMap_.find(node_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(iterator == parser->adaptedOpTypeMap_.end(),
                                 REPORT_INNER_ERROR("E19999", "get adapted op type failed, node name = %s",
                                                    node_name.c_str());
                                 return FAILED,
                                 "get adapted op type failed, node name = %s", node_name.c_str());

  string op_type = iterator->second;
  // Log printing for determining operator type
  domi::ImplyType implyType = domi::OpRegistry::Instance()->GetImplyType(op_type);
  GE_IF_BOOL_EXEC((implyType == domi::ImplyType::TVM) && (op_type != ge::parser::FRAMEWORKOP),
                  GELOGD("TBE %s parsering", node_op.c_str()););
  GE_IF_BOOL_EXEC((implyType == domi::ImplyType::CCE) && (op_type != ge::parser::FRAMEWORKOP),
                  GELOGD("CCE %s parsering", node_op.c_str()););
  GE_IF_BOOL_EXEC((implyType == domi::ImplyType::HCCL) && (op_type != ge::parser::FRAMEWORKOP),
                  GELOGD("HCCL %s parsering", node_op.c_str()););
  GE_IF_BOOL_EXEC(op_type == ge::parser::FRAMEWORKOP, GELOGD("FRAMEWORKOP %s parsering", node_op.c_str()););
  GELOGD("TF op node name = %s, op type= %s, trans to op type %s", node_name.c_str(), node_op.c_str(), op_type.c_str());

  // Construct operator by IR
  ge::OpDescPtr op;
  ge::Operator op_factory = ge::OperatorFactory::CreateOperator(node_name, op_type);
  if (op_factory.GetName() != node_name) {
    if (std::find(kMakeOperatorNotByIr.begin(), kMakeOperatorNotByIr.end(), op_type) != kMakeOperatorNotByIr.end()) {
      op = ge::parser::MakeShared<ge::OpDesc>(node_name, op_type);
      GE_CHECK_NOTNULL(op);
    } else if (node_name == op_type) {
      GE_RETURN_IF_ERROR(parser->DefunToPartitionedCall(node_def, op));
      GE_CHECK_NOTNULL(op);
      ge::Operator op_tmp = ge::OpDescUtils::CreateOperatorFromOpDesc(op);
      GE_CHK_STATUS(domi::AutoMappingFn(node_def, op_tmp));
      op_tmp.BreakConnect();
      ge::NodePtr node;
      {
        std::lock_guard<std::mutex> lock(*graphMutex);
        node = graph->AddNode(op);
      }
      GE_CHECK_NOTNULL(node);
      {
        std::lock_guard<std::mutex> lock(parser->nodeMapMutex_);
        parser->node_map_[node_name] = node;
      }
      return SUCCESS;
    } else {
      REPORT_INPUT_ERROR("E10501", std::vector<std::string>({"opname", "optype"}),
                         std::vector<std::string>({node_name, op_type}));
      GELOGE(INTERNAL_ERROR, "op[%s] type[%s] have no ir factory.]", node_name.c_str(), op_type.c_str());
      return FAILED;
    }
  } else {
    op = ge::OpDescUtils::GetOpDescFromOperator(op_factory);
    GE_CHECK_NOTNULL(op);
    GELOGD("After GetOpDescFromOperator op[%s] type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
           op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());

    GE_RETURN_IF_ERROR(parser->AddTensorDescToOpDesc(op, node_def));
    GELOGD("After AddTensorDescToOpDesc op[%s] type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
           op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());
  }
  GELOGD("TF op node name = %s, outpusize= %zu", node_name.c_str(), op->GetAllOutputsDesc().size());
  op_factory.BreakConnect();

  // create OpParser
  shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
  GE_CHECK_NOTNULL(factory);
  bool needFusion = parser->IsFusionOp(scope_graph, node_def);
  GELOGD("TF op node name = %s, op type= %s is fusion op(NO: 0; YES: 1)= %d", node_name.c_str(), node_op.c_str(),
         needFusion);

  Status status = FAILED;
  if (!needFusion) {
    shared_ptr<OpParser> op_parser = factory->CreateOpParser(op_type);
    status = parser->ParseOpParams(node_def, op, op_parser);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for node[%s] failed", node_name.c_str());
      return status;
    }
  }
  GELOGD("After op parser op[%s] type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
         op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());

  // checkout op input number with IR
  GE_RETURN_IF_ERROR(parser->CheckoutInputNum(op, node_def));

  if (needFusion) {
    status = RecordFusionResult(scope_graph, node_def, op);
    if (status != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "Record fusion result for fusion op: %s failed", op->GetName().c_str());
      return status;
    }
  }

  ge::NodePtr node;
  {
    std::lock_guard<std::mutex> lock(*graphMutex);
    node = graph->AddNode(op);
  }

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((node == nullptr),
                                 REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                                   op->GetName().c_str(), op->GetType().c_str(),
                                                   graph->GetName().c_str());
                                 return INTERNAL_ERROR, "add node failed.");

  if (needFusion) {
    shared_ptr<OpParser> fusion_op_parser = factory->CreateFusionOpParser(op_type);
    status = parser->FusionNodeParseParams(fusion_op_parser, node_def, node);
    GE_CHK_STATUS_EXEC(status, return status, "Parse Params for node %s failed", node_name.c_str());
  }

  {
    std::lock_guard<std::mutex> lock(parser->nodeMapMutex_);
    parser->node_map_[node_name] = node;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::AdaptOpType(const domi::tensorflow::NodeDef *node_def, bool isDatasetInit) {
  // The caller guarantees that the pointer is not null
  string node_name = node_def->name();
  string node_op = node_def->op();
  string op_type;
  if (tensorflow_train_op_map.find(node_op) != tensorflow_train_op_map.end()) {
    op_type = tensorflow_train_op_map.at(node_op);
    GE_CHK_STATUS_RET(CheckOpType(node_def, op_type), "Failed to check op type");
  } else {
    op_type = ge::parser::FRAMEWORKOP;
    domi::tensorflow::AttrValue attr_call_inference;
    if ((node_name == node_op) &&
        ge::TensorFlowUtil::FindAttrValue(node_def, "_disable_call_shape_inference", attr_call_inference)) {
      op_type = node_op;
    }
  }

  GE_IF_BOOL_EXEC(isDatasetInit, op_type = ge::parser::FRAMEWORKOP);
  adaptedOpTypeMap_[node_name] = op_type;

  return SUCCESS;
}

Status TensorFlowModelParser::AddFmkNode(ge::ComputeGraphPtr &graph, shared_ptr<ge::ScopeGraph> &scope_graph,
                                         vector<string> &op_node_name_list, bool is_dataset_init) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(scope_graph);

  GE_RETURN_IF_ERROR(AddFusionNodeDef(scope_graph, op_node_name_list));
  size_t op_node_list_size = op_node_name_list.size();
  for (size_t i = 0; i < op_node_list_size; ++i) {
    const string op_node_name = op_node_name_list[i];
    const domi::tensorflow::NodeDef *node_def = nodedef_map_[op_node_name];
    GE_CHECK_NOTNULL(node_def);
    GE_RETURN_IF_ERROR(AdaptOpType(node_def, is_dataset_init));
  }
  GELOGD("Add fusion nodedef and Adapt op type success");

  // Multithreading parallel parsing nodedef
  ThreadPool executor(kThreadNum);
  std::mutex graphMutex;
  std::vector<std::future<Status>> vectorFuture(op_node_list_size);
  ge::ComputeGraphPtr graph_tmp = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  GE_CHECK_NOTNULL(graph_tmp);
  for (size_t j = 0; j < op_node_list_size; j++) {
    const string op_node_name = op_node_name_list[j];
    const domi::tensorflow::NodeDef *node_def = nodedef_map_[op_node_name];
    GE_CHECK_NOTNULL(node_def);
    std::future<Status> f =
        executor.commit(TensorFlowModelParser::ParseNodeDef, this, graph_tmp, &graphMutex, scope_graph, node_def,
                        ErrorManager::GetInstance().GetErrorManagerContext());
    if (!f.valid()) {
      GELOGE(FAILED, "Future is invalid");
      return FAILED;
    }
    vectorFuture[j] = std::move(f);
  }
  GELOGD("Parse nodedef success");
  // Wait for the return value of each thread. If the thread does not finish processing, it will block here
  bool ret_flag = true;
  size_t futureSize = vectorFuture.size();
  for (size_t i = 0; i < futureSize; ++i) {
    Status retStatus = vectorFuture[i].get();
    if (retStatus != SUCCESS) {
      ret_flag = false;
    }
  }
  if (!ret_flag) {
    return FAILED;
  }
  return AddNodeToGraphAndMarkFormat(graph, op_node_name_list);
}

Status TensorFlowModelParser::AddNodeToGraphAndMarkFormat(ge::ComputeGraphPtr &graph,
                                                          const vector<string> &op_node_name_list) {
  // Add ge:: nodeptr to graph in order
  size_t op_node_list_size = op_node_name_list.size();
  for (size_t j = 0; j < op_node_list_size; j++) {
    const string op_node_name = op_node_name_list[j];
    auto iterator = node_map_.find(op_node_name);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((iterator == node_map_.end()),
                                   REPORT_INNER_ERROR("E19999", "node:%s can't find in node_map_, check invalid",
                                                      op_node_name.c_str());
                                   return INTERNAL_ERROR, "add node failed.");
    GE_CHECK_NOTNULL(iterator->second);
    GE_CHK_STATUS_RET(iterator->second->SetOwnerComputeGraph(graph), "set owner compute graph failed");
    graph->AddNode(iterator->second);
  }

  return SUCCESS;
}

Status TensorFlowModelParser::ExcuteScopeFusionPasses(domi::tensorflow::GraphDef *graph_def,
                                                      shared_ptr<ge::ScopeGraph> &scope_graph) {
  // Identifying scope fusion operators based on scope rules
  GE_CHECK_NOTNULL(graph_def);
  ScopePassManager passmanager;
  PARSER_TIMESTAMP_START(BuildScopeGraph);
  scope_graph = passmanager.BuildScopeGraph(graph_def);
  GE_CHECK_NOTNULL(scope_graph);
  PARSER_TIMESTAMP_END(BuildScopeGraph, "TensorFlowModelParser::BuildScopeGraph");
  PARSER_TIMESTAMP_START(ScopeGraphPass);
  // Validate the non-general scope fusion pass.
  // The parameter is set to the name of the fusion rule.
  // Multiple names can be set and separated by ",".
  std::vector<std::string> enable_pass_names =
      ge::StringUtils::Split(ge::GetParserContext().enable_scope_fusion_passes, ',');
  auto &impl = ge::ScopeFusionPassRegistry::GetInstance().impl_;
  if (impl == nullptr) {
    REPORT_INNER_ERROR("E19999", "ScopeFusionPassRegistry is not properly initialized.");
    GELOGE(ge::MEMALLOC_FAILED, "ScopeFusionPassRegistry is not properly initialized.");
    return ge::MEMALLOC_FAILED;
  }

  for (size_t i = 0; i < enable_pass_names.size(); ++i) {
    if (enable_pass_names[i].empty()) {
      continue;
    }
    if (!impl->SetPassEnableFlag(enable_pass_names[i], true)) {
      GELOGW("Failed to set enable flag of scope fusion pass:%s", enable_pass_names[i].c_str());
    }
  }
  std::vector<std::string> scope_passes_list = impl->GetAllRegisteredPasses();
  Status ret = RunScopeFusionPass(scope_passes_list, passmanager, scope_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Run scope fusion failed, ret:%u.", ret);
    return ret;
  }
  PARSER_TIMESTAMP_END(ScopeGraphPass, "TensorFlowModelParser::ScopeGraphPass");

  return SUCCESS;
}

Status TensorFlowModelParser::ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(data);
  GE_CHECK_NOTNULL(graph);

  // Store objects parsed from pb files
  domi::tensorflow::GraphDef OriDef;

  bool read = ge::parser::ReadProtoFromArray(data, static_cast<int>(size), &OriDef);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(!read,
                                 REPORT_INNER_ERROR("E19999", "read graph proto from binary failed");
                                 return INTERNAL_ERROR, "read_proto_from_binary failed.");

  domi::tensorflow::GraphDef graph_def;
  if (ge::GetParserContext().input_dims.empty() && ge::GetParserContext().out_nodes_map.empty()) {
    graph_def = OriDef;
  } else {
    GELOGI("Before Trim, the Graph Node size is:%d", OriDef.node_size());
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(TrimGraph(OriDef, &graph_def), return INTERNAL_ERROR, "Trim Graph fail.");
    GELOGI("After Trim, The graph_def.node_size():%d", graph_def.node_size());
  }

  GE_RETURN_WITH_LOG_IF_ERROR(ProtoTypePassManager::Instance().Run(&graph_def, domi::TENSORFLOW),
                              "Run ProtoType Pass Failed");

  shared_ptr<ge::ScopeGraph> scope_graph = nullptr;
  Status ret = ExcuteScopeFusionPasses(&graph_def, scope_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[TF ParseFromMemory] scope fusion failed.");
    return ret;
  }
  GELOGD("[TF ParseFromMemory] scope fusion success");

  // Add nodedef in the model to prechecker and check the general parameters
  for (int i = 0; i < graph_def.node_size(); i++) {
    const domi::tensorflow::NodeDef &node = graph_def.node(i);
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().AddOp(&node, node.name(), node.op()),
                                "Add node_def to PreChecker failed, node name: %s.", node.name().c_str());
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().CheckName(&node), "Check node_def name failed, node name: %s.",
                                node.name().c_str());
    if (node.op() != TENSORFLOWF_NODE_OP_IDENTITY) {
      GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().CheckType(&node, true),
                                  "Check node_def type failed, node name: %s.", node.name().c_str());
    }
  }

  bool has_error = false;
  // save node name
  vector<string> op_node_name_list;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const domi::tensorflow::NodeDef *node_def = graph_def.mutable_node(i);

    // If it is a fusion operator, put nodedef in the fusion_op_nodedef_map_
    GE_IF_BOOL_EXEC(MaybeFusionOp(scope_graph, node_def),
                    GELOGI("Node: %s maybe a fusion op.", node_def->name().c_str()););

    // Do not exit immediately when there is an error, wait until all errors are collected before exiting
    GE_CHK_STATUS_EXEC(AddFmkNodeDefToMap(graph_def, node_def, op_node_name_list), has_error = true,
                       "add node failed.");
  }

  // Verify the validity of fusionop
  GE_RETURN_IF_ERROR(CheckFusionOpValid());

  // The fusion operator has passed the verification.
  // The errors of internal non key operators (which will be ignored later)
  // do not affect the transformation of the whole model,
  // So clear the error information of non key operators
  // This function call affects the return value of prechecker::instance().Haserror()
  GE_RETURN_IF_ERROR(ClearFusionOpError(op_node_name_list));

  // Check the input validity of the node, the input attribute must have a corresponding node
  GE_RETURN_IF_ERROR(CheckGraphDefValid(graph_def));

  // Building input and input relationships for all OP nodes
  GE_RETURN_IF_ERROR(GetOpNodesContextFromGraph(graph_def));
  GELOGD("[TF ParseFromMemory] get op nodes context from graph success");

  // Infer input formats
  ge::GetParserContext().format = InferInputFormats();
  GELOGD("[TF ParseFromMemory] infer input formats success");

  // Building input-output relationship between fusionop and common op
  GE_RETURN_IF_ERROR(UpdateAllNodeOpContext(scope_graph, graph_def, op_node_name_list));

  ret = AddFusionNodeDef(scope_graph, op_node_name_list);
  if (ret != SUCCESS) {
    GELOGE(ret, "Add fusion NodeDef failed.");
    DeleteFuisonNodeDef();
    return ret;
  }
  GELOGI("TF op node size = %zu.", op_node_name_list.size());
  // Loop analysis of op_nodes and map them to nodes in graph
  for (size_t i = 0; i < op_node_name_list.size(); i++) {
    GELOGI("TF op node name = %s.", op_node_name_list[i].c_str());
    const string op_node_name = op_node_name_list[i];
    const domi::tensorflow::NodeDef *node_def = nodedef_map_[op_node_name_list[i]];
    if (node_def == nullptr) {
      REPORT_INNER_ERROR("E19999", "Node:%s can't find in nodedef_map_, check invalid", op_node_name.c_str());
      GELOGE(INTERNAL_ERROR, "Node def is nullptr, name:%s.", op_node_name.c_str());
      DeleteFuisonNodeDef();
      return INTERNAL_ERROR;
    }
    const string &node_op = node_def->op();
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((tensorflow_op_map.find(node_op) == tensorflow_op_map.end()), DeleteFuisonNodeDef();
                                   REPORT_INNER_ERROR("E19999", "Op type %s unsupport", node_op.c_str());
                                   return INTERNAL_ERROR, "Unsupport op type %s", node_op.c_str());

    ret = AddNode(node_def, graph, scope_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "Add node failed, name:%s.", op_node_name.c_str());
      DeleteFuisonNodeDef();
      return ret;
    }
  }

  DeleteFuisonNodeDef();

  GE_RETURN_IF_ERROR(AddEdges(graph));
  GE_RETURN_IF_ERROR(graph->TopologicalSorting());

  has_error = has_error || PreChecker::Instance().HasError();
  if (has_error) {
    GELOGE(PARAM_INVALID, "Precheck has errors.");
    return PARAM_INVALID;
  }
  GELOGI("[TF ParseFromMemory] Parse from memory success.");
  return SUCCESS;
}

Status TensorFlowModelParser::GetFunctionProto(const string &file,
                                               domi::tensorflow::GraphDefLibrary &graph_def_library) {
  int pos = file.rfind('/');
  string graph_def_path = (pos == -1) ? kFuncDefLibraryFilePath : file.substr(0, pos) + "/" + kFuncDefLibraryFilePath;
  GELOGI("Function def libraray path is %s.", graph_def_path.c_str());

  bool read = ge::parser::ReadProtoFromText(graph_def_path.c_str(), &graph_def_library);
  if (!read) {
    GELOGE(INTERNAL_ERROR,
           "Get subgraph library failed. "
           "The model contains function operators. "
           "Need to use the script func2graph.py in the atc package to save the subgraphs to graph_def_library.pbtxt");
    ErrorManager::GetInstance().ATCReportErrMessage("E12029");
    return FAILED;
  }

  GELOGI("Get subgraph library success.");
  return SUCCESS;
}

Status TensorFlowModelParser::Parse(const char *model_path, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(model_path);
  ge::ComputeGraphPtr root_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(root_graph);

  Status ret = Parse(model_path, root_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "Parser graph %s failed.", graph.GetName().c_str());
    return ret;
  }

  GELOGI("Parser graph %s success.", graph.GetName().c_str());
  return SUCCESS;
}

Status TensorFlowModelParser::Parse(const char *model_path, ge::ComputeGraphPtr &root_graph) {
  GE_CHECK_NOTNULL(model_path);
  GE_CHECK_NOTNULL(root_graph);

  GELOGI("Parse file %s", model_path);
  // Store objects parsed from pb files
  domi::tensorflow::GraphDef ori_def;
  bool read = ge::parser::ReadProtoFromBinaryFile(model_path, &ori_def);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(!read, return INTERNAL_ERROR, "read_proto_from_binary failed.");

  // Trim graph by user input and output.
  domi::tensorflow::GraphDef graph_def;
  if (ge::GetParserContext().input_dims.empty() && ge::GetParserContext().out_nodes_map.empty()) {
    graph_def = ori_def;
  } else {
    GELOGI("Before Trim, the Graph Node size is:%d", ori_def.node_size());
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(TrimGraph(ori_def, &graph_def), return INTERNAL_ERROR, "Trim Graph fail.");
    GELOGI("After Trim, The graph_def.node size is:%d", graph_def.node_size());
  }

  // Construct ParseArg for root graph.
  google::protobuf::Message *root_proto = &graph_def;
  std::deque<ParseArg> tasks;
  tasks.push_back({root_proto, "root", nullptr, "", root_graph});

  // Get sub graph from graph_def_library.pbtxt which prepared before and stored in model_path.
  std::map<std::string, domi::tensorflow::GraphDef> function_name_to_graphdef;

  // Parse all root graph and sub graph.
  while (!tasks.empty()) {
    auto arg = tasks.front();
    tasks.pop_front();

    if (arg.proto == nullptr) {
      if (function_name_to_graphdef.empty() && (ori_def.library().function_size() > 0)) {
        GELOGI("Graph has function size: %d ", ori_def.library().function_size());
        domi::tensorflow::GraphDefLibrary graph_def_library;
        GE_CHK_STATUS_RET(GetFunctionProto(model_path, graph_def_library));
        for (auto &ge_graph_def : graph_def_library.graph_def()) {
          function_name_to_graphdef[ge_graph_def.name()] = ge_graph_def.graph();
          GELOGD("Graph_def name: %s, node size: %d", ge_graph_def.name().c_str(), ge_graph_def.graph().node_size());
        }
      }

      auto iter = function_name_to_graphdef.find(arg.function_name);
      if (iter == function_name_to_graphdef.end()) {
        ErrorManager::GetInstance().ATCReportErrMessage("E12013", {"functionname"}, {arg.function_name});
        GELOGE(FAILED, "Failed to get subgraph by function name %s", arg.function_name.c_str());
        return FAILED;
      }
      arg.proto = &(iter->second);
    }

    GELOGI("Begin to parse graph %s", arg.function_name.c_str());
    auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    auto ret = model_parser->ParseAllGraph(arg.proto, arg.graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to parse graph %s, instance name %s", arg.function_name.c_str(),
             arg.graph->GetName().c_str());
      return ret;
    }

    ret = PostOpProcessForSubgraph(arg);
    if (ret != SUCCESS) {
      // the error log has been printed inner the function
      return ret;
    }

    ret = GenSubgraphParseTasks(arg.graph, tasks);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to gen tasks on graph:%s for next iteration", arg.graph->GetName().c_str());
      GELOGE(ret, "Failed to gen tasks on graph %s for next iteration", arg.graph->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::ParseAllGraph(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(proto);
  GE_CHECK_NOTNULL(graph);

  const domi::tensorflow::GraphDef *ori_graph = reinterpret_cast<const domi::tensorflow::GraphDef *>(proto);
  // Make a copy for operation without modifying the original graph def.
  domi::tensorflow::GraphDef graph_def = *ori_graph;

  GE_RETURN_WITH_LOG_IF_ERROR(ProtoTypePassManager::Instance().Run(&graph_def, domi::TENSORFLOW),
                              "Run ProtoType Pass Failed");

  shared_ptr<ge::ScopeGraph> scope_graph = nullptr;
  Status ret = ExcuteScopeFusionPasses(&graph_def, scope_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[TF Parse] scope fusion failed.");
    return ret;
  }
  GELOGD("[TF Parse] scope fusion success");

  GE_RETURN_IF_ERROR(OptimizeConstNodes4CustomOp(&graph_def));
  GELOGD("[TF Parse] optimize const nodes for custom op base success");

  // Add nodedef in the model to prechecker and check the general parameters
  // Prevent data residue in multiple calls
  PreChecker::Instance().Clear();
  for (int i = 0; i < graph_def.node_size(); i++) {
    const domi::tensorflow::NodeDef &node = graph_def.node(i);
    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().AddOp(&node, node.name(), node.op()),
                                "Add node_def to PreChecker failed, node name: %s.", node.name().c_str());

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(PreChecker::Instance().CheckName(&node) != SUCCESS, return FAILED,
                                   "Check op[%s] failed, name repeat in tensorflow pb file.", node.name().c_str());
    GE_CHK_BOOL_EXEC_NOLOG(
        node.op() == TENSORFLOWF_NODE_OP_IDENTITY,
        GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(PreChecker::Instance().CheckType(&node, true) != SUCCESS, return FAILED,
                                       "Check op[%s]'s optype failed, type is not supported.", node.name().c_str());)
  }

  bool has_error = false;
  // save node name
  vector<string> op_node_name_list;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const domi::tensorflow::NodeDef *node_def = graph_def.mutable_node(i);

    // If it is a fusion operator, put nodedef in the fusion_op_nodedef_map_
    if (MaybeFusionOp(scope_graph, node_def)) {
      GELOGI("Node: %s maybe a fusion op.", node_def->name().c_str());
    }

    // Do not exit immediately when there is an error, wait until all errors are collected before exiting
    GE_CHK_STATUS_EXEC(AddFmkNodeDefToMap(graph_def, node_def, op_node_name_list), has_error = true);
  }

  // Verify the validity of fusionop
  GE_RETURN_IF_ERROR(CheckFusionOpValid());

  // The fusion operator has passed the verification.
  // The errors of internal non key operators (which will be ignored later)
  // do not affect the transformation of the whole model,
  // So clear the error information of non key operators
  // This function call affects the return value of prechecker::instance().Haserror()
  GE_RETURN_IF_ERROR(ClearFusionOpError(op_node_name_list));

  // Check the input validity of the node, the input attribute must have a corresponding node
  GE_RETURN_IF_ERROR(CheckGraphDefValid(graph_def));
  GELOGD("[TF Parse] check graph success");

  // Building input and input relationships for all OP nodes
  GE_RETURN_IF_ERROR(GetOpNodesContextFromGraph(graph_def));
  GELOGD("[TF Parse] get op nodes context from graph success");

  // Infer input formats
  ge::GetParserContext().format = InferInputFormats();
  GELOGD("[TF Parse] infer input formats success");

  // Building input-output relationship between fusionop and common op
  GE_RETURN_IF_ERROR(UpdateAllNodeOpContext(scope_graph, graph_def, op_node_name_list));
  GELOGD("[TF Parse] update all node op context success");

  // set user-designate-inputs-order
  std::vector<std::string> user_inputs_order;
  for (auto &input : ge::GetParserContext().user_input_dims) {
    user_inputs_order.push_back(input.first);
  }
  graph->SetInputsOrder(user_inputs_order);

  ret = AddFusionNodeDef(scope_graph, op_node_name_list);
  if (ret != SUCCESS) {
    GELOGE(ret, "Add fusion NodeDef failed.");
    DeleteFuisonNodeDef();
    return ret;
  }
  GELOGI("TF op node size = %zu.", op_node_name_list.size());

  // Loop analysis of op_nodes and map them to nodes in graph
  for (size_t i = 0; i < op_node_name_list.size(); i++) {
    GELOGI("TF op node name = %s.", op_node_name_list[i].c_str());
    const string op_node_name = op_node_name_list[i];
    const domi::tensorflow::NodeDef *node_def = nodedef_map_[op_node_name_list[i]];
    if (node_def == nullptr) {
      REPORT_INNER_ERROR("E19999", "Node:%s can't find in nodedef_map_, check invalid", op_node_name.c_str());
      GELOGE(INTERNAL_ERROR, "Cannot find [%s] in nodedef map.", op_node_name_list[i].c_str());
      DeleteFuisonNodeDef();
      return INTERNAL_ERROR;
    }
    const string &node_op = node_def->op();

    if (tensorflow_op_map.find(node_op) == tensorflow_op_map.end()) {
      GELOGW("%s not found in tensorflow_op_map.", node_op.c_str());
    }
    Status ret = AddNode(node_def, graph, scope_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "Add op[%s] failed", node_def->name().c_str());
      DeleteFuisonNodeDef();
      return ret;
    }
  }

  GELOGD("[TF Parse] parse tf node to geop success");

  DeleteFuisonNodeDef();

  GE_RETURN_IF_ERROR(AddEdges(graph));
  Graph dest_graph = GraphUtils::CreateGraphFromComputeGraph(graph);
  GE_RETURN_IF_ERROR(ParserUtils::ExpandOneToManyGraph(dest_graph));
  GE_RETURN_IF_ERROR(RemoveIsolateNode(graph));
  GE_RETURN_IF_ERROR(CheckAndUpdateInputDesc(graph));
  GE_RETURN_IF_ERROR(graph->TopologicalSorting());

  if (has_error) {
    GELOGE(PARAM_INVALID, "Precheck has errors.");
    return PARAM_INVALID;
  }
  GELOGI("[TF Parser] Parse proto success.");
  return SUCCESS;
}

Status TensorFlowModelParser::CheckGraphDefValid(const domi::tensorflow::GraphDef &graph_def) {
  // Number of data nodes
  uint32_t data_node_count = 0;
  for (const domi::tensorflow::NodeDef &node_def : graph_def.node()) {
    // Check that all input is valid
    for (const string &node_name : node_def.input()) {
      string tmp_node_name;
      GE_RETURN_IF_ERROR(CheckInputNodeName(node_name, &tmp_node_name, nullptr, nullptr));

      if (nodedef_map_.find(tmp_node_name) == nodedef_map_.end()) {
        ErrorManager::GetInstance().ATCReportErrMessage("E12009", {"opname", "inputopname"},
                                                        {node_def.name(), node_name});
        GELOGE(INTERNAL_ERROR, "Op[%s]'s input op[%s] is not exist in the graph_def.", node_def.name().c_str(),
               node_name.c_str());
        return INTERNAL_ERROR;
      }
    }

    if (node_def.op() == TENSORFLOWF_NODE_OP_PLACEHOLDER || node_def.op() == ge::parser::ARG) {
      data_node_count++;
    }
  }
  if (data_node_count == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E12010");
    GELOGE(INTERNAL_ERROR, "Model has no Placeholder node.");
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::GetOpNodesContextFromGraph(const domi::tensorflow::GraphDef &graph_def) {
  // Build the input relationship first
  for (auto &iter : op_node_context_map_) {
    map<string, std::vector<std::pair<int32_t, int32_t>>> input_map;
    const string &op_node_name = iter.first;
    GE_RETURN_IF_ERROR(GetOpNodeInputMap(op_node_name, input_map));

    OpNodeContext &op_node_context = iter.second;
    op_node_context.input_map = input_map;
  }

  // Then build the output relationship
  GE_RETURN_IF_ERROR(GetOpNodeOutputMap(graph_def));

  return SUCCESS;
}

// Get the input relation of opnode includeing input_op and input_const
Status TensorFlowModelParser::GetOpNodeInputMap(const string &op_node_name,
                                                map<string, std::vector<std::pair<int32_t, int32_t>>> &input_map) {
  // Get the current nodedef according to the node_name
  const domi::tensorflow::NodeDef *node_def = nodedef_map_[op_node_name];
  GE_CHECK_NOTNULL(node_def);
  int32_t input_index = 0;
  int32_t output_index = 0;
  for (const string &input_node_name : node_def->input()) {
    GELOGD("Get Op InputMap, node_name : %s, input node:%s", node_def->name().c_str(), input_node_name.c_str());
    string tmp_node_name;
    bool control = false;
    GE_RETURN_IF_ERROR(CheckInputNodeName(input_node_name, &tmp_node_name, &output_index, &control));
    input_map[tmp_node_name].push_back({output_index, control ? kControlSlot : input_index});
    SaveEdgesControlInfo(node_def->name(), control);
    input_index = control ? input_index : input_index + 1;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::GetOpNodeOutputMap(const domi::tensorflow::GraphDef &graph_def) {
  // Loop through all nodes in graphdef
  for (const domi::tensorflow::NodeDef &node_def : graph_def.node()) {
    auto currentIter = op_node_context_map_.find(node_def.name());
    if (currentIter != op_node_context_map_.end()) {
      OpNodeContext &op_node_context = currentIter->second;
      // Find all input nodes of the current node
      for (auto &inputIter : op_node_context.input_map) {
        auto iter = op_node_context_map_.find(inputIter.first);
        if (iter != op_node_context_map_.end()) {
          std::vector<std::pair<int32_t, int32_t>> inputpairs = inputIter.second;
          OpNodeContext &op_node_context1 = iter->second;
          op_node_context1.output_map[node_def.name()].assign(inputpairs.begin(), inputpairs.end());
        }
      }
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::GeStoi(const string &input_node_name, const string &index_str, int32_t *index) {
  try {
    int32_t tmp_index = static_cast<int32_t>(std::stoi(index_str.c_str(), nullptr, 10));
    *index = tmp_index;
  } catch (std::invalid_argument &) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"},
                                                    {"input_node_name(" + input_node_name + ")", index_str});
    GELOGE(INTERNAL_ERROR, "stl[stoi] input_node_name[%s] indexstr[%s] is invalid argument!", input_node_name.c_str(),
           index_str.c_str());
    return INTERNAL_ERROR;
  } catch (std::out_of_range &) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"},
                                                    {"input_node_name(" + input_node_name + ")", index_str});
    GELOGE(INTERNAL_ERROR, "stl[stoi] input_node_name[%s] indexstr[%s] is out of range!", input_node_name.c_str(),
           index_str.c_str());
    return INTERNAL_ERROR;
  } catch (...) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10015", {"parameter", "value"},
                                                    {"input_node_name(" + input_node_name + ")", index_str});
    GELOGE(INTERNAL_ERROR, "stl[stoi] input_node_name[%s] indexstr[%s] is bad argument!", input_node_name.c_str(),
           index_str.c_str());
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::CheckInputNodeName(const string &input_node_name, string *node_name, int32_t *index,
                                                 bool *control) {
  // Processing scene: input: "^fastrcnn_predictions/map/while/Identity""
  string tmp_input_node_name = input_node_name;
  if (tmp_input_node_name.find("^") == 0) {
    tmp_input_node_name = tmp_input_node_name.substr(1, tmp_input_node_name.length() - 1);
    if (control != nullptr) {
      *control = true;
    }
  } else {
    if (control != nullptr) {
      *control = false;
    }
  }

  int32_t tmp_index = 0;
  auto find = tmp_input_node_name.find(":");
  if (find == string::npos) {
    *node_name = tmp_input_node_name;

    if (index == nullptr) {
      return SUCCESS;
    }
    *index = tmp_index;

    return SUCCESS;
  }

  string indexstr = tmp_input_node_name.substr(find + 1, tmp_input_node_name.length() - find - 1);
  *node_name = tmp_input_node_name.substr(0, find);

  if (index == nullptr) {
    return SUCCESS;
  }

  if (GeStoi(input_node_name, indexstr, index) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::RunScopeFusionPass(const vector<string> &scope_passes_list,
                                                 ScopePassManager &pass_manager,
                                                 shared_ptr<ge::ScopeGraph> &scope_graph) {
  if (scope_passes_list.empty()) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(scope_graph);
  auto &impl = ge::ScopeFusionPassRegistry::GetInstance().impl_;
  if (impl == nullptr) {
    REPORT_INNER_ERROR("E19999", "ScopeFusionPassRegistry is not properly initialized.");
    GELOGE(ge::MEMALLOC_FAILED, "ScopeFusionPassRegistry is not properly initialized.");
    return ge::MEMALLOC_FAILED;
  }

  for (auto &pass_name : scope_passes_list) {
    auto pass = impl->CreateScopeFusionPass(pass_name);
    if (pass == nullptr) {
      REPORT_INNER_ERROR("E19999", "Scope fusion pass[%s] is not registered.", pass_name.c_str());
      GELOGE(INTERNAL_ERROR, "Scope fusion pass[%s] is not registered.", pass_name.c_str());
      return INTERNAL_ERROR;
    }
    Status ret = pass_manager.AddPass(pass);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add scope fusion pass[%s] failed.", pass_name.c_str());
      GELOGE(INTERNAL_ERROR, "Add scope fusion pass[%s] failed.", pass_name.c_str());
      return INTERNAL_ERROR;
    }
  }
  Status ret = pass_manager.Run(scope_graph);
  if (ret != SUCCESS && ret != domi::SCOPE_NOT_CHANGED) {
    GELOGE(FAILED, "Run scope fusion pass failed, ret:%u.", ret);
    return FAILED;
  }
  return SUCCESS;
}

bool TensorFlowModelParser::MaybeFusionOp(shared_ptr<ge::ScopeGraph> &scope_graph,
                                          const domi::tensorflow::NodeDef *node_def) {
  GE_CHECK_NOTNULL(scope_graph);
  GE_CHECK_NOTNULL(node_def);
  // If it is a fusion operator, put nodedef in the fusion_op_nodedef_map_
  ge::ScopeFusionOpInfo info;
  std::vector<ge::ScopeFusionOpInfo> info_list;
  auto &impl = scope_graph->impl_;
  if (TensorFlowFunsionOPUtil::MaybeFusionOp(node_def->name(), &info) ||
      impl->IsFusionOpChild(node_def->name(), info_list)) {
    GE_IF_BOOL_EXEC(
        info_list.size() > 0, for (size_t i = 0; i < info_list.size(); ++i) {
          fusion_op_type_map_[info_list[i].fusion_node_name].push_back(info_list[i].fusion_op_type);
          fusion_op_type_map_[info_list[i].fusion_node_name].push_back(info_list[i].description);
          fusion_op_nodedef_map_[info_list[i].fusion_node_name].push_back(node_def);
          if (info_list[i].fusion_op_type == "Dropout" &&
              (node_def->op() == "Add" || node_def->op() == "RandomUniform")) {
            fusion_op_nodedef_map_[info_list[i].fusion_node_name].push_back(nodedef_map_[node_def->input(0)]);
          }
          if (info_list[i].fusion_op_type == "LayerNorm" && node_def->op() == "Mean") {
            fusion_op_nodedef_map_[info_list[i].fusion_node_name].push_back(nodedef_map_[node_def->input(1)]);
          }
          fusion_op_policy_[info_list[i].fusion_node_name] = info_list[i].scope_pass;
          fusion_op_children_[node_def->name()] = info_list[i];
        });
    GE_IF_BOOL_EXEC(info_list.size() == 0, fusion_op_type_map_[info.fusion_node_name].push_back(info.fusion_op_type);
                    fusion_op_type_map_[info.fusion_node_name].push_back(info.description);
                    fusion_op_nodedef_map_[info.fusion_node_name].push_back(node_def);
                    fusion_op_policy_[info.fusion_node_name] = info.scope_pass;
                    fusion_op_children_[node_def->name()] = info);

    return true;
  }

  return false;
}

bool TensorFlowModelParser::IsFusionOpChild(const string &node_name, ge::ScopeFusionOpInfo *info) {
  GE_CHK_BOOL_EXEC(info != nullptr,
                   REPORT_CALL_ERROR("E19999", "Param info is nullptr, check invalid");
                   return false, "fusion info is null.");
  // 1.View in full match fusion strategy first
  // 2.View in scope fusion policy then
  auto iter = fusion_op_children_.find(node_name);
  if (iter != fusion_op_children_.end()) {
    info->node_name = fusion_op_children_[node_name].node_name;
    info->fusion_node_name = fusion_op_children_[node_name].fusion_node_name;
    info->fusion_op_type = fusion_op_children_[node_name].fusion_op_type;
    info->description = fusion_op_children_[node_name].description;
    info->scope_pass = fusion_op_children_[node_name].scope_pass;

    return true;
  }

  return false;
}

bool TensorFlowModelParser::FusionOpChildIgnore(shared_ptr<ge::ScopeGraph> &scope_graph,
                                                const ge::ScopeFusionOpInfo &info) {
  GE_CHECK_NOTNULL(scope_graph);
  bool ignore = false;
  if (info.scope_pass) {
    // Scope fusion strategy
    auto &impl = scope_graph->impl_;
    ignore = impl->FusionOpChildIgnore(info);
  } else {
    // Full match fusion strategy
    ignore = TensorFlowFunsionOPUtil::FusionOpChildIgnore(info);
  }
  return ignore;
}

bool TensorFlowModelParser::IsFusionOp(shared_ptr<ge::ScopeGraph> &scope_graph,
                                       const domi::tensorflow::NodeDef *node_def) {
  // The caller guarantees that the pointer is not null
  auto &impl = scope_graph->impl_;
  if (TensorFlowFunsionOPUtil::IsFusionOp(node_def) || impl->IsFusionOp(node_def)) {
    return true;
  }

  return false;
}
Status TensorFlowModelParser::GetInPutIndex(shared_ptr<ge::ScopeGraph> &scope_graph, const ge::ScopeFusionOpInfo &info,
                                            const int32_t old_index, int32_t &new_index) {
  GE_CHECK_NOTNULL(scope_graph);
  Status ret;
  if (info.scope_pass) {
    auto &impl = scope_graph->impl_;
    ret = impl->GetInputOrOutputIndex(info, old_index, true, new_index);
  } else {
    ret = TensorFlowFunsionOPUtil::GetInPutIndex(info, old_index, new_index);
  }

  return ret;
}
Status TensorFlowModelParser::GetOutPutIndex(shared_ptr<ge::ScopeGraph> &scope_graph, const ge::ScopeFusionOpInfo &info,
                                             const int32_t old_index, int32_t &new_index) {
  GE_CHECK_NOTNULL(scope_graph);
  Status ret;
  if (info.scope_pass) {
    auto &impl = scope_graph->impl_;
    ret = impl->GetInputOrOutputIndex(info, old_index, false, new_index);
  } else {
    ret = TensorFlowFunsionOPUtil::GetOutPutIndex(info, old_index, new_index);
  }

  return ret;
}

Status TensorFlowModelParser::CheckFusionOpValid() {
  for (auto &iter : fusion_op_nodedef_map_) {
    const string fusion_node_name = iter.first;
    vector<const NodeDef *> nodedef_list = iter.second;
    vector<string> funsion_op_info = fusion_op_type_map_[fusion_node_name];
    // vecotr index 0 is fusion_op_type
    const string funsion_op_type = funsion_op_info[0];
    if (!fusion_op_policy_[fusion_node_name]) {
      // Check the validity of the fusion_op_nodedef_map children operator
      GE_RETURN_IF_ERROR(
          TensorFlowFunsionOPUtil::CheckFusionOpChildren(fusion_node_name, nodedef_list, funsion_op_type));

      // Because there are many scenes in tensorflow graph,
      // in order to avoid the problem of omission, the error is returned directly.
      // In the future, functions like rollback can be implemented according to the definition of fusion operator
    }
  }
  return SUCCESS;
}

bool TensorFlowModelParser::ConstOpNeedUpdate(const string &op_name) {
  if (nodedef_map_[op_name]->op() != TENSORFLOWF_NODE_OP_CONST) {
    // Normal op need to update
    return true;
  } else {
    auto iter = op_node_context_map_.find(op_name);
    if (iter != op_node_context_map_.end()) {
      ge::ScopeFusionOpInfo info;
      auto outmap = iter->second.output_map;
      for (auto &out_node : outmap) {
        // if the const op output connected to are all fusion ops and the cosnt op is not in the update vector
        if (!IsFusionOpChild(out_node.first, &info)) {
          return true;
        }
      }
      if (std::find(const_op_update_vec.begin(), const_op_update_vec.end(), op_name) == const_op_update_vec.end()) {
        return false;
      }
    }
    return true;
  }
}

Status TensorFlowModelParser::UpdateAllNodeOpContext(shared_ptr<ge::ScopeGraph> &scope_graph,
                                                     const domi::tensorflow::GraphDef &graph_def,
                                                     vector<string> &op_node_name_list) {
  GE_CHECK_NOTNULL(scope_graph);
  vector<string> tmp_op_node_name_list;
  map<string, OpNodeContext> tmp_fusion_op_node_context_map;

  for (auto &op_node_name : op_node_name_list) {
    auto iter = op_node_context_map_.find(op_node_name);
    if (iter != op_node_context_map_.end()) {
      ge::ScopeFusionOpInfo info;
      if (IsFusionOpChild(op_node_name, &info) && nodedef_map_[op_node_name]->op() != TENSORFLOWF_NODE_OP_CONST) {
        // This node is a fusion operator
        auto fusion_iter = tmp_fusion_op_node_context_map.find(info.fusion_node_name);
        if (fusion_iter == tmp_fusion_op_node_context_map.end()) {
          OpNodeContext op_node_context;
          tmp_fusion_op_node_context_map[info.fusion_node_name] = op_node_context;
          tmp_op_node_name_list.push_back(info.fusion_node_name);
        }

        OpNodeContext &fusion_op_node_context = tmp_fusion_op_node_context_map[info.fusion_node_name];
        OpNodeContext &normal_op_node_context = op_node_context_map_[op_node_name];
        GE_RETURN_IF_ERROR(UpdateFusionOpContext(scope_graph, info, fusion_op_node_context, normal_op_node_context));

        // Delete fusion operator context
        op_node_context_map_.erase(iter);
      } else {
        // This node is a common operator
        OpNodeContext &normal_op_node_context = op_node_context_map_[op_node_name];
        GE_RETURN_IF_ERROR(UpdateNormalOpContext(scope_graph, op_node_name, normal_op_node_context));
        tmp_op_node_name_list.push_back(op_node_name);
      }
    }
  }

  // update op_node_name_list
  op_node_name_list.clear();
  op_node_name_list.assign(tmp_op_node_name_list.begin(), tmp_op_node_name_list.end());

  // update op_node_context_map_
  for (const auto &iter : tmp_fusion_op_node_context_map) {
    op_node_context_map_[iter.first] = iter.second;
  }
  // Normalized context
  GE_RETURN_IF_ERROR(NormalizeAllNodeOpContext());

  return SUCCESS;
}

Status TensorFlowModelParser::UpdateFusionOpContext(shared_ptr<ge::ScopeGraph> &scope_graph,
                                                    const ge::ScopeFusionOpInfo &info,
                                                    OpNodeContext &fusion_op_node_context,
                                                    OpNodeContext &normal_op_node_context) {
  GE_CHECK_NOTNULL(scope_graph);
  if (FusionOpChildIgnore(scope_graph, info)) {
    // The inner children operators of the fusion operator can be ignored directly
    // if they do not establish the edge relationship with other outer ordinary / fusion operators
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(UppdateInputMap(scope_graph, info, fusion_op_node_context, normal_op_node_context),
                    "UppdateInputMap ret fail");
  GE_CHK_STATUS_RET(UppdateOutputMap(scope_graph, info, fusion_op_node_context, normal_op_node_context),
                    "UppdateOutputMap ret fail");

  return SUCCESS;
}

Status TensorFlowModelParser::UppdateInputMap(shared_ptr<ge::ScopeGraph> &scope_graph,
                                              const ge::ScopeFusionOpInfo &info, OpNodeContext &fusion_op_node_context,
                                              OpNodeContext &normal_op_node_context) {
  GE_CHECK_NOTNULL(scope_graph);
  for (auto &iter : normal_op_node_context.input_map) {
    string input_node_name = iter.first;
    std::vector<std::pair<int32_t, int32_t>> &pairs = iter.second;
    ge::ScopeFusionOpInfo from_info;
    int32_t from_index = 0;
    int32_t to_index = 0;
    if (!ConstOpNeedUpdate(input_node_name)) {
      GELOGI("%s is const node connected to a fusion child, ignore", input_node_name.c_str());
      continue;
    }
    if (IsFusionOpChild(input_node_name, &from_info)) {
      if (info.fusion_node_name == from_info.fusion_node_name) {
        // Ignore two sub operators in the same fusion operator
        continue;
      }

      for (auto &pair : pairs) {
        GE_RETURN_WITH_LOG_IF_ERROR(GetOutPutIndex(scope_graph, from_info, pair.first, from_index),
                                    "GetOutPutIndex failed ,input_node_name %s.", input_node_name.c_str());
        GE_RETURN_WITH_LOG_IF_ERROR(GetInPutIndex(scope_graph, info, pair.second, to_index),
                                    "GetInPutIndex failed ,input_node_name %s.", input_node_name.c_str());
        fusion_op_node_context.input_map[from_info.fusion_node_name].push_back({from_index, to_index});
        UpdateEdgesControlInfo(info);
        GELOGD("[Update op context] update fusion input map for fusion input, %s:%d  TO  %s:%d",
               from_info.fusion_node_name.c_str(), from_index, info.fusion_node_name.c_str(), to_index);
      }
    } else {
      for (auto &pair : pairs) {
        from_index = pair.first;
        GE_RETURN_WITH_LOG_IF_ERROR(GetInPutIndex(scope_graph, info, pair.second, to_index),
                                    "GetInPutIndex input_node_name %s.", input_node_name.c_str());
        fusion_op_node_context.input_map[input_node_name].push_back({from_index, to_index});
        UpdateEdgesControlInfo(info);
        GELOGD("[Update op context] update fusion input map for normal input, %s:%d  TO  %s:%d",
               input_node_name.c_str(), from_index, info.fusion_node_name.c_str(), to_index);
      }
    }
  }
  return SUCCESS;
}
Status TensorFlowModelParser::UppdateOutputMap(shared_ptr<ge::ScopeGraph> &scope_graph,
                                               const ge::ScopeFusionOpInfo &info, OpNodeContext &fusion_op_node_context,
                                               OpNodeContext &normal_op_node_context) {
  GE_CHECK_NOTNULL(scope_graph);
  for (auto &iter : normal_op_node_context.output_map) {
    string output_node_name = iter.first;
    std::vector<std::pair<int32_t, int32_t>> &pairs = iter.second;
    ge::ScopeFusionOpInfo to_info;
    int32_t from_index = 0;
    int32_t to_index = 0;
    if (IsFusionOpChild(output_node_name, &to_info)) {
      if (info.fusion_node_name == to_info.fusion_node_name) {
        // Ignore two sub operators in the same fusion operator
        continue;
      }
      for (auto &pair : pairs) {
        GE_RETURN_WITH_LOG_IF_ERROR(GetOutPutIndex(scope_graph, info, pair.first, from_index),
                                    "fusion GetOutPutIndex failed ,output_node_name %s.", output_node_name.c_str());
        GE_RETURN_WITH_LOG_IF_ERROR(GetInPutIndex(scope_graph, to_info, pair.second, to_index),
                                    "fusion GetInPutIndex failed ,output_node_name %s.", output_node_name.c_str());
        fusion_op_node_context.output_map[to_info.fusion_node_name].push_back({from_index, to_index});
        GELOGD("[Update op context] update fusion output map for fusion output, %s:%d  TO  %s:%d",
               info.fusion_node_name.c_str(), from_index, to_info.fusion_node_name.c_str(), to_index);
      }
    } else {
      for (auto &pair : pairs) {
        to_index = pair.second;
        GE_RETURN_WITH_LOG_IF_ERROR(GetOutPutIndex(scope_graph, info, pair.first, from_index),
                                    "not fusion,GetOutPutIndex failed ,output_node_name %s.", output_node_name.c_str());
        fusion_op_node_context.output_map[output_node_name].push_back({from_index, to_index});
        GELOGD("[Update op context] update fusion output map for normal output, %s:%d  TO  %s:%d",
               info.fusion_node_name.c_str(), from_index, output_node_name.c_str(), to_index);
      }
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::EraseNormalOpOutputIfChild(shared_ptr<ge::ScopeGraph> &scope_graph,
                                                         const string &op_node_name,
                                                         OpNodeContext &normal_op_node_context) {
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> tmp_output_map;
  for (auto iter = normal_op_node_context.output_map.begin(); iter != normal_op_node_context.output_map.end();) {
    string output_node_name = iter->first;
    ge::ScopeFusionOpInfo to_info;
    int32_t from_index = 0;
    int32_t to_index = 0;

    if (IsFusionOpChild(output_node_name, &to_info) &&
        nodedef_map_[output_node_name]->op() != TENSORFLOWF_NODE_OP_CONST) {
      // Fuse operator, update index
      std::vector<std::pair<int32_t, int32_t>> &pairs = iter->second;
      for (auto &pair : pairs) {
        from_index = pair.first;
        GE_RETURN_WITH_LOG_IF_ERROR(GetInPutIndex(scope_graph, to_info, pair.second, to_index),
                                    "GetInPutIndex failed ,output_node_name %s.", output_node_name.c_str());
        tmp_output_map[to_info.fusion_node_name].push_back({from_index, to_index});
        GELOGD("[Update op context] update normal output map for fusion output, %s:%d  TO  %s:%d", op_node_name.c_str(),
               from_index, to_info.fusion_node_name.c_str(), to_index);
      }

      iter = normal_op_node_context.output_map.erase(iter);
    } else {
      iter++;
    }
  }

  for (auto &iter : tmp_output_map) {
    normal_op_node_context.output_map[iter.first] = iter.second;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::UpdateNormalOpContext(shared_ptr<ge::ScopeGraph> &scope_graph, const string &op_node_name,
                                                    OpNodeContext &normal_op_node_context) {
  GE_CHECK_NOTNULL(scope_graph);
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> tmp_input_map;

  for (auto iter = normal_op_node_context.input_map.begin(); iter != normal_op_node_context.input_map.end();) {
    string input_node_name = iter->first;
    ge::ScopeFusionOpInfo from_info;
    int32_t from_index = 0;
    int32_t to_index = 0;

    if (IsFusionOpChild(input_node_name, &from_info) &&
        nodedef_map_[input_node_name]->op() != TENSORFLOWF_NODE_OP_CONST) {
      // Fuse operator, update index
      std::vector<std::pair<int32_t, int32_t>> &pairs = iter->second;
      for (auto &pair : pairs) {
        to_index = pair.second;
        GE_RETURN_WITH_LOG_IF_ERROR(GetOutPutIndex(scope_graph, from_info, pair.first, from_index),
                                    "GetOutPutIndex failed ,input_node_name %s.", input_node_name.c_str());
        tmp_input_map[from_info.fusion_node_name].push_back({from_index, to_index});
        GELOGD("[Update op context] update normal input map for fusion input, %s:%d  TO  %s:%d",
               from_info.fusion_node_name.c_str(), from_index, op_node_name.c_str(), to_index);
      }

      iter = normal_op_node_context.input_map.erase(iter);
    } else {
      iter++;
    }
  }

  Status ret = EraseNormalOpOutputIfChild(scope_graph, op_node_name, normal_op_node_context);
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto &iter : tmp_input_map) {
    normal_op_node_context.input_map[iter.first] = iter.second;
  }

  return SUCCESS;
}

Status TensorFlowModelParser::NormalizeAllNodeOpContext() {
  for (auto iter = op_node_context_map_.begin(); iter != op_node_context_map_.end();) {
    OpNodeContext &context = iter->second;
    NormalizeInputOrOutputMap(context.input_map);
    NormalizeInputOrOutputMap(context.output_map);

    if ((context.input_map.size() == 0) && (context.output_map.size() == 0)) {
      GELOGD("[Update op context] node: %s will be removed at the back.", iter->first.c_str());
      iter = op_node_context_map_.erase(iter);
    } else {
      iter++;
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::NormalizeInputOrOutputMap(
    std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> &context_map) {
  if (context_map.size() == 0) {
    return SUCCESS;
  }

  for (auto iter = context_map.begin(); iter != context_map.end();) {
    std::vector<std::pair<int32_t, int32_t>> &pairs = iter->second;
    std::vector<std::pair<int32_t, int32_t>> temp_pairs;
    std::set<std::string> compare_set;

    for (auto &pair : pairs) {
      if ((pair.first == ge::kFusionDisableIndex) || (pair.second == ge::kFusionDisableIndex)) {
        // The edge will be cut off at the back, ignoring
        continue;
      }

      string name = to_string(pair.first) + ":" + to_string(pair.second);
      auto compare_iter = compare_set.find(name);
      if (compare_iter != compare_set.end()) {
        // pair<from,to> repeat, ignore
        continue;
      }

      temp_pairs.push_back(pair);
      compare_set.insert(name);
    }

    if (temp_pairs.size() == 0) {
      // If there is no pair, the context can be deleted
      iter = context_map.erase(iter);
      continue;
    } else {
      iter++;
    }

    pairs.clear();
    pairs.assign(temp_pairs.begin(), temp_pairs.end());
  }

  return SUCCESS;
}

void TensorFlowModelParser::DeleteFuisonNodeDef() {
  for (auto &fusion_nodedef : fusion_nodedef_list) {
    GE_DELETE_NEW_SINGLE(fusion_nodedef);
  }
}

void TensorFlowModelParser::SaveEdgesControlInfo(const string &node_name, const bool control) {
  if (control) {
    // If the control attribute is true, save the control attribute to edges_control_map
    edges_control_map[node_name].push_back(kControlSlot);
  }
}

void TensorFlowModelParser::UpdateEdgesControlInfo(const ge::ScopeFusionOpInfo &info) {
  auto iter = edges_control_map.find(info.node_name);
  if (iter != edges_control_map.end()) {
    // Delete the original fusion operator node information and add the fusion operator control edge information
    edges_control_map.erase(iter);
    edges_control_map[info.fusion_node_name].push_back(kControlSlot);
  }
}

bool TensorFlowModelParser::GetEdgesControlInfo(const string &node_name, const int32_t index) {
  // If the node name is included, then confirm whether the index is the same
  auto iter = edges_control_map.find(node_name);
  if (iter != edges_control_map.end()) {
    for (auto &i : iter->second) {
      if (i == index) {
        return true;
      }
    }
  }

  return false;
}

Status TensorFlowModelParser::ClearFusionOpError(const vector<string> &op_node_name_list) {
  for (const auto &name : op_node_name_list) {
    ge::ScopeFusionOpInfo info;
    if (IsFusionOpChild(name, &info)) {
      const NodeDef *node = nodedef_map_[name];
      GE_CHECK_NOTNULL(node);
      GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().Clear(node, "fused and removed."),
                                  "Clear pre-checking for node %s failed.", node->name().c_str());
    }
  }

  return SUCCESS;
}

Status TensorFlowModelParser::ToJson(const char *model_file, const char *json_file) {
  GE_CHK_BOOL_RET_STATUS(model_file != nullptr, FAILED, "model_file is nullptr.");
  GE_CHK_BOOL_RET_STATUS(json_file != nullptr, FAILED, "json_file is nullptr.");
  domi::tensorflow::GraphDef graph_def;
  nlohmann::json j;

  GE_RETURN_WITH_LOG_IF_FALSE(ge::parser::ReadProtoFromBinaryFile(model_file, &graph_def),
                              "ReadProtoFromBinaryFile failed, file:%s.", model_file);

  Pb2Json::Message2Json(graph_def, kTfBlackFields, j, true);
  return ModelSaver::SaveJsonToFile(json_file, j);
}

Status TensorFlowWeightsParser::ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) {
  return SUCCESS;
}

Status TensorFlowWeightsParser::Parse(const char *file, ge::Graph &graph) { return SUCCESS; }

Status TensorFlowModelParser::ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  ErrorManager::GetInstance().GenWorkStreamIdDefault();
  PARSER_TIMESTAMP_START(ParseProto);
  GE_CHECK_NOTNULL(proto);
  GE_CHECK_NOTNULL(graph);
  ge::GetParserContext().train_flag = true;

  const domi::tensorflow::GraphDef *graph_def_in = reinterpret_cast<const domi::tensorflow::GraphDef *>(proto);
  // Make a copy for operation without modifying the original graph def.
  domi::tensorflow::GraphDef graph_def_operation = *graph_def_in;
  domi::tensorflow::GraphDef *graph_def = &graph_def_operation;
  GELOGI("[TF Parser] graph def version:%d", graph_def->version());

  GE_RETURN_WITH_LOG_IF_ERROR(ProtoTypePassManager::Instance().Run(graph_def, domi::TENSORFLOW),
                              "Run ProtoType Pass Failed");

  shared_ptr<ge::ScopeGraph> scope_graph = nullptr;
  Status ret = ExcuteScopeFusionPasses(graph_def, scope_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[TF Parser] scope fusion failed.");
    return ret;
  }
  GELOGD("[TF Parser] scope fusion success");

  bool has_error = false;

  // Graphdef optimizes identity
  PARSER_TIMESTAMP_START(GraphDefOptimize);
  GE_RETURN_IF_ERROR(GraphDefOptimize(graph_def));
  PARSER_TIMESTAMP_END(GraphDefOptimize, "TensorFlowModelParser::GraphDefOptimize");
  GELOGD("[TF Parser] graph def optimize success");

  // Optimization for TVM operator
  PARSER_TIMESTAMP_START(OptimizeConstNodes4CustomOp);
  GE_RETURN_IF_ERROR(OptimizeConstNodes4CustomOp(graph_def));
  PARSER_TIMESTAMP_END(OptimizeConstNodes4CustomOp, "TensorFlowModelParser::OptimizeConstNodes4CustomOp");
  GELOGD("[TF Parser] optimize const nodes for custom op success");

  GE_RETURN_IF_ERROR(GetTensorflowGraphInOutMap(graph_def));
  GE_RETURN_IF_ERROR(RemoveIsolateNode(graph_def));

  vector<string> op_node_name_list;
  bool isDatasetInit = false;
  PARSER_TIMESTAMP_START(AddFmkNodeDefToMap);
  for (int i = 0; i < graph_def->node_size(); i++) {
    const domi::tensorflow::NodeDef *node_def = graph_def->mutable_node(i);
    if (node_def->op() == ge::parser::IDENTITY && node_def->input_size() == 0) {
      continue;
    }
    if (node_def->op() == ge::parser::SNAPSHOT && node_def->input_size() == 0) {
      continue;
    }
    GE_IF_BOOL_EXEC(node_def->op() == "MakeIterator", isDatasetInit = true);

    // If it is a fusion operator, put nodedef in the fusion_op_nodedef_map_
    if (MaybeFusionOp(scope_graph, node_def)) {
      GELOGI("Node: %s maybe a fusion op.", node_def->name().c_str());
    }

    // Do not exit immediately when there is an error, wait until all errors are collected before exiting
    Status ret = AddFmkNodeDefToMap(*graph_def, node_def, op_node_name_list);
    GE_CHK_STATUS_EXEC(ret, return PARAM_INVALID, "add node_def to map failed");
  }
  PARSER_TIMESTAMP_END(AddFmkNodeDefToMap, "TensorFlowModelParser::AddFmkNodeDefToMap");
  GELOGI("[TF Parser] TF subgraph isDatasetInit: %d.", isDatasetInit);

  // Verify the validity of fusionop
  GE_RETURN_IF_ERROR(CheckFusionOpValid());

  // Build input and output relationships for all OP nodes
  PARSER_TIMESTAMP_START(GetOpNodesContextFromGraph);
  GE_RETURN_IF_ERROR(GetOpNodesContextFromGraph(*graph_def));
  PARSER_TIMESTAMP_END(GetOpNodesContextFromGraph, "TensorFlowModelParser::GetOpNodesContextFromGraph");
  GELOGD("[TF Parser] Get op nodes context from graph success");

  // Building input-output relationship between fusionop and common op
  GE_RETURN_IF_ERROR(UpdateAllNodeOpContext(scope_graph, *graph_def, op_node_name_list));

  GELOGI("[TF Parser] TF op node size = %zu.", op_node_name_list.size());
  PARSER_TIMESTAMP_START(AddFmkNode);
  // Loop analysis of op_nodes and map them to nodes in graph
  ret = AddFmkNode(graph, scope_graph, op_node_name_list, isDatasetInit);
  PARSER_TIMESTAMP_END(AddFmkNode, "TensorFlowModelParser::AddFmkNode");
  GE_CHK_STATUS_EXEC(ret, DeleteFuisonNodeDef(); return ret, "AddFmkNode failed");
  GELOGD("[TF Parser] Add framework node success");

  ret = AddEdges(graph);

  Graph dest_graph = GraphUtils::CreateGraphFromComputeGraph(graph);
  GE_RETURN_IF_ERROR(ParserUtils::ExpandOneToManyGraph(dest_graph));

  DeleteFuisonNodeDef();
  GE_CHK_STATUS_EXEC(ret, return ret, "AddEdges failed");
  GELOGD("[TF Parser] Add edges success");

  PARSER_TIMESTAMP_START(RemoveIsolateNode);
  // Delete isolated nodes
  GE_RETURN_IF_ERROR(RemoveIsolateNode(graph));
  GE_RETURN_IF_ERROR(CheckAndUpdateInputDesc(graph));

  PARSER_TIMESTAMP_END(RemoveIsolateNode, "TensorFlowModelParser::RemoveIsolateNode");
  PARSER_TIMESTAMP_START(TopologicalSorting);
  GE_RETURN_IF_ERROR(graph->TopologicalSorting());
  PARSER_TIMESTAMP_END(TopologicalSorting, "TensorFlowModelParser::TopologicalSorting");

  ge::parser::PassManager iterator_fusion_pass;
  try {
    (void)iterator_fusion_pass.AddPass("ParseProto::IteratorFusionPass",
                                       new ge::IteratorFusionPass(domi::TENSORFLOW, false));
  } catch (std::bad_alloc &e) {
    GELOGE(INTERNAL_ERROR, "Add pass failed, bad memory allocation occurs.");
    return INTERNAL_ERROR;
  }
  ret = iterator_fusion_pass.Run(graph);
  if (ret != SUCCESS && ret != ge::NOT_CHANGED) {
    GELOGE(ret, "Run graph passes optimize for preprocess failed, ret:%u.", ret);
    return ret;
  }

  has_error = has_error || PreChecker::Instance().HasError();
  if (has_error) {
    GELOGE(PARAM_INVALID, "Precheck has errors.");
    return PARAM_INVALID;
  }
  GELOGI("[TF Parser] Parse proto success.");
  PARSER_TIMESTAMP_END(ParseProto, "TensorFlowModelParser::ParseProto");
  return SUCCESS;
}

Status TensorFlowModelParser::ParseProtoWithSubgraph(const google::protobuf::Message *root_proto,
                                                     domi::GetGraphCallback callback, ge::ComputeGraphPtr &root_graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  ErrorManager::GetInstance().GenWorkStreamIdDefault();
  GE_CHECK_NOTNULL(root_proto);
  GE_CHECK_NOTNULL(callback);
  GE_CHECK_NOTNULL(root_graph);

  PARSER_TIMESTAMP_START(ParseProtoWithSubgraph);
  std::vector<std::unique_ptr<google::protobuf::Message>> proto_holder;
  std::deque<ParseArg> tasks;
  tasks.push_back({root_proto, "root", nullptr, "", root_graph});

  while (!tasks.empty()) {
    auto arg = tasks.front();
    tasks.pop_front();

    if (arg.proto == nullptr) {
      auto proto = callback(root_proto, arg.function_name);
      if (proto == nullptr) {
        REPORT_CALL_ERROR("E19999", "callback execute failed, func_name:%s", arg.function_name.c_str());
        GELOGE(FAILED, "Failed to get function by name %s", arg.function_name.c_str());
        return FAILED;
      }
      arg.proto = proto.get();
      proto_holder.emplace_back(std::move(proto));
    }

    GELOGI("Begin to parse graph %s", arg.function_name.c_str());
    auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    auto ret = model_parser->ParseProto(arg.proto, arg.graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to parse graph %s, instance name %s", arg.function_name.c_str(),
             arg.graph->GetName().c_str());
      return ret;
    }

    ret = PostOpProcessForSubgraph(arg);
    if (ret != SUCCESS) {
      // the error log has been printed inner the function
      return ret;
    }

    ret = GenSubgraphParseTasks(arg.graph, tasks);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to gen tasks on graph %s for next iteration", arg.graph->GetName().c_str());
      return ret;
    }
  }
  PARSER_TIMESTAMP_EVENT_END(ParseProtoWithSubgraph, "TensorFlowModelParser::ParseProtoWithSubgraph");
  return SUCCESS;
}

Status TensorFlowModelParser::ParseProto(const std::string &serialized_proto, ge::ComputeGraphPtr &graph) {
  if (serialized_proto.empty()) {
    GELOGE(FAILED, "Deserialize proto failed as serialized proto is empty");
    return FAILED;
  }
  domi::tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(serialized_proto)) {
    GELOGE(FAILED, "Proto object GraphDef parse serialized proto failed");
    return FAILED;
  }
  return ParseProto(reinterpret_cast<const google::protobuf::Message *>(&graph_def), graph);
}

Status TensorFlowModelParser::ParseProtoWithSubgraph(const std::string &root_proto,
                                                     domi::GetGraphCallbackV2 callback,
                                                     ge::ComputeGraphPtr &root_graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  ErrorManager::GetInstance().GenWorkStreamIdDefault();
  GE_CHECK_NOTNULL(callback);
  GE_CHECK_NOTNULL(root_graph);

  PARSER_TIMESTAMP_START(ParseProtoWithSubgraph);
  std::deque<ParseArg> tasks;
  tasks.push_back({nullptr, "root", nullptr, "", root_graph});
  bool root_parsed = false;

  while (!tasks.empty()) {
    auto arg = tasks.front();
    tasks.pop_front();

    auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);

    Status ret = SUCCESS;
    if (root_parsed) {
      GELOGI("Begin to parse serialized proto of sub graph %s", arg.function_name.c_str());
      ret = model_parser->ParseProto(callback(arg.function_name), arg.graph);
    } else {
      GELOGI("Begin to parse serialized proto of root graph");
      ret = model_parser->ParseProto(root_proto, arg.graph);
      root_parsed = true;
    }

    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to parse graph %s, instance name %s", arg.function_name.c_str(),
             arg.graph->GetName().c_str());
      return ret;
    }

    ret = PostOpProcessForSubgraph(arg);
    if (ret != SUCCESS) {
      return ret;  // the error log has been printed inner the function
    }

    ret = GenSubgraphParseTasks(arg.graph, tasks);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to gen tasks for sub graph of graph %s", arg.graph->GetName().c_str());
      return ret;
    }
  }
  PARSER_TIMESTAMP_EVENT_END(ParseProtoWithSubgraph, "TensorFlowModelParser::ParseProtoWithSubgraph");
  return SUCCESS;
}

// For the identity operator whose output is "_retval", optimize it.
Status TensorFlowModelParser::OptimizeIdentityByOutput(map<string, NodeDef *> &nodedef_map,
                                                       const string &curr_node_name, bool &clear_input_flag) {
  auto context_iter = op_node_context_map_.find(curr_node_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((context_iter == op_node_context_map_.end()),
                                 REPORT_INNER_ERROR("E19999",
                                                    "Node:%s can't find in op_node_context_map_, check invalid",
                                                    curr_node_name.c_str());
                                 return INTERNAL_ERROR,
                                 "Can't find op node context.");
  OpNodeContext op_node_context = context_iter->second;

  auto node_def_iter = nodedef_map.find(curr_node_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((node_def_iter == nodedef_map.end()),
                                 REPORT_INNER_ERROR("E19999",
                                                    "Node:%s can't find in nodedef_map, check invalid",
                                                    curr_node_name.c_str());
                                 return INTERNAL_ERROR, "Can't find nodedef");
  domi::tensorflow::NodeDef *curr_node_def = node_def_iter->second;
  GE_CHECK_NOTNULL(curr_node_def);
  bool has_out_retval = false;
  // For the identity operator whose output is "_retval", optimize it
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> output_map = op_node_context.output_map;
  for (auto output_iter = output_map.begin(); output_iter != output_map.end(); ++output_iter) {
    const string &output_node_name = output_iter->first;
    domi::tensorflow::NodeDef *output_node_def = nodedef_map[output_node_name];
    GE_CHECK_NOTNULL(output_node_def);
    if (output_node_def->op() == "_Retval") {
      GELOGD("_Retval Identity need optimize.");
      output_node_def->set_input(0, curr_node_def->input(0).c_str());
      has_out_retval = true;
      GELOGD("op %s set input(0):%s.", output_node_def->name().c_str(), curr_node_def->input(0).c_str());
    }
  }

  // Deal with non _Retval output operator of Identity.
  if (has_out_retval) {
    for (auto output_iter = output_map.begin(); output_iter != output_map.end(); ++output_iter) {
      const string &output_node_name = output_iter->first;
      domi::tensorflow::NodeDef *output_node_def = nodedef_map[output_node_name];
      GE_CHECK_NOTNULL(output_node_def);
      GE_IF_BOOL_EXEC(output_node_def->op() == "_Retval", continue);
      for (int k = 0; k < output_node_def->input_size(); ++k) {
        GE_IF_BOOL_EXEC(
            output_node_def->input(k) == curr_node_name, output_node_def->set_input(k, curr_node_def->input(0).c_str());
            GELOGD("%s op set input(%d):%s.", output_node_def->name().c_str(), k, curr_node_def->input(0).c_str());)
      }
    }
    clear_input_flag = true;
  }
  return SUCCESS;
}

Status TensorFlowModelParser::GraphDefOptimizeIdentity(domi::tensorflow::GraphDef *graph_def,
                                                       map<string, NodeDef *> &nodedef_map,
                                                       const vector<NodeDef *> &nodedef_to_optimize) {
  GE_CHECK_NOTNULL(graph_def);
  if (!nodedef_to_optimize.empty()) {
    // Building input and input relationships for all OP nodes
    GE_RETURN_IF_ERROR(GetOpNodesContextFromGraph(*graph_def));
  } else {
    return SUCCESS;
  }
  for (auto &curr_node_def : nodedef_to_optimize) {
    GE_CHECK_NOTNULL(curr_node_def);
    bool clear_input_flag = false;
    const string &curr_node_name = curr_node_def->name();
    GE_RETURN_IF_ERROR(OptimizeIdentityByOutput(nodedef_map, curr_node_name, clear_input_flag));
    if (clear_input_flag) {
      curr_node_def->clear_input();
    }
  }
  GELOGI("GraphDefOptimizeIdentity success.");
  return SUCCESS;
}

Status TensorFlowModelParser::OptimizeSnapShot(domi::tensorflow::NodeDef *curr_mode_def,
                                               map<string, NodeDef *> &nodedef_map,
                                               const std::pair<string, int> &input_data,
                                               const std::vector<string> &control_list) {
  GE_CHECK_NOTNULL(curr_mode_def);
  if (curr_mode_def == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param curr_mode_def is nullptr, check invalid");
    GELOGE(FAILED, "input param is nullptr.");
    return PARAM_INVALID;
  }
  string curr_node_name = curr_mode_def->name();
  auto context_iter = op_node_context_map_.find(curr_node_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((context_iter == op_node_context_map_.end()),
                                 REPORT_INNER_ERROR("E19999",
                                                    "Node:%s can't find in op_node_context_map_, check invalid",
                                                    curr_node_name.c_str());
                                 return INTERNAL_ERROR,
                                 "Can't find op node context.");
  OpNodeContext op_node_context = context_iter->second;

  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> output_map = op_node_context.output_map;
  for (auto &output_iter : output_map) {
    const string &output_node_name = output_iter.first;
    domi::tensorflow::NodeDef *output_node_def = nodedef_map[output_node_name];
    GE_CHECK_NOTNULL(output_node_def);
    auto inputs = output_node_def->mutable_input();
    for (auto &input : *inputs) {
      string node_name;
      bool is_control = false;
      if (CheckInputNodeName(input, &node_name, nullptr, &is_control) != SUCCESS) {
        GELOGE(FAILED, "parse node input info failed, node %s, input %s.", output_node_def->name().c_str(),
               input.c_str());
        return FAILED;
      }
      if (node_name == curr_node_name) {
        if (is_control) {
          input = "^" + input_data.first;
        } else if (input_data.second == 0) {
          input = input_data.first;
        } else {
          input = input_data.first + ":" + std::to_string(input_data.second);
        }
        GELOGD("Optimize Snapshot node, dest:%s, set input:%s.", output_node_name.c_str(), input.c_str());

        for (auto &item : control_list) {
          bool is_exist_input = false;
          for (auto &tmp_input : output_node_def->input()) {
            string tmp_node_name;
            if (CheckInputNodeName(tmp_input, &tmp_node_name, nullptr, nullptr) != SUCCESS) {
              GELOGE(INTERNAL_ERROR, "parse node input info failed, node %s, input %s.",
                     output_node_def->name().c_str(), tmp_input.c_str());
              return FAILED;
            }
            if (tmp_node_name == item) {
              is_exist_input = true;
              break;
            }
          }
          if (!is_exist_input) {
            output_node_def->add_input("^" + item);
            GELOGD("Optimize Snapshot node, dest:%s, set control input:%s.", output_node_name.c_str(), item.c_str());
          }
        }
      }
    }
  }
  // Clear the input of snapshot and become an isolated node
  curr_mode_def->clear_input();
  return SUCCESS;
}

Status TensorFlowModelParser::GraphDefOptimizeSnapShot(domi::tensorflow::GraphDef *graph_def,
                                                       map<string, NodeDef *> &nodedef_map,
                                                       const vector<NodeDef *> &nodedef_to_optimize) {
  GE_CHECK_NOTNULL(graph_def);
  if (!nodedef_to_optimize.empty()) {
    // Building input and input relationships for all OP nodes
    GE_RETURN_IF_ERROR(GetOpNodesContextFromGraph(*graph_def));
    GELOGD("Optimize snapshot num:%zu.", nodedef_to_optimize.size());
  } else {
    return SUCCESS;
  }

  for (auto &curr_node_def : nodedef_to_optimize) {
    GE_CHECK_NOTNULL(curr_node_def);
    std::pair<string, int> input_data;  // src node name, src index
    vector<string> control_list;
    uint32_t data_input_cnt = 0;
    for (auto &input : curr_node_def->input()) {
      string node_name;
      int input_index = 0;
      bool is_control = false;
      if (CheckInputNodeName(input, &node_name, &input_index, &is_control) != SUCCESS) {
        GELOGE(FAILED, "parse SnapShot input info failed, node %s, input %s.", curr_node_def->name().c_str(),
               input.c_str());
        return FAILED;
      }
      if (is_control) {
        control_list.push_back(node_name);
      } else {
        data_input_cnt++;
        input_data = std::make_pair(node_name, input_index);
      }
    }
    if (data_input_cnt != 1) {
      REPORT_INNER_ERROR("E19999", "Node:%s's input data size:%u not equal to 1, check invalid",
                         curr_node_def->name().c_str(), data_input_cnt);
      GELOGE(FAILED, "%s op data input size %u invalid", curr_node_def->name().c_str(), data_input_cnt);
      return FAILED;
    }
    // Optimize Snapshot Node
    GE_CHK_STATUS_RET(OptimizeSnapShot(curr_node_def, nodedef_map, input_data, control_list));
  }
  GELOGI("GraphDefOptimizeSnapShot success.");
  return SUCCESS;
}

void TensorFlowModelParser::OptimizeDestroyTemporaryVariable(domi::tensorflow::GraphDef *graph_def,
                                                             domi::tensorflow::NodeDef *nodeCurrent,
                                                             bool &clearInputFlag) {
  // Internal call to ensure that the parameter is not empty.
  GELOGI("DestroyTemporaryVariable optimizing.");
  for (int w = 0; w < graph_def->node_size(); w++) {
    domi::tensorflow::NodeDef *nodeDst = graph_def->mutable_node(w);
    GE_IF_BOOL_EXEC(nodeDst->name() == nodeCurrent->name(), continue);
    for (int k = 0; k < nodeDst->input_size(); k++) {
      string nodeDstInputName = nodeDst->input(k);
      string nodeDstInputNameTmp;
      bool isControl = false;
      if (CheckInputNodeName(nodeDstInputName, &nodeDstInputNameTmp, nullptr, &isControl) != SUCCESS) {
        GELOGE(FAILED, "CheckInputNodeName failed, node is: %s", nodeDstInputName.c_str());
        return;
      }
      if (nodeDstInputNameTmp == nodeCurrent->name()) {
        GELOGI("current node name is %s ", nodeCurrent->name().c_str());
        clearInputFlag = true;
        if (isControl) {
          string nodeCurrentName = nodeCurrent->input(0);
          string nodeCurrentNameTmp;
          if (CheckInputNodeName(nodeCurrentName, &nodeCurrentNameTmp, nullptr, nullptr) != SUCCESS) {
            GELOGE(FAILED, "CheckInputNodeName failed, node is: %s", nodeCurrentName.c_str());
            return;
          }
          nodeCurrentNameTmp = "^" + nodeCurrentNameTmp;
          GELOGI("set nodeCurrentNameTmp: %s", nodeCurrentNameTmp.c_str());
          nodeDst->set_input(k, nodeCurrentNameTmp);
        } else {
          nodeDst->set_input(k, nodeCurrent->input(0).c_str());
          GELOGD("%s op set input:%s.", nodeDst->name().c_str(), nodeCurrent->input(0).c_str());
        }
        // DestroyTemporaryVariable node have only one input and one output.
        // If the number of inputs is greater than 1, all subsequent inputs are
        // control edge inputs. Therefore, after deleting DestroyTemporaryVariable,
        // these control edge inputs can be directly connected to nodeDst.
        if (nodeCurrent->input_size() > 1) {
          for (int i = 1; i < nodeCurrent->input_size(); ++i) {
            nodeDst->add_input(nodeCurrent->input(i));
          }
        }
        GELOGI("Optimize DestroyTemporaryVariable successful.");
      }
    }
  }
}

Status TensorFlowModelParser::GraphDefOptimizeDestroyTemporaryVariable(domi::tensorflow::GraphDef *graph_def,
                                                                       domi::tensorflow::NodeDef *nodeCurrent) {
  if (graph_def == nullptr || nodeCurrent == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param graph_def or nodeCurrent is nullptr, check invalid");
    GELOGE(FAILED, "input param is nullptr.");
    return FAILED;
  }
  if (nodeCurrent->op() != ge::parser::DESTROYTEMPORARYVARIABLE) {
    return SUCCESS;
  }

  GELOGI("Optimize DestroyTemporaryVariable, node name is :%s.", nodeCurrent->name().c_str());
  bool clearInputFlag = false;

  google::protobuf::Map<std::string, domi::tensorflow::AttrValue> *attr_map_destroy = nodeCurrent->mutable_attr();
  domi::tensorflow::AttrValue var_name_attr_destroy = (*attr_map_destroy)[ge::VAR_ATTR_NAME];

  for (int j = 0; j < graph_def->node_size(); j++) {
    domi::tensorflow::NodeDef *nodeTmpVar = graph_def->mutable_node(j);
    GE_IF_BOOL_EXEC(nodeTmpVar->op() != ge::parser::TEMPORARYVARIABLE, continue);

    google::protobuf::Map<std::string, domi::tensorflow::AttrValue> *attr_map_tmp = nodeTmpVar->mutable_attr();
    domi::tensorflow::AttrValue var_name_attr_tmp = (*attr_map_tmp)[ge::VAR_ATTR_NAME];

    if (var_name_attr_destroy.s() != var_name_attr_tmp.s()) {
      continue;
    }

    // Optimize destroytemporaryvariable operator
    OptimizeDestroyTemporaryVariable(graph_def, nodeCurrent, clearInputFlag);

    if (clearInputFlag) {
      nodeCurrent->clear_input();  // Clear the destroytemporaryvariable input to become an isolated node
      break;
    }
  }
  if (!clearInputFlag) {
    REPORT_INNER_ERROR("E19999", "Optimize DestroyTemporaryVariable failed, node name is :%s.",
                       nodeCurrent->name().c_str());
    GELOGE(INTERNAL_ERROR, "Optimize DestroyTemporaryVariable failed, node name is :%s.", nodeCurrent->name().c_str());
    return FAILED;
  }

  return SUCCESS;
}

struct DelTransposeInfo {
  domi::tensorflow::NodeDef *node_def;     // transpose
  domi::tensorflow::NodeDef *nextNodeDef;  // transpose --> [next]
  int inputIdx;
};

Status GetTransposeInfo(GraphDef *graph_def, std::map<std::string, std::string> &softmaxInfo,
                        std::map<std::string, DelTransposeInfo> &transposeInfo) {
  GE_CHECK_NOTNULL(graph_def);
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node_def = graph_def->mutable_node(i);
    if (node_def->op() == ge::parser::TRANSPOSE) {
      DelTransposeInfo transpose;
      transpose.node_def = node_def;
      transposeInfo.insert(std::make_pair(node_def->name(), transpose));
    } else if (node_def->op() == ge::parser::SOFTMAX) {
      softmaxInfo.insert(std::make_pair(node_def->name(), node_def->input(0)));
      GELOGI("softmax name:%s, input name:%s", node_def->name().c_str(), node_def->input(0).c_str());
    }
  }

  for (auto &itTranspose : transposeInfo) {
    for (int j = 0; j < graph_def->node_size(); ++j) {
      auto nextNodeDef = graph_def->mutable_node(j);
      bool bFind = false;
      for (int k = 0; k < nextNodeDef->input_size(); ++k) {
        if (nextNodeDef->input(k) == itTranspose.first) {
          itTranspose.second.nextNodeDef = nextNodeDef;
          itTranspose.second.inputIdx = k;
          GELOGI("transpose info name:%s, next name:%s, idx:%d", itTranspose.second.node_def->name().c_str(),
                 nextNodeDef->name().c_str(), k);
          bFind = true;
          break;
        }
      }
      if (bFind) {
        break;
      }
    }
  }
  return SUCCESS;
}

Status EraseTransposeNode(std::map<std::string, std::string> &softmaxInfo,
                          std::map<std::string, DelTransposeInfo> &transposeInfo) {
  auto itTranspose = transposeInfo.begin();
  for (; itTranspose != transposeInfo.end();) {
    // transpose --> softmax
    bool bErase = true;
    if (softmaxInfo.find(itTranspose->second.node_def->input(0)) != softmaxInfo.end() ||
        softmaxInfo.find(itTranspose->second.nextNodeDef->name()) != softmaxInfo.end()) {
      bErase = false;
    }

    if (bErase) {
      GELOGI("erase node name:%s, input(0):%s", itTranspose->first.c_str(),
             itTranspose->second.node_def->input(0).c_str());
      itTranspose = transposeInfo.erase(itTranspose);
    } else {
      itTranspose++;
    }
  }

  if ((softmaxInfo.size() <= SIZE_MAX / kSoftmaxMultiple) &&
      (softmaxInfo.size() * kSoftmaxMultiple != transposeInfo.size())) {
    GELOGW("softmax size[%zu], transpose size[%zu]", softmaxInfo.size(), transposeInfo.size());
    return FAILED;
  }

  return SUCCESS;
}

void TensorFlowModelParser::OptimizeTranspose(std::map<std::string, DelTransposeInfo> &transposeInfo) {
  for (auto &it : transposeInfo) {
    auto transpose = it.second;
    transpose.nextNodeDef->set_input(transpose.inputIdx, transpose.node_def->input(kTransposeInputIdx));
    transpose.node_def->clear_input();
  }
}

void TensorFlowModelParser::SoftmaxAddAttr(GraphDef *graph_def) {
  // The caller guarantees that the pointer is not null
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node_def = graph_def->mutable_node(i);
    if (node_def->op() == ge::parser::SOFTMAX) {
      domi::tensorflow::AttrValue attr_value;
      attr_value.set_i(1);
      ge::TensorFlowUtil::AddNodeAttr("axis", attr_value, node_def);
      GELOGI("SoftmaxAddAttr, name: %s, input name:%s", node_def->name().c_str(), node_def->input(0).c_str());
    }
  }
}

Status TensorFlowModelParser::GraphDefOptimize(domi::tensorflow::GraphDef *graph_def) {
  GE_CHECK_NOTNULL(graph_def);
  map<string, NodeDef *> nodedef_map;
  vector<string> op_node_name_list;
  // Save Identity and ReadVariableOp
  vector<NodeDef *> identity_to_optimize;
  // Save Snapshot
  vector<NodeDef *> snapshot_to_optimize;

  for (int i = 0; i < graph_def->node_size(); i++) {
    // mutable_node return vale is not empty
    domi::tensorflow::NodeDef *node_def = graph_def->mutable_node(i);
    const string &node_name = node_def->name();
    Status ret = AddFmkNodeDefToMap(*graph_def, node_def, op_node_name_list);
    GE_CHK_STATUS_EXEC(ret, return PARAM_INVALID, "add node_def to map failed");
    if (node_def->op() == ge::parser::IDENTITY || node_def->op() == ge::parser::READVARIABLEOP) {
      identity_to_optimize.push_back(node_def);
    } else if (node_def->op() == ge::parser::SNAPSHOT) {
      snapshot_to_optimize.push_back(node_def);
    }
    nodedef_map[node_name] = node_def;
  }

  // Optimize for Identity/ReadVariableOp
  GE_RETURN_IF_ERROR(GraphDefOptimizeIdentity(graph_def, nodedef_map, identity_to_optimize));
  // Optimize for Snapshot
  GE_RETURN_IF_ERROR(GraphDefOptimizeSnapShot(graph_def, nodedef_map, snapshot_to_optimize));

  for (int i = 0; i < graph_def->node_size(); i++) {
    domi::tensorflow::NodeDef *nodeCurrent = graph_def->mutable_node(i);
    GE_CHK_STATUS_RET(GraphDefOptimizeDestroyTemporaryVariable(graph_def, nodeCurrent));
  }

  // These member variables will be rebuilt later and need to be cleared here.
  nodedef_map_.clear();
  op_node_context_map_.clear();
  return SUCCESS;
}

Status TensorFlowModelParser::RemoveIsolateNode(ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);

  auto nodes = graph->GetDirectNode();
  for (auto &n : nodes) {
    // get front 4 char
    if (n->GetName().substr(0, 4) == "dpop") {
      continue;
    }
    if ((n->GetType() == ge::parser::DATA) ||
        (ge::GetParserContext().out_nodes_map.find(n->GetName()) != ge::GetParserContext().out_nodes_map.end())) {
      GELOGI("Can not remove op [%s] because it is data or out node.", n->GetName().c_str());
      continue;
    }
    GE_IF_BOOL_EXEC((((n->GetInAllNodes().size() == 0) && (n->GetOutDataNodes().size() == 0)) ||
                     ((n->GetType() == ge::parser::CONSTANTOP || n->GetType() == ge::parser::CONSTANT) &&
                      (n->GetOutDataNodes().size() == 0))),
                    GE_CHK_STATUS_RET(ge::GraphUtils::IsolateNode(n, {}), "Isolate removed node: %s, type: %s failed",
                                      n->GetName().c_str(), n->GetType().c_str());
                    GE_CHK_STATUS_RET(ge::GraphUtils::RemoveNodeWithoutRelink(graph, n),
                                      "Remove node: %s, type: %s without relink failed", n->GetName().c_str(),
                                      n->GetType().c_str()););
  }
  return SUCCESS;
}

// The format specified by the command line argument is preferred,
// if not specified, use InferInputFormats to infer,
// and if the inference fails, the default NHWC format is used.
domiTensorFormat_t TensorFlowModelParser::InferInputFormats() {
  GE_IF_BOOL_EXEC(ge::GetParserContext().format != DOMI_TENSOR_RESERVED, return ge::GetParserContext().format);

  domiTensorFormat_t global_input_format = DOMI_TENSOR_RESERVED;
  set<const NodeDef *> visited_node;
  for (auto &node_item : nodedef_map_) {
    // Infer format for data node and save it to ge::GetParserContext().format.
    domiTensorFormat_t format = DOMI_TENSOR_RESERVED;
    const NodeDef *node = node_item.second;
    if (node == nullptr) {
      return format;
    }
    auto it = tensorflow_op_map.find(node->op());
    if (it != tensorflow_op_map.end() && it->second == ge::parser::DATA) {
      GE_IF_BOOL_EXEC(GetNodeFormat(node, NO_TRANSPOSE, format, visited_node) != SUCCESS,
                      GELOGW("Cannot infer input format, the NHWC format is used by default, and you can also "
                             "specify format by command line arguments.");
                      return domi::DOMI_TENSOR_NHWC);

      GE_IF_BOOL_EXEC(global_input_format == DOMI_TENSOR_RESERVED, global_input_format = format);

      GE_IF_BOOL_EXEC(
          format != DOMI_TENSOR_RESERVED && format != global_input_format,
          GELOGW("Multiple data ops with different formats are not supported, "
                 "the NHWC format is used by default, and you can also specify format by command line arguments.");
          return domi::DOMI_TENSOR_NHWC);
    }
  }

  return global_input_format == DOMI_TENSOR_RESERVED ? domi::DOMI_TENSOR_NHWC : global_input_format;
}

Status TensorFlowModelParser::GetNodeFormat(const NodeDef *node, TfTranspose pred_transpose, domiTensorFormat_t &format,
                                            set<const NodeDef *> &visited_node) {
  GE_CHECK_NOTNULL(node);
  // Avoid repeated visits.
  GE_IF_BOOL_EXEC(visited_node.find(node) != visited_node.end(), return SUCCESS);
  visited_node.emplace(node);

  GE_IF_BOOL_EXEC(node->op() == TENSORFLOWF_NODE_OP_SWITCH || node->op() == TENSORFLOWF_NODE_OP_MERGE, return SUCCESS);

  // If node has a data_format attribute, format is set according to data_format.
  domi::tensorflow::AttrValue attr;
  if (ge::TensorFlowUtil::FindAttrValue(node, TENSORFLOW_ATTR_DATA_FORMAT, attr) && node->op() != ge::parser::BIASADD) {
    GE_RETURN_IF_ERROR(ge::TensorFlowUtil::CheckAttrHasType(attr, TENSORFLOW_ATTR_TYPE_STRING));

    format = (attr.s() == TENSORFLOWF_TENSOR_NCHW) ? domi::DOMI_TENSOR_NCHW : domi::DOMI_TENSOR_NHWC;

    GE_IF_BOOL_EXEC(format == domi::DOMI_TENSOR_NCHW && pred_transpose == TO_NCHW, format = domi::DOMI_TENSOR_NHWC);
    GE_IF_BOOL_EXEC(format == domi::DOMI_TENSOR_NHWC && pred_transpose == TO_NHWC, format = domi::DOMI_TENSOR_NCHW);
    GE_IF_BOOL_EXEC((format == domi::DOMI_TENSOR_NCHW && pred_transpose == TO_NHWC) ||
                        (format == domi::DOMI_TENSOR_NHWC && pred_transpose == TO_NCHW),
                    GELOGI("Format conflicts with transpose.");
                    return FAILED);

    return SUCCESS;
  }

  TfTranspose transpose;
  GE_RETURN_IF_ERROR(GetFormatTranspose(node, transpose));
  GE_IF_BOOL_EXEC(pred_transpose == transpose && pred_transpose != NO_TRANSPOSE,
                  GELOGI("Multiple transpose conflicts.");
                  return FAILED);

  // If node does not have the data_format attribute, format is set according to the output node.
  string node_name = node->name();
  GE_IF_BOOL_EXEC(op_node_context_map_.find(node_name) == op_node_context_map_.end(),
                  GELOGI("node %s not found in op_node_context_map_", node_name.c_str());
                  return FAILED);

  domiTensorFormat_t inferred_format = DOMI_TENSOR_RESERVED;
  const OpNodeContext &node_ctx = op_node_context_map_.at(node_name);

  for (const auto &output_item : node_ctx.output_map) {
    auto node_iter = nodedef_map_.find(output_item.first);
    GE_IF_BOOL_EXEC(node_iter == nodedef_map_.end(),
                    GELOGI("node %s not found in nodedef_map_", output_item.first.c_str());
                    return FAILED);

    const NodeDef *output_node = node_iter->second;
    GE_CHECK_NOTNULL(output_node);
    domiTensorFormat_t output_format = DOMI_TENSOR_RESERVED;
    GE_RETURN_IF_ERROR(GetNodeFormat(output_node, transpose, output_format, visited_node));

    GE_IF_BOOL_EXEC(output_format != DOMI_TENSOR_RESERVED && inferred_format != DOMI_TENSOR_RESERVED &&
                        output_format != inferred_format,
                    GELOGI("Multiple output formats conflict.");
                    return FAILED);

    inferred_format = output_format;
  }

  format = inferred_format;

  return SUCCESS;
}

Status TensorFlowModelParser::GetFormatTranspose(const NodeDef *transpose_node, TfTranspose &transpose_direc) {
  GE_CHECK_NOTNULL(transpose_node);
  transpose_direc = NO_TRANSPOSE;

  GE_IF_BOOL_EXEC(transpose_node->op() != TENSORFLOWF_NODE_OP_TRANSPOSE, return SUCCESS);

  GE_IF_BOOL_EXEC(transpose_node->input_size() != kInputNumInt, GELOGI("Input size of transpose is not 2.");
                  return FAILED);

  string perm_node_name = transpose_node->input(1);
  auto it = nodedef_map_.find(perm_node_name);
  GE_IF_BOOL_EXEC(it == nodedef_map_.end(), GELOGI("Node %s not found in nodedef_map_.", perm_node_name.c_str());
                  return FAILED);

  const NodeDef *perm_node = it->second;
  GE_CHECK_NOTNULL(perm_node);
  domi::tensorflow::AttrValue attr_value;
  GE_IF_BOOL_EXEC(perm_node->op() != TENSORFLOWF_NODE_OP_CONST, GELOGI("Input node of transpose is not const.");
                  return FAILED);

  GE_IF_BOOL_EXEC(!ge::TensorFlowUtil::FindAttrValue(perm_node, TENSORFLOW_ATTR_DTYPE, attr_value), return FAILED);
  GE_IF_BOOL_EXEC(ge::TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TYPE) != SUCCESS,
                  return FAILED);
  domi::tensorflow::DataType type = attr_value.type();
  GE_IF_BOOL_EXEC(type != domi::tensorflow::DT_INT32 && type != domi::tensorflow::DT_INT64, return FAILED);

  GE_IF_BOOL_EXEC(!ge::TensorFlowUtil::FindAttrValue(perm_node, TENSORFLOW_ATTR_VALUE, attr_value), return FAILED);
  GE_IF_BOOL_EXEC(ge::TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TENSOR) != SUCCESS,
                  return FAILED);
  const TensorProto &tensor = attr_value.tensor();
  const TensorShapeProto &tensor_shape = tensor.tensor_shape();
  GE_IF_BOOL_EXEC(tensor_shape.dim_size() != 1 || tensor_shape.dim(0).size() != parser::DIM_DEFAULT_SIZE,
                  return SUCCESS);
  GE_IF_BOOL_EXEC(tensor.tensor_content().empty(), return SUCCESS);

  vector<int64_t> perm_value;

  GE_IF_BOOL_EXEC(
      type == domi::tensorflow::DT_INT32,
      const int32_t *data = reinterpret_cast<const int32_t *>(tensor.tensor_content().data());
      for (int i = 0; i < parser::DIM_DEFAULT_SIZE; i++) { perm_value.push_back(data[i]); });

  GE_IF_BOOL_EXEC(
      type == domi::tensorflow::DT_INT64,
      const int64_t *data = reinterpret_cast<const int64_t *>(tensor.tensor_content().data());
      for (int i = 0; i < parser::DIM_DEFAULT_SIZE; i++) { perm_value.push_back(data[i]); });

  // 0, 1, 2, 3 present dim num.
  vector<int64_t> perm_to_nchw = {0, 3, 1, 2};
  vector<int64_t> perm_to_nhwc = {0, 2, 3, 1};
  GE_IF_BOOL_EXEC(perm_value == perm_to_nchw, transpose_direc = TO_NCHW);
  GE_IF_BOOL_EXEC(perm_value == perm_to_nhwc, transpose_direc = TO_NHWC);

  return SUCCESS;
}

Status TensorFlowModelParser::TrimGraph(const domi::tensorflow::GraphDef &input_graph_def,
                                        domi::tensorflow::GraphDef *output_graph_def) {
  GE_CHECK_NOTNULL(output_graph_def);
  if (!ge::GetParserContext().input_dims.empty() && ge::GetParserContext().out_nodes_map.empty()) {
    return TrimGraphByInput(input_graph_def, output_graph_def);
  } else {
    return TrimGraphByOutput(input_graph_def, output_graph_def);
  }
}
Status TensorFlowModelParser::TrimGraphByInput(const domi::tensorflow::GraphDef &input_graph_def,
                                               domi::tensorflow::GraphDef *output_graph_def) {
  // The caller guarantees that the pointer is not null
  std::set<string> delete_nodes;
  std::set<string> input_nodes;
  for (auto &iter : ge::GetParserContext().input_dims) {
    input_nodes.insert(iter.first);
  }
  std::map<string, const NodeDef *> node_lookup;
  for (const NodeDef &node : input_graph_def.node()) {
    node_lookup[node.name()] = &node;
  }
  std::vector<string> current_inputs;
  for (auto &iter : ge::GetParserContext().input_dims) {
    current_inputs.push_back(iter.first);
  }
  while (!current_inputs.empty()) {
    std::set<string> next_inputs;
    for (const string &current_input : current_inputs) {
      delete_nodes.insert(current_input);
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(!node_lookup.count(current_input),
                                     ErrorManager::GetInstance().ATCReportErrMessage(
                                          "E10016", {"parameter", "opname"}, {"input_shape", current_input});
                                     return FAILED, "Input op[%s] not found in graph.", current_input.c_str());
      const NodeDef *current_node = node_lookup[current_input];
      GE_CHECK_NOTNULL(current_node);
      for (const string &input_name : current_node->input()) {
        string input_node_name = NodeNameFromInput(input_name);
        if (!delete_nodes.count(input_node_name)) {
          next_inputs.insert(input_node_name);
        }
      }
    }
    current_inputs = std::vector<string>(next_inputs.begin(), next_inputs.end());
  }
  domi::tensorflow::GraphDef filtered_graph_def;
  filtered_graph_def.mutable_node()->Clear();
  for (const NodeDef &node : input_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      *(filtered_graph_def.mutable_node()->Add()) = node;
    }
    if (!delete_nodes.count(node.name())) {
      *(filtered_graph_def.mutable_node()->Add()) = node;
    }
  }
  output_graph_def->Clear();
  for (const NodeDef &node : filtered_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      NodeDef placeholder_node;
      placeholder_node = node;
      placeholder_node.clear_input();
      GE_IF_BOOL_EXEC(node.op() != "Placeholder", placeholder_node.set_op("Placeholder"));
      domi::tensorflow::AttrValue attr_value;
      TensorShapeProto *data_shape = attr_value.mutable_shape();
      GE_CHECK_NOTNULL(data_shape);
      const ge::ParserContext &ctx = ge::GetParserContext();
      std::map<std::string, std::vector<int64_t>> input_dims = ctx.input_dims;
      std::vector<int64_t> designated_dims = input_dims.at(node.name());
      for (int32_t i = 0; i < (int32_t)designated_dims.size(); i++) {
        data_shape->add_dim()->set_size(designated_dims[i]);
      }
      google::protobuf::Map<std::string, domi::tensorflow::AttrValue> *attr = placeholder_node.mutable_attr();
      (*attr)[TENSORFLOW_ATTR_SHAPE] = attr_value;
      GE_CHECK_NOTNULL(output_graph_def->mutable_node());
      *(output_graph_def->mutable_node()->Add()) = placeholder_node;
    } else {
      GE_CHECK_NOTNULL(output_graph_def->mutable_node());
      *(output_graph_def->mutable_node()->Add()) = node;
    }
  }
  return SUCCESS;
}
Status TensorFlowModelParser::TrimGraphByOutput(const domi::tensorflow::GraphDef &input_graph_def,
                                                domi::tensorflow::GraphDef *output_graph_def) {
  // The caller guarantees that the pointer is not null
  std::set<string> required_nodes;
  std::set<string> input_nodes;
  for (auto &iter : ge::GetParserContext().input_dims) {
    required_nodes.insert(iter.first);
    input_nodes.insert(iter.first);
  }
  for (auto &iter : ge::GetParserContext().out_nodes_map) {
    required_nodes.insert(iter.first);
  }
  std::map<string, const NodeDef *> node_lookup;
  for (const NodeDef &node : input_graph_def.node()) {
    node_lookup[node.name()] = &node;
  }
  std::vector<string> current_inputs;
  for (auto &iter : ge::GetParserContext().out_nodes_map) {
    current_inputs.push_back(iter.first);
  }
  while (!current_inputs.empty()) {
    std::set<string> next_inputs;
    for (const string &current_input : current_inputs) {
      required_nodes.insert(current_input);
      GE_IF_BOOL_EXEC(input_nodes.count(current_input), continue);
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(!node_lookup.count(current_input),
                                     ErrorManager::GetInstance().ATCReportErrMessage(
                                          "E10016", {"parameter", "opname"}, {"out_nodes", current_input});
                                     return FAILED, "Input op[%s] not found in graph.", current_input.c_str());
      const NodeDef *current_node = node_lookup[current_input];
      GE_CHECK_NOTNULL(current_node);
      for (const string &input_name : current_node->input()) {
        string input_node_name = NodeNameFromInput(input_name);
        if (!required_nodes.count(input_node_name)) {
          next_inputs.insert(input_node_name);
        }
      }
    }
    current_inputs = std::vector<string>(next_inputs.begin(), next_inputs.end());
  }
  domi::tensorflow::GraphDef filtered_graph_def;
  filtered_graph_def.mutable_node()->Clear();
  for (const NodeDef &node : input_graph_def.node()) {
    if (required_nodes.count(node.name())) {
      *(filtered_graph_def.mutable_node()->Add()) = node;
    }
  }
  output_graph_def->Clear();
  for (const NodeDef &node : filtered_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      NodeDef placeholder_node;
      placeholder_node = node;
      placeholder_node.clear_input();
      GE_IF_BOOL_EXEC(node.op() != "Placeholder", placeholder_node.set_op("Placeholder"));
      domi::tensorflow::AttrValue attr_value;
      TensorShapeProto *data_shape = attr_value.mutable_shape();
      GE_CHECK_NOTNULL(data_shape);
      const ge::ParserContext &ctx = ge::GetParserContext();
      std::map<std::string, std::vector<int64_t>> input_dims = ctx.input_dims;
      std::vector<int64_t> designated_dims = input_dims.at(node.name());
      for (int32_t i = 0; i < (int32_t)designated_dims.size(); i++) {
        data_shape->add_dim()->set_size(designated_dims[i]);
      }
      google::protobuf::Map<std::string, domi::tensorflow::AttrValue> *attr = placeholder_node.mutable_attr();
      (*attr)[TENSORFLOW_ATTR_SHAPE] = attr_value;
      GE_CHECK_NOTNULL(output_graph_def->mutable_node());
      *(output_graph_def->mutable_node()->Add()) = placeholder_node;
    } else {
      GE_CHECK_NOTNULL(output_graph_def->mutable_node());
      *(output_graph_def->mutable_node()->Add()) = node;
    }
  }
  return SUCCESS;
}
string TensorFlowModelParser::NodeNameFromInput(const string &input_name) {
  string prefix;
  string node_name;
  string suffix;
  std::vector<string> input_parts = ge::StringUtils::Split(input_name, ':');
  suffix = (input_parts.size() < kInputNumUint) ? "" : (":" + input_parts[1]);
  string tmp_name = input_parts[0];
  GE_IF_BOOL_EXEC(input_parts[0].find("^") == 0, tmp_name = tmp_name.substr(1, tmp_name.length() - 1));
  node_name = tmp_name;
  return node_name;
}

Status TensorFlowModelParser::FusionNodeParseParams(shared_ptr<OpParser> &op_parser,
                                                    const domi::tensorflow::NodeDef *node_def, ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node_def);
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(op_parser);

  GELOGI("FusionNodeParseParams:node name:%s.", node_def->name().c_str());

  // The fusion operator deals with parseparams separately
  shared_ptr<TensorFlowFusionOpParser> tensorflow_fusion_op_parser =
      std::dynamic_pointer_cast<TensorFlowFusionOpParser>(op_parser);
  GE_IF_BOOL_EXEC(tensorflow_fusion_op_parser == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param op_parser is not TensorFlowFusionOpParser Type, check invalid");
                  GELOGE(FAILED, "node :%s can not get fusion parser, please check!", node_def->name().c_str());
                  return INTERNAL_ERROR);

  // Find all children of the fusion operator
  auto iter = fusion_op_nodedef_map_.find(node_def->name());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(iter == fusion_op_nodedef_map_.end(),
                                 REPORT_INNER_ERROR("E19999",
                                                    "Node:%s can't find in fusion_op_nodedef_map_, check invalid",
                                                    node_def->name().c_str());
                                 return INTERNAL_ERROR,
                                 "FusionOp node %s has no children node!", node_def->name().c_str());

  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATTR_NAME_FUSIONOP_ORIGINAL_TYPE, node_def->op());
  vector<const domi::tensorflow::NodeDef *> node_def_v = iter->second;
  domi::FusionParseParamByOpFunc parse_param_func =
      domi::OpRegistry::Instance()->GetFusionParseParamByOpFunc(node->GetType(), node_def->op());
  Status status = FAILED;
  if (parse_param_func == nullptr) {
    status = tensorflow_fusion_op_parser->ParseParams(node_def_v, node);
    GE_CHK_STATUS_EXEC(status, return status, "Parse Params for fusionop node %s failed", node_def->name().c_str());
  } else {
    vector<ge::Operator> op_src_vec;
    for (const auto &node_def_src : node_def_v) {
      ge::Operator op_src(node_def_src->name(), node_def_src->op());
      status = domi::AutoMappingFn(node_def_src, op_src);
      if (status != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Auto mapping node_def:%s(%s) to operator failed",
                          node_def_src->name().c_str(), node_def_src->op().c_str());
        GELOGE(status, "Node[%s] auto mapping failed", node_def_src->name().c_str());
        return status;
      }
      auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_src);
      GE_CHECK_NOTNULL(op_desc);
      for (int32_t i = 0; i < node_def_src->input_size(); i++) {
        ge::GeTensorDesc tensor_desc;
        tensor_desc.SetName(node_def_src->input(i));
        if (op_desc->AddInputDesc(tensor_desc) != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                            op_desc->GetName().c_str(), op_desc->GetType().c_str());
          GELOGE(FAILED, "Op [%s] type[%s] add input(%d) tensor failed.", op_desc->GetName().c_str(),
                 op_desc->GetType().c_str(), i);
          return FAILED;
        }
      }
      op_src_vec.push_back(op_src);
    }
    shared_ptr<TensorFlowFusionCustomParserAdapter> tf_custom_fusion_op_paser =
        std::dynamic_pointer_cast<TensorFlowFusionCustomParserAdapter>(tensorflow_fusion_op_parser);
    status = tf_custom_fusion_op_paser->ParseParams(op_src_vec, node);
    if (status != SUCCESS) {
      GELOGE(status, "Parse params for fusionop node %s failed", node_def->name().c_str());
      return status;
    }
  }

  return SUCCESS;
}

/**
 * @ingroup domi_omg
 * @brief Optimizing const nodes for custom operators
 * @param [in] graph_def graph object
 * @return true optimize successfully
 * @return false optimize failed
 *
 */
Status TensorFlowModelParser::OptimizeConstNodes4CustomOp(domi::tensorflow::GraphDef *graph_def) {
  GE_CHECK_NOTNULL(graph_def);
  // 1. find all the nodes in the graph and save them to all_nodedef_map
  map<string, NodeDef *> all_nodedef_map;
  int graph_node_size = graph_def->node_size();
  for (int i = 0; i != graph_node_size; ++i) {
    // mutable_node return vale is not empty
    domi::tensorflow::NodeDef *current_node = graph_def->mutable_node(i);
    string node_name = current_node->name();
    all_nodedef_map[node_name] = current_node;
  }
  GE_CHK_BOOL_EXEC_INFO(!all_nodedef_map.empty(), return SUCCESS, "all_nodedef_map is empty");

  // 2. move input to attr.
  for (auto &it_node_map : all_nodedef_map) {
    domi::tensorflow::NodeDef *current_node = it_node_map.second;
    GE_CHECK_NOTNULL(current_node);
    string current_op_name = current_node->op();

    // 2.1. check whether the current op is register for move to attr.
    const std::vector<domi::RemoveInputConfigure> &move_input_vec =
        domi::OpRegistry::Instance()->GetRemoveInputConfigure(current_op_name);
    GE_CHK_BOOL_EXEC_NOLOG(!move_input_vec.empty(), continue);
    GELOGD("Current op %s is registered for remove input.", current_op_name.c_str());

    // 2.2 check whether the current op is a TVM op.
    GE_CHK_BOOL_EXEC_INFO(
        domi::OpRegistry::Instance()->GetImplyTypeByOriOpType(current_op_name) == domi::ImplyType::TVM, continue,
        "op %s is not TVM op", current_op_name.c_str());
    GELOGD("handle tvm op %s", current_op_name.c_str());

    // 2.3 copy input to attr
    set<uint32_t> unused_inputs;
    for (const auto &it : move_input_vec) {
      uint32_t move_index;
      if (it.inputIdx >= 0) {
        move_index = it.inputIdx;
      } else {
        GE_IF_BOOL_EXEC(
            -it.inputIdx > current_node->input_size(),
            ErrorManager::GetInstance().ATCReportErrMessage(
                "E12004", {"opname", "inputIdx", "inputsize"},
                {current_op_name, std::to_string(-it.inputIdx), std::to_string(current_node->input_size())});
            GELOGE(INTERNAL_ERROR,
                   "Op[%s] register failed, inputIdx[-%d] should be greater than inputsize[%d] when inputIdx < 0.",
                   current_op_name.c_str(), it.inputIdx, current_node->input_size());
            return PARAM_INVALID);
        move_index = current_node->input_size() + it.inputIdx;
      }
      // For an isolated node in deep lab V3 networ.
      // solve the problem of protobuf index less current_size.
      GE_IF_BOOL_EXEC(current_node->input_size() == 0, GELOGI("Input size is 0, already optimized"); continue);

      if (it.moveType == domi::OMG_REMOVE_TYPE_WITH_COND) {
        domi::tensorflow::AttrValue attr_value;
        GE_IF_BOOL_EXEC(!(ge::TensorFlowUtil::FindAttrValue(current_node, it.attrName, attr_value)),
                        REPORT_INNER_ERROR("E19999", "Op:%s register AttrName[%s] has no value, check invalid",
                                           current_op_name.c_str(), it.attrName.c_str());
                        GELOGE(INTERNAL_ERROR, "AttrName[%s] has no value!", it.attrName.c_str());
                        return PARAM_INVALID);
        GE_IF_BOOL_EXEC(attr_value.b() == it.attrValue, unused_inputs.insert(move_index));
      } else if (it.moveType == domi::OMG_REMOVE_INPUT_WITH_ORIGINAL_TYPE && it.originalType == current_op_name) {
        GELOGD("Input %s:%d will be removed.", current_op_name.c_str(), move_index);
        unused_inputs.insert(move_index);
      } else if (it.moveType == domi::OMG_INPUT_REORDER) {
        auto inputs = current_node->input();
        if (static_cast<size_t>(inputs.size()) != it.input_order.size()) {
          REPORT_INNER_ERROR("E19999", "Input size of node:%s(%s) is mismatched, new order size:%zu, input size:%d",
                             current_node->name().c_str(), current_node->op().c_str(),
                             it.input_order.size(), inputs.size());
          GELOGE(INTERNAL_ERROR, "Size of input is mismatched, new order size is %zu, input size is %d.",
                 it.input_order.size(), inputs.size());
          return INTERNAL_ERROR;
        }
        for (size_t i = 0; i < it.input_order.size(); ++i) {
          int new_index = it.input_order[i];
          if (new_index < 0 || new_index >= inputs.size()) {
            REPORT_INNER_ERROR("E19999", "New order of %s has invalid index %d, out of range(0, %d)",
                               it_node_map.first.c_str(), new_index, inputs.size());
            GELOGE(INTERNAL_ERROR, "New order of %s has invalid index %d.", it_node_map.first.c_str(), new_index);
            return INTERNAL_ERROR;
          }
          current_node->set_input(i, inputs[new_index]);
        }
        GELOGI("The input sequence of the node has been rearranged, node name:%s.", it_node_map.first.c_str());
      }
    }

    // 2.4 remove the input const nodes
    Status ret = RemoveInputs(graph_def, current_node, unused_inputs, all_nodedef_map);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "remove input for op:%s failed", current_op_name.c_str());
      GELOGE(INTERNAL_ERROR, "Op[%s] remove input failed.", current_op_name.c_str());
      return ret;
    }
  }

  return SUCCESS;
}

Status TensorFlowModelParser::AddControlEdgeAfterRemoveInputs(domi::tensorflow::GraphDef *graph_def,
                                                              domi::tensorflow::NodeDef *node_def,
                                                              const map<string, NodeDef *> &all_node_map,
                                                              const vector<string> &removed_inputs_vec) {
  GE_CHECK_NOTNULL(graph_def);
  GE_CHECK_NOTNULL(node_def);
  for (const auto &remove_input : removed_inputs_vec) {
    string input_node_name = NodeNameFromInput(remove_input);
    auto it = all_node_map.find(input_node_name);
    if (it == all_node_map.end()) {
      REPORT_INNER_ERROR("E19999", "Node:%s can't find in all_node_map, check invalid", input_node_name.c_str());
      GELOGE(FAILED, "Can not find node name:%s in all node map.", input_node_name.c_str());
      return FAILED;
    }
    NodeDef *input_node_def = it->second;
    if (input_node_def->op() == parser::SWITCH || input_node_def->op() == parser::REFSWITCH) {
      NodeDef *identity_node_def = graph_def->add_node();
      GE_CHECK_NOTNULL(identity_node_def);
      input_node_name = input_node_name + "identity";
      identity_node_def->set_name(input_node_name);
      identity_node_def->set_op(parser::IDENTITY);
      identity_node_def->add_input(remove_input);
    }
    string control_input = "^" + input_node_name;
    node_def->add_input(control_input);
    GELOGD("Add control input:%s for node:%s", control_input.c_str(), node_def->name().c_str());
  }
  return SUCCESS;
}
/**
 * @ingroup domi_omg
 * @brief Delete input from nodedef
 * @param [in] node_def Nodedef object
 * @param [in] remove_index_set Index collection of input nodes to be deleted
 * @return true remove successfully
 * @return false remove failed
 *
 */
Status TensorFlowModelParser::RemoveInputs(domi::tensorflow::GraphDef *graph_def,
                                           domi::tensorflow::NodeDef *node_def,
                                           const set<uint32_t> &remove_index_set,
                                           const map<string, NodeDef *> &all_node_map) {
  GE_CHECK_NOTNULL(node_def);
  if (remove_index_set.empty()) {
    GELOGI("The size of remove_index_set is zero.");
    return SUCCESS;
  }

  map<string, vector<int>> remove_inputs_map;
  for (auto &it : remove_index_set) {
    const string &input_node_name = node_def->input(it);
    remove_inputs_map[input_node_name].emplace_back(it);
    GELOGD("Push input:%s, index:%d into remove map.", input_node_name.c_str(), it);
  }

  RemoveInputAttr(node_def, remove_inputs_map);

  int index = 0;
  vector<string> removed_inputs_vec;
  auto *inputs = node_def->mutable_input();
  for (auto input_it = inputs->begin(); input_it != inputs->end(); ++index) {
    // 1.decide whether to remove the input
    bool flag = false;
    for (auto &remove_input : remove_inputs_map) {
      string remove_input_name = remove_input.first;
      vector<int> remove_input_indexs = remove_input.second;
      if ((*input_it) == remove_input_name &&
          std::find(remove_input_indexs.begin(), remove_input_indexs.end(), index) != remove_input_indexs.end()) {
        GELOGD("Remove input:%s, index:%d", remove_input_name.c_str(), index);
        flag = true;
        removed_inputs_vec.emplace_back(remove_input_name);
        break;
      }
    }

    if (flag) {
      // 2 remove the input
      input_it = inputs->erase(input_it);
    } else {
      ++input_it;
    }
  }

  Status ret = AddControlEdgeAfterRemoveInputs(graph_def, node_def, all_node_map, removed_inputs_vec);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Add control edges for node:%s failed.", node_def->name().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void TensorFlowModelParser::RemoveInputAttr(domi::tensorflow::NodeDef *node_def,
                                            const map<string, vector<int>> &remove_inputs_map) {
  // The caller guarantees that the pointer is not null
  auto *inputs = node_def->mutable_input();
  google::protobuf::Map<std::string, domi::tensorflow::AttrValue> *attr_map = node_def->mutable_attr();
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::iterator it =
      attr_map->find(ge::ATTR_NAME_INPUT_TENSOR_DESC);
  if (it == attr_map->end()) {
    GELOGW("Failed to find input desc from tf node_def[%s]", node_def->name().c_str());
  } else {
    domi::tensorflow::AttrValue *input_attr_value = &(it->second);
    auto tmp_attr = input_attr_value->mutable_list()->mutable_func();
    auto attr_it = tmp_attr->begin();
    int index = 0;
    for (auto input_it = inputs->begin(); input_it != inputs->end(); ++input_it, ++index) {
      // 1.decide whether to remove the input
      bool flag = false;
      for (auto &remove_input : remove_inputs_map) {
        string remove_input_name = remove_input.first;
        vector<int> remove_input_indexs = remove_input.second;
        if ((*input_it) == remove_input_name &&
            std::find(remove_input_indexs.begin(), remove_input_indexs.end(), index) != remove_input_indexs.end()) {
          GELOGD("Remove input attr:%s, index:%d", remove_input_name.c_str(), index);
          flag = true;
          break;
        }
      }

      if (flag) {
        // 2.1 remove the input attr
        if (!tmp_attr->empty() && attr_it != tmp_attr->end()) {
          attr_it = tmp_attr->erase(attr_it);
        } else {
          ++attr_it;
        }
      } else {
        ++attr_it;
      }
    }
  }
}

Status TensorFlowModelParser::GetTensorflowGraphInOutMap(domi::tensorflow::GraphDef *graph_def) {
  GE_CHECK_NOTNULL(graph_def);
  for (int i = 0; i < graph_def->node_size(); i++) {
    domi::tensorflow::NodeDef *node = graph_def->mutable_node(i);
    const string &node_name = node->name();
    node_inputs_outputs_map_.emplace(node_name, std::pair<set<string>, set<string>>{});
    for (const auto &input : node->input()) {
      string input_node_name;
      GE_RETURN_IF_ERROR(CheckInputNodeName(input, &input_node_name, nullptr, nullptr));
      node_inputs_outputs_map_[node_name].first.insert(input_node_name);
      node_inputs_outputs_map_[input_node_name].second.insert(node_name);
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::RemoveIsolateNode(domi::tensorflow::GraphDef *graph_def) {
  GE_CHECK_NOTNULL(graph_def);
  set<string> node_to_delete;
  for (int i = 0; i < graph_def->node_size(); i++) {
    domi::tensorflow::NodeDef *node = graph_def->mutable_node(i);
    const string &node_name = node->name();
    if (node_inputs_outputs_map_.find(node_name) == node_inputs_outputs_map_.end()) {
      REPORT_INNER_ERROR("E19999", "Node:%s can't find in node_inputs_outputs_map_, check invalid", node_name.c_str());
      GELOGE(FAILED, "Can not find input output context, node:%s.", node_name.c_str());
      return FAILED;
    }
    if ((node_inputs_outputs_map_[node_name].first.empty() && node_inputs_outputs_map_[node_name].second.empty() &&
         node->op() != kDpop) ||
        (node->op() == ge::parser::CONSTANT && node_inputs_outputs_map_[node_name].second.empty())) {
      GELOGI("%s will inset to node_to_delete", node_name.c_str());
      node_to_delete.insert(node_name);
    }
  }

  // delete isolate nodes
  auto nodeList = graph_def->mutable_node();
  for (auto iter = nodeList->begin(); iter != nodeList->end();) {
    if (node_to_delete.count(iter->name()) != 0) {
      GELOGI("%s has zero input and output, will delete.", iter->name().c_str());
      iter = nodeList->erase(iter);
    } else {
      iter++;
    }
  }
  return SUCCESS;
}

Status TensorFlowModelParser::RecordFusionResult(std::shared_ptr<ge::ScopeGraph> &scope_graph,
                                                 const domi::tensorflow::NodeDef *node, ge::OpDescPtr &op_desc) {
  // The caller guarantees that the pointer is not null
  GELOGI("RecordFusionResult for %s start.", op_desc->GetName().c_str());
  auto &impl_scope_graph = scope_graph->impl_;
  ge::FusionScopesResult *fusion_result = impl_scope_graph->GetFusionScopesResults(node);
  if (fusion_result == nullptr) {
    GELOGW("fusion_result is not found.");
    return SUCCESS;
  }

  std::vector<std::string> original_names;
  auto nodes = fusion_result->Nodes();
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(original_names),
                 [](ge::OperatorPtr n) -> std::string { return n->GetName(); });

  GELOGI("Op %s original_names size = %zu.", op_desc->GetName().c_str(), original_names.size());
  bool ret = ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  if (!ret) {
    GELOGW("Set %s to %s fail.", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str(), op_desc->GetName().c_str());
  }
  auto outputs_desc = op_desc->GetAllOutputsDesc();
  auto &impl = fusion_result->impl_;
  for (auto &fusion_output : impl->GetOutputs()) {
    for (size_t i = 0; i < fusion_output.second.size(); ++i) {
      if (fusion_output.second[i] == ge::kFusionDisableIndex) {
        continue;
      }

      if (fusion_output.second[i] >= static_cast<int32_t>(op_desc->GetOutputsSize())) {
        REPORT_INNER_ERROR("E19999", "fusion output index:%d of node:%s(%s) must less than outputs desc size %zu.",
                           fusion_output.second[i], op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                           op_desc->GetOutputsSize());
        GELOGE(PARAM_INVALID, "fusion output index %d must less than outputs desc size %zu.", fusion_output.second[i],
               op_desc->GetOutputsSize());
        return PARAM_INVALID;
      }

      ret = ge::AttrUtils::SetStr(op_desc->MutableOutputDesc(fusion_output.second[i]),
                                  ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, fusion_output.first);
      if (!ret) {
        GELOGW("Set %s to %s %d output fail.", ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME.c_str(), op_desc->GetName().c_str(),
               fusion_output.second[i]);
      }

      ret = ge::AttrUtils::SetInt(op_desc->MutableOutputDesc(fusion_output.second[i]),
                                  ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, i);
      if (!ret) {
        GELOGW("Set %s to %s %d output fail.", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX.c_str(),
               op_desc->GetName().c_str(), fusion_output.second[i]);
      }
    }
  }

  return SUCCESS;
}

Status TensorFlowModelParser::SetOriginNodeContext(NodeDef *node_def, OpNodeContext &op_node_context,
                                                   const std::vector<std::pair<std::string, int32_t>> &inputs,
                                                   const std::vector<std::pair<std::string, int32_t>> &outputs) {
  int32_t in_index = 0;
  for (const auto &in : inputs) {
    bool is_ctrl = in.second == kControlSlot;
    op_node_context.input_map[in.first].emplace_back(std::make_pair(in.second, is_ctrl ? kControlSlot : in_index));
    SaveEdgesControlInfo(node_def->name(), is_ctrl);
    in_index = is_ctrl ? in_index : in_index + 1;
  }
  int32_t out_index = 0;
  for (const auto &out : outputs) {
    bool is_ctrl = out.second == kControlSlot;
    op_node_context.output_map[out.first].emplace_back(std::make_pair(is_ctrl ? kControlSlot : out_index, out.second));
    out_index = is_ctrl ? out_index : out_index + 1;
  }
  return SUCCESS;
}

void TensorFlowModelParser::GetFusionInputInfo(
    const string &fusion_op_name, OpNodeContext &fusion_context,
    std::map<string, std::pair<std::string, std::pair<int32_t, int32_t>>> &remap_data_input,
    std::map<string, std::vector<string>> &remap_ctrl_input, std::set<string> &fusion_input_nodes) {
  for (const auto &fusion_input : fusion_context.input_map) {
    string fusion_src_name = fusion_input.first;
    for (const auto &fusion_idx_pair : fusion_input.second) {
      string key = fusion_op_name + std::to_string(fusion_idx_pair.second);
      if (fusion_idx_pair.second != kControlSlot) {
        remap_data_input[key] = {fusion_src_name, {fusion_idx_pair.first, fusion_idx_pair.second}};
      } else {
        remap_ctrl_input[key].emplace_back(fusion_src_name);
      }
    }
    fusion_input_nodes.insert(fusion_src_name);
  }
}

void TensorFlowModelParser::GetFusionOutputInfo(
    const string &fusion_op_name, OpNodeContext &fusion_context,
    std::map<string, std::vector<std::pair<std::string, std::pair<int32_t, int32_t>>>> &remap_data_output,
    std::map<string, std::vector<string>> &remap_ctrl_output, std::set<string> &fusion_output_nodes) {
  for (const auto &fusion_output : fusion_context.output_map) {
    string fusion_dst_name = fusion_output.first;
    for (const auto &fusion_idx_pair : fusion_output.second) {
      string key = fusion_op_name + std::to_string(fusion_idx_pair.first);
      if (fusion_idx_pair.first != kControlSlot) {
        remap_data_output[key].emplace_back(
            std::make_pair(fusion_dst_name, std::make_pair(fusion_idx_pair.first, fusion_idx_pair.second)));
      } else {
        remap_ctrl_output[key].emplace_back(fusion_dst_name);
      }
    }
    fusion_output_nodes.insert(fusion_dst_name);
  }
}

void TensorFlowModelParser::UpdateInnerInputMap(const string &fusion_op_name, OpNodeContext &fusion_context,
                                                const std::vector<std::string> &inner_nodes_name,
                                                std::set<string> &fusion_input_nodes) {
  std::map<string, std::pair<std::string, std::pair<int32_t, int32_t>>> remap_data_input;
  std::map<string, std::vector<string>> remap_ctrl_input;
  GetFusionInputInfo(fusion_op_name, fusion_context, remap_data_input, remap_ctrl_input, fusion_input_nodes);

  for (const auto &node_name : inner_nodes_name) {
    auto context_iter = op_node_context_map_.find(node_name);
    if (context_iter != op_node_context_map_.end()) {
      OpNodeContext &op_node_context = context_iter->second;
      // update input map of inner node
      std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> tmp_input_map;
      for (auto iter = op_node_context.input_map.begin(); iter != op_node_context.input_map.end();) {
        string src_name = iter->first;
        std::vector<std::pair<int32_t, int32_t>> &input_idx = iter->second;
        if (src_name == ge::kInputFromFusionScope) {
          for (const auto &in_pair : input_idx) {
            if (in_pair.second != kControlSlot) {
              auto data = remap_data_input[fusion_op_name + std::to_string(in_pair.first)];
              tmp_input_map[data.first].emplace_back(std::make_pair(data.second.first, in_pair.second));
              GELOGI("Update inner input, src:%s, idx:%u->%u", data.first.c_str(), data.second.first, in_pair.second);
            }
          }
          auto ctrl = remap_ctrl_input[fusion_op_name + std::to_string(kControlSlot)];
          for (const auto &ctrl_in : ctrl) {
            tmp_input_map[ctrl_in].emplace_back(std::make_pair(kControlSlot, kControlSlot));
            SaveEdgesControlInfo(node_name, kControlSlot);
          }
          iter = op_node_context.input_map.erase(iter);
        } else {
          ++iter;
        }
      }
      op_node_context.input_map.insert(tmp_input_map.begin(), tmp_input_map.end());
      // update output map of pre node
      for (const auto &in_iter : op_node_context.input_map) {
        auto src_iter = op_node_context_map_.find(in_iter.first);
        if (src_iter != op_node_context_map_.end()) {
          std::vector<std::pair<int32_t, int32_t>> input_pairs = in_iter.second;
          OpNodeContext &src_context = src_iter->second;
          src_context.output_map[node_name].assign(input_pairs.begin(), input_pairs.end());
        }
      }
    }
  }
}

void TensorFlowModelParser::UpdateInnerOutputMap(const string &fusion_op_name, OpNodeContext &fusion_context,
                                                 const std::vector<std::string> &inner_nodes_name,
                                                 std::set<string> &fusion_output_nodes) {
  std::map<string, std::vector<std::pair<std::string, std::pair<int32_t, int32_t>>>> remap_data_output;
  std::map<string, std::vector<string>> remap_ctrl_output;
  GetFusionOutputInfo(fusion_op_name, fusion_context, remap_data_output, remap_ctrl_output, fusion_output_nodes);
  for (const auto &node_name : inner_nodes_name) {
    auto context_iter = op_node_context_map_.find(node_name);
    if (context_iter != op_node_context_map_.end()) {
      OpNodeContext &op_node_context = context_iter->second;
      // update output map of inner node
      std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> tmp_output_map;
      for (auto iter = op_node_context.output_map.begin(); iter != op_node_context.output_map.end();) {
        string dst_name = iter->first;
        std::vector<std::pair<int32_t, int32_t>> &output_idx = iter->second;
        if (dst_name == ge::kOutputToFusionScope) {
          for (const auto &out_pair : output_idx) {
            if (out_pair.second != kControlSlot) {
              auto data_outputs = remap_data_output[fusion_op_name + std::to_string(out_pair.second)];
              for (const auto &data : data_outputs) {
                tmp_output_map[data.first].emplace_back(std::make_pair(out_pair.first, data.second.second));
                GELOGI("Update inner output, dst:%s, idx:%u->%u.", data.first.c_str(), out_pair.first,
                       data.second.second);
              }
            }
          }
          auto ctrl = remap_ctrl_output[fusion_op_name + std::to_string(kControlSlot)];
          for (const auto &ctrl_in : ctrl) {
            tmp_output_map[ctrl_in].emplace_back(std::make_pair(kControlSlot, kControlSlot));
          }
          iter = op_node_context.output_map.erase(iter);
        } else {
          ++iter;
        }
      }
      op_node_context.output_map.insert(tmp_output_map.begin(), tmp_output_map.end());
      // update input map of pre node
      for (const auto &out_iter : op_node_context.output_map) {
        auto dst_iter = op_node_context_map_.find(out_iter.first);
        if (dst_iter != op_node_context_map_.end()) {
          std::vector<std::pair<int32_t, int32_t>> output_pairs = out_iter.second;
          OpNodeContext &dst_context = dst_iter->second;
          dst_context.input_map[node_name].assign(output_pairs.begin(), output_pairs.end());
        }
      }
    }
  }
}

Status TensorFlowModelParser::UpdateInnerNodeContext(const string &fusion_op_name,
                                                     const std::vector<std::string> &inner_nodes_name) {
  auto fusion_iter = op_node_context_map_.find(fusion_op_name);
  if (fusion_iter == op_node_context_map_.end()) {
    REPORT_INNER_ERROR("E19999", "Node:%s can't find in op_node_context_map_, check invalid", fusion_op_name.c_str());
    GELOGE(INTERNAL_ERROR, "Can't find context for fusion node %s.", fusion_op_name.c_str());
    return INTERNAL_ERROR;
  }
  OpNodeContext &fusion_context = fusion_iter->second;
  std::set<string> fusion_input_nodes;
  std::set<string> fusion_output_nodes;
  UpdateInnerInputMap(fusion_op_name, fusion_context, inner_nodes_name, fusion_input_nodes);
  UpdateInnerOutputMap(fusion_op_name, fusion_context, inner_nodes_name, fusion_output_nodes);
  for (const auto &in_name : fusion_input_nodes) {
    auto fusion_in = op_node_context_map_.find(in_name);
    if (fusion_in != op_node_context_map_.end()) {
      OpNodeContext &fusion_in_context = fusion_in->second;
      fusion_in_context.output_map.erase(fusion_op_name);
    }
  }
  for (const auto &out_name : fusion_output_nodes) {
    auto fusion_out = op_node_context_map_.find(out_name);
    if (fusion_out != op_node_context_map_.end()) {
      OpNodeContext &fusion_out_context = fusion_out->second;
      fusion_out_context.input_map.erase(fusion_op_name);
    }
  }
  op_node_context_map_.erase(fusion_op_name);
  return SUCCESS;
}

Status TensorFlowModelParser::AddFusionInnerNodeDef(shared_ptr<ge::ScopeGraph> &scope_graph,
                                                    const string &fusion_op_name, vector<string> &node_name_list) {
  auto &impl_scope_graph = scope_graph->impl_;
  GE_CHECK_NOTNULL(impl_scope_graph);
  ge::FusionScopesResult *fusion_result = impl_scope_graph->GetFusionScopesResults(fusion_op_name);
  GE_CHECK_NOTNULL(fusion_result);
  auto &impl_fusion_rlt = fusion_result->impl_;
  GE_CHECK_NOTNULL(impl_fusion_rlt);
  ge::FusionInnerNodesInfo inner_nodes_info = impl_fusion_rlt->GetInnerNodesInfo();
  vector<string> inner_nodes_name;
  for (const auto &info : inner_nodes_info) {
    string node_name;
    string type;
    std::vector<std::pair<std::string, int32_t>> inputs;
    std::vector<std::pair<std::string, int32_t>> outputs;
    const ge::Operator *op = nullptr;
    std::tie(node_name, type, inputs, outputs, op) = info;
    NodeDef *node_def = new (std::nothrow) NodeDef();
    GE_CHECK_NOTNULL(node_def);
    node_def->set_name(node_name);
    node_def->set_op(type);
    nodedef_map_[node_name] = node_def;
    fusion_nodedef_list.push_back(node_def);
    for (const auto &in : inputs) {
      // The input value is not used in the subsequent process. The value is added only for placeholders.
      node_def->add_input(in.first);
    }
    domi::tensorflow::AttrValue attr_value;
    attr_value.set_b(true);
    ge::TensorFlowUtil::AddNodeAttr(kAttrNameIsScopeInnerNode, attr_value, node_def);
    OpNodeContext &op_node_context = op_node_context_map_[node_name];
    Status ret = SetOriginNodeContext(node_def, op_node_context, inputs, outputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "Failed to add context and attrs, node:%s.", node_name.c_str());
      return ret;
    }
    scope_inner_node_map_.insert({node_name, op});
    node_name_list.emplace_back(node_name);
    inner_nodes_name.emplace_back(node_name);
    GELOGI("Add fusion inner node def, name:%s, type:%s.", node_name.c_str(), type.c_str());
  }
  Status ret = UpdateInnerNodeContext(fusion_op_name, inner_nodes_name);
  if (ret != SUCCESS) {
    GELOGE(ret, "Failed to update inner node context, fusion_op_name:%s.", fusion_op_name.c_str());
    return ret;
  }
  return SUCCESS;
}

Status TensorFlowModelParser::AddFusionNodeDef(shared_ptr<ge::ScopeGraph> &scope_graph,
                                               vector<string> &node_name_list) {
  vector<string> node_name_list_new;
  size_t op_node_list_size = node_name_list.size();
  DumpAllNodeContext("BeforeAddFusionNodeDef");
  for (size_t i = 0; i < op_node_list_size; ++i) {
    const string op_node_name = node_name_list[i];
    auto iter = fusion_op_nodedef_map_.find(op_node_name);
    if (iter != fusion_op_nodedef_map_.end()) {
      vector<string> fusion_op_info = fusion_op_type_map_[op_node_name];
      if (fusion_op_info[0] != ge::kScopeToMultiNodes) {
        NodeDef *node_def = new (std::nothrow) NodeDef();
        GE_CHECK_NOTNULL(node_def);
        node_def->set_name(op_node_name);
        node_def->set_op(fusion_op_info[0]);
        nodedef_map_[op_node_name] = node_def;
        fusion_nodedef_list.push_back(node_def);
        OpNodeContext &node_context = op_node_context_map_[node_def->name()];
        for (const auto &input : node_context.input_map) {
          // The input value is not used in the subsequent process. The value is added only for placeholders.
          node_def->add_input(input.first);
        }
        node_name_list_new.emplace_back(op_node_name);
        GELOGI("Add Fusion node def, name:%s, type:%s.", node_def->name().c_str(), node_def->op().c_str());
      } else {
        Status ret = AddFusionInnerNodeDef(scope_graph, op_node_name, node_name_list_new);
        if (ret != SUCCESS) {
          REPORT_INNER_ERROR("E19999", "Failed to add fusion inner nodes for fusion op:%s, "
                             "please check FusionScopesResult set in scope fusion pass", op_node_name.c_str());
          GELOGE(ret, "Failed to add fusion inner node, fusion_op_name:%s.", op_node_name.c_str());
          return ret;
        }
        GELOGI("Add fusion inner nodes successfully, fusion name:%s.", op_node_name.c_str());
        op_node_context_map_.erase(op_node_name);
      }
    } else {
      node_name_list_new.emplace_back(op_node_name);
    }
  }
  node_name_list.clear();
  node_name_list.assign(node_name_list_new.begin(), node_name_list_new.end());
  DumpAllNodeContext("AfterAddFusionNodeDef");
  return SUCCESS;
}

Status TensorFlowModelParser::AddScopeInnerNode(TensorFlowModelParser *parser, ge::ComputeGraphPtr &graph,
                                                std::mutex *graph_mutex, const domi::tensorflow::NodeDef *node_def) {
  // This is an internal function. The pointer input parameter is not empty when this function is invoked.
  string node_name = node_def->name();
  string node_op = node_def->op();
  auto iter = parser->scope_inner_node_map_.find(node_name);
  if (iter == parser->scope_inner_node_map_.end()) {
    REPORT_INNER_ERROR("E19999", "Node:%s can't find in scope_inner_node_map_, check invalid", node_name.c_str());
    GELOGE(PARAM_INVALID, "Failed to find scope inner node:%s, type:%s.", node_name.c_str(), node_op.c_str());
    return PARAM_INVALID;
  }
  const ge::Operator *op = iter->second;
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  GE_CHECK_NOTNULL(op_desc);
  ge::NodePtr node;
  {
    std::lock_guard<std::mutex> lock(*graph_mutex);
    node = graph->AddNode(op_desc);
  }
  if (node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "Failed to Add scope inner node:%s, type:%s.", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  {
    std::lock_guard<std::mutex> lock(parser->nodeMapMutex_);
    parser->node_map_[node_name] = node;
  }
  GELOGI("Add scope inner node successfully, node name:%s, type:%s.", op_desc->GetName().c_str(),
         op_desc->GetType().c_str());
  return SUCCESS;
}

void TensorFlowModelParser::DumpNodeContext(const string &node_name, const OpNodeContext &ctx, const string &phase) {
  GELOGD("phase:%s === Begin to dump context for node:%s ===", phase.c_str(), node_name.c_str());
  for (const auto &input : ctx.input_map) {
    for (const auto &input_idx : input.second) {
      GELOGD("  Input info: %s:%d --> in_idx %d.", input.first.c_str(), input_idx.first, input_idx.second);
    }
  }
  for (const auto &output : ctx.output_map) {
    for (const auto &output_idx : output.second) {
      GELOGD("  Output info: out_idx %d --> %s:%d.", output_idx.first, output.first.c_str(), output_idx.second);
    }
  }
  GELOGD("phase:%s === End to dump context for node:%s ===", phase.c_str(), node_name.c_str());
}

void TensorFlowModelParser::DumpAllNodeContext(const string &phase) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }
  for (const auto &iter : op_node_context_map_) {
    DumpNodeContext(iter.first, iter.second, phase);
  }
}

Status TensorFlowModelParser::CheckAndUpdateInputDesc(ge::ComputeGraphPtr &compute_graph) {
  GE_CHECK_NOTNULL(compute_graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (auto &in_anchor : node->GetAllInDataAnchors()) {
      if (!(op_desc->IsOptionalInput(static_cast<uint32_t>(in_anchor->GetIdx())))) {
        continue;
      }
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      auto in_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(in_anchor->GetIdx()));
      if ((peer_out_anchor != nullptr) && (in_desc == nullptr)) {
        // The input is connected to the peer output but TensorDesc is invalid, update TensorDesc to valid.
        ge::GeTensorDesc tensor_desc;
        auto ret = op_desc->UpdateInputDesc(static_cast<uint32_t>(in_anchor->GetIdx()), tensor_desc);
        if (ret != ge::GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Update index:%d of input desc in op:%s(%s) failed", in_anchor->GetIdx(),
                            op_desc->GetName().c_str(), op_desc->GetType().c_str());
          GELOGE(ret, "Failed to update input desc, node:%s, index:%d.", node->GetName().c_str(), in_anchor->GetIdx());
          return ret;
        }
        GELOGI("Update input desc to valid, node:%s, index:%d.", node->GetName().c_str(), in_anchor->GetIdx());
      } else if ((peer_out_anchor == nullptr) && (in_desc != nullptr)) {
        // The input is not connected to the peer output but TensorDesc is valid, update TensorDesc to invalid.
        ge::GeTensorDesc tensor_desc(ge::GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
        auto ret = op_desc->UpdateInputDesc(static_cast<uint32_t>(in_anchor->GetIdx()), tensor_desc);
        if (ret != ge::GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Update index:%d of input desc in op:%s(%s) failed", in_anchor->GetIdx(),
                            op_desc->GetName().c_str(), op_desc->GetType().c_str());
          GELOGE(ret, "Failed to update input desc, node:%s, index:%d.", node->GetName().c_str(), in_anchor->GetIdx());
          return ret;
        }
        GELOGI("Update input desc to invalid, node:%s, index:%d.", node->GetName().c_str(), in_anchor->GetIdx());
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge

namespace domi {
REGISTER_MODEL_PARSER_CREATOR(TENSORFLOW, ge::TensorFlowModelParser);
REGISTER_WEIGHTS_PARSER_CREATOR(TENSORFLOW, ge::TensorFlowWeightsParser);
}  // namespace domi
