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

#ifndef PARSER_TENSORFLOW_TENSORFLOW_PARSER_H_
#define PARSER_TENSORFLOW_TENSORFLOW_PARSER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/range_vistor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/parser/model_parser.h"
#include "omg/parser/op_parser.h"
#include "omg/parser/weights_parser.h"
#include "parser/tensorflow/tensorflow_fusion_op_parser.h"
#include "parser/tensorflow/tensorflow_fusionop_util.h"
#include "parser/tensorflow/tensorflow_util.h"
#include "proto/om.pb.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "proto/tensorflow/graph_library.pb.h"
#include "external/register/scope/scope_fusion_pass_register.h"
#include "scope/scope_pass_manager.h"

using ge::ScopePassManager;
using domi::tensorflow::GraphDef;
using domi::tensorflow::DT_HALF;
using domi::tensorflow::NodeDef;
using domi::tensorflow::GraphDef;
using domi::tensorflow::AttrValue;
using domi::tensorflow::DataType;
using ge::OpParser;

namespace ge {
using std::string;
using std::vector;
using std::set;
using std::map;
using std::unordered_map;
using std::mutex;
using std::shared_ptr;

enum TfTranspose { TO_NCHW, TO_NHWC, NO_TRANSPOSE };

struct OpNodeContext {
  // save <name,indexlist> for input
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> input_map;
  // save <name,index> for output
  std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> output_map;
};

struct DelTransposeInfo;
class PARSER_FUNC_VISIBILITY TensorFlowModelParser : public domi::ModelParser {
 public:
  TensorFlowModelParser() {}
  virtual ~TensorFlowModelParser() {}

  /**
   * @ingroup domi_omg
   * @brief Parse the relevant data from the model file and save it to graph
   * @param [in] file Path of the model file
   * @param [in|out] graph save model information after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed

   */
  Status Parse(const char *file, ge::Graph &graph) override;

  /**
   * @ingroup domi_omg
   * @brief Parse the relevant data from memory and save it to graph
   * @param [in] memory buffer of model file
   * @param [in] buffer size
   * @param [in|out] graph graph for saving model information
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) override;

  Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph) override {
    return domi::SUCCESS;
  }

  /**
   * @ingroup domi_omg
   * @brief Convert model files to JSON format
   * @param [in] model_file  Model file path to be converted
   * @param [out] json_file Converted JSON file path
   * @return SUCCESS parse successfully
   * @return others parse failed
   */
  Status ToJson(const char *model_file, const char *json_file) override;

  /**
  * @ingroup domi_omg
  * @brief Parse the relevant data from the model file and save it to graph
  * @param [in] graph_def input tensorflow model
  * @param [in|out] graph save model informati:on after parsing
  * @return SUCCESS parse successfully
  * @return FAILED parse failed
  */
  Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) override;

  Status ParseProtoWithSubgraph(const google::protobuf::Message *root_proto,
                                domi::GetGraphCallback callback,
                                ge::ComputeGraphPtr &graph) override;

  /*
  * @ingroup domi_omg
  * @brief Mapping TF's datatype to GE's datatype
  * @param [in] type, datatype types of operators in TF networks
  * @return ge::DataType
  */
  ge::DataType ConvertToGeDataType(const uint32_t type) override;

  Status ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) override ;

  /**
   * @ingroup domi_omg
   * @brief Analyze network model data
   * @param [in] proto  serialized network model
   * @param [in|out]  graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  Status ParseProto(const std::string &serialized_proto, ge::ComputeGraphPtr &graph) override;

  /**
   * @ingroup domi_omg
   * @brief Analyze callback model data in subgraph
   * @param [in] proto serialized network model
   * @param [in] callback callback of subgraph
   * @param [in|out] graph Save the network information after analysis
   * @return SUCCESS
   * @return Others failed
   */
  Status ParseProtoWithSubgraph(const std::string &serialized_proto, domi::GetGraphCallbackV2 callback,
                                ge::ComputeGraphPtr &graph) override;
 private:
  Status Parse(const char *file, ge::ComputeGraphPtr &graph);

  /**
   * @ingroup domi_omg
   * @brief Add node information to graph
   * @param [in|out] op_node_name_list
   * @param [in|out] graph save model information after parsing
   * @return SUCCESS add successfully
   * @return FAILED add failed

   */
  Status AddFmkNode(ge::ComputeGraphPtr &graph, shared_ptr<ge::ScopeGraph> &scope_graph,
                    vector<string> &op_node_name_list, bool is_dataset_init = false);

  Status AddNodeToGraphAndMarkFormat(ge::ComputeGraphPtr &graph, const vector<string> &op_node_name_list);

  /**
   * @ingroup domi_omg
   * @brief Add node def into node map
   * @param NodeDef*
   * @return SUCCESS add successfully
   * @return FAILED add failed

   */
  Status AddFmkNodeDefToMap(const domi::tensorflow::GraphDef &graph_def, const domi::tensorflow::NodeDef *node_def,
                            vector<string> &op_node_name_list);

  /**
   * @ingroup domi_omg
   * @brief Add node information to graph
   * @param [in] layer layer infomation
   * @param [in|out] graph save model information after parsing
   * @return SUCCESS add successfully
   * @return FAILED add failed

   */
  Status AddNode(const domi::tensorflow::NodeDef *node_def,
                 ge::ComputeGraphPtr &graph,
                 shared_ptr<ge::ScopeGraph> &scope_graph);
  /**
   * @ingroup domi_omg
   * @brief Add edge information to graph
   * @param [in|out] graph save model information after parsing
   * @return SUCCESS add successfully
   * @return FAILED add failed

   */
  Status AddEdges(ge::ComputeGraphPtr &graph);

  /**
  * @ingroup domi_omg
  * @brief get op context from the parsed graph
  */
  Status GetOpNodesContextFromGraph(const domi::tensorflow::GraphDef &graph_def);

  /**
  * @ingroup domi_omg
  * @brief get input，include opNode and constNode
  * @param [in] op_node_name op name
  * @param [out] input_map input node and index
  * @return SUCCESS get successfully
  * @return FAILED get failed
  */
  Status GetOpNodeInputMap(const string &op_node_name,
                           map<string, std::vector<std::pair<int32_t, int32_t>>> &input_map);

  /**
  * @ingroup domi_omg
  * @brief get output of node
  * @param [in] graph_def graph
  * @return SUCCESS get successfully
  * @return FAILED get failed
  */
  Status GetOpNodeOutputMap(const domi::tensorflow::GraphDef &graph_def);

  /**
  * @ingroup domi_omg
  * @brief Verifying the validity of graphdef object parsed by pb
  * @param [in] graph_def Parsed tensorflow:: graphdef object
  * @return SUCCESS check successfully
  * @return FAILED check failed
  */
  Status CheckGraphDefValid(const domi::tensorflow::GraphDef &graph_def);

  /**
  * @ingroup domi_omg
  * @brief whether const OP need to update context
  * @param const op name
  * @return true or false
  */
  bool ConstOpNeedUpdate(const string &op_name);


  Status ExcuteScopeFusionPasses(domi::tensorflow::GraphDef *graph_def, shared_ptr<ge::ScopeGraph> &scope_graph);
  /**
  * @ingroup domi_omg
  * @brief Run the scope fusion optimizer in list scope_passes_list
  * @param [in] scope_passes_list optimizer list
  * @param [in/out] pass_manager an object to manager the optimizers
  * @param [in/out] scope_graph Save the result of scope fusion
  * @return SUCCESS Run successfully
  * @return others  Run failed
  */
  Status RunScopeFusionPass(const vector<string> &scope_passes_list,
                            ScopePassManager &pass_manager,
                            shared_ptr<ge::ScopeGraph> &scope_graph);

  /**
  * @ingroup domi_omg
  * @brief Check whether the nodedef parsed from pb is a fusion operator, put NodeDef into fusion_op_nodedef_map_
  * @param [in] graph_def Parsed tensorflow:: graphdef object
  * @return maybe a fusion operator
  */
  bool MaybeFusionOp(shared_ptr<ge::ScopeGraph> &scope_graph, const domi::tensorflow::NodeDef *node_def);

  /**
  * @Confirm whether it is a child operator of the fusion operator
  */
  bool IsFusionOpChild(const string &node_name, ge::ScopeFusionOpInfo *info);

  /**
  * @brief Inner child operators of fusion operators
  */
  bool FusionOpChildIgnore(shared_ptr<ge::ScopeGraph> &scope_graph, const ge::ScopeFusionOpInfo &info);

  // Is it a fusion operator
  bool IsFusionOp(shared_ptr<ge::ScopeGraph> &scope_graph, const domi::tensorflow::NodeDef *node_def);

  /**
  * @brief get inPut index of the fusion operator
  * @param [in] info Child node description of fusion operator
  * @param [in] old_index Child node original index
  * @return old_index as input index of the fusion operator
  * @return return code
  */
  static Status GetInPutIndex(shared_ptr<ge::ScopeGraph> &scope_graph,
                              const ge::ScopeFusionOpInfo &info,
                              const int32_t old_index,
                              int32_t &new_index);

  /**
  * @brief get output index of the fusion operator
  * @param [in] info Child node description of fusion operator
  * @param [in] old_index Child node original index
  * @return old_index as output index of the fusion operator
  * @return return code
  */
  static Status GetOutPutIndex(shared_ptr<ge::ScopeGraph> &scope_graph,
                               const ge::ScopeFusionOpInfo &info,
                               const int32_t old_index,
                               int32_t &new_index);
  /**
    * @ingroup domi_omg
    * @brief Check the validity of fusionop，put it into op_node_name_list if Misjudgement
    * @param op_node_name_list
    * @return SUCCESS check successfully
    * @return FAILED check failed

    */
  Status CheckFusionOpValid();

  /**
   * @ingroup domi_omg
   * @brief Update input-output relationships of all operators
   * @param graph_def和op_node_name_list
   * @return SUCCESS
   * @return FAILED

   */
  Status UpdateAllNodeOpContext(shared_ptr<ge::ScopeGraph> &scope_graph, const domi::tensorflow::GraphDef &graph_def,
                                vector<string> &op_node_name_list);

  /**
   * @ingroup domi_omg
   * @brief Updating the input-output relationship of fusion operators
   * @param info Description of fusion operator
   * @param fusion_op_node_context  Input-output relationship of fusion operator
   * @param normal_op_node_context  Input-output relationship of normal operators
   * @return SUCCESS
   * @return FAILED

   */
  Status UpdateFusionOpContext(shared_ptr<ge::ScopeGraph> &scope_graph, const ge::ScopeFusionOpInfo &info,
                               OpNodeContext &fusion_op_node_context, OpNodeContext &normal_op_node_context);

  /**
   * @ingroup domi_omg
   * @brief Updating the input-output relationship of normal operators
   * @param normal_op_node_context Input-output relationship of normal operators
   * @return SUCCESS
   * @return FAILED

   */
  Status UpdateNormalOpContext(shared_ptr<ge::ScopeGraph> &scope_graph, const string &op_node_name,
                               OpNodeContext &normal_op_node_context);

  Status EraseNormalOpOutputIfChild(shared_ptr<ge::ScopeGraph> &scope_graph, const string &op_node_name,
                                    OpNodeContext &normal_op_node_context);

  /**
   * @ingroup domi_omg
   * @brief Normalized I / O relationship: de duplication and de outliers

   */
  Status NormalizeAllNodeOpContext();

  /**
   * @ingroup domi_omg
   * @brief Normalized I / O relationship: according to context map, de duplicate and de outliers

   */
  Status NormalizeInputOrOutputMap(std::map<std::string, std::vector<std::pair<int32_t, int32_t>>> &context_map);

  /**
   * @ingroup domi_omg
   * @brief delete fusionNodeDef

   */
  void DeleteFuisonNodeDef();

  /**
   * @ingroup domi_omg
   * @brief Save the control attribute to edges control map

   */
  void SaveEdgesControlInfo(const string &node_name, const bool control);

  /**
   * @ingroup domi_omg
   * @brief Update the control property to edges control map

   */
  void UpdateEdgesControlInfo(const ge::ScopeFusionOpInfo &info);

  /**
   * @ingroup domi_omg
   * @brief get contral information

   */
  bool GetEdgesControlInfo(const string &node_name, const int32_t index);

  /**
   * @ingroup domi_omg
   * @brief Check the validity of input_name
   * @param input_node_name，Consider the input: n scenario
   * @param index ,return index，"input":return 0，"input:n":return n
   * @param index ,control index， input: "^cond/switch_t"
   * @return SUCCESS
   * @return FAILED

   */
  Status CheckInputNodeName(const string &input_node_name, string *node_name, int32_t *index, bool *control);

  /**
   * @ingroup domi_omg
   * @brief ge stoi
   * @param input_node_name，Consider the input: n scenario
   * @param index_str ,stoi param
   * @param index ,return index，"input":return 0，"input:n":return n
   * @return SUCCESS
   * @return FAILED

 */
  Status GeStoi(const string &input_node_name, const string &index_str, int32_t *index);

  /**
   * @ingroup domi_omg
   * @brief Clearing the error information of non key operators in fusion operators

   */
  Status ClearFusionOpError(const vector<string> &op_node_name_list);

  /**
  * @ingroup domi_omg
  * @brief Delete the connection relationship of the identity operator connecting the Arg node in graphdef
  */
  Status GraphDefOptimize(domi::tensorflow::GraphDef *graph_def);
  /**
  * @ingroup domi_omg
  * @brief Optimize for Identity/ReadVariableOp operator
  * @param [in] graph_def GraphDef to be optimized
  * @param [in] nodedef_map Map of all nodes in graph
  * @param [in] nodedef_to_optimize vector of NodeDef to be optimized
  * @return SUCCESS  optimize successfully
  * @return others   failed
  */
  Status GraphDefOptimizeIdentity(domi::tensorflow::GraphDef *graph_def, map<string, NodeDef *> &nodedef_map,
                                  const vector<NodeDef *> &nodedef_to_optimize);
  /**
  * @ingroup domi_omg
  * @brief For the identity operator whose output is "_retval", optimize it.
  * @param [in] nodedef_map Map of all nodes in graph
  * @param [in] curr_node_name Name of node to be optimized
  * @param [in] clear_input_flag Flag of whether to clear the input of the current node
  * @return SUCCESS  optimize successfully
  * @return others   failed
  */
  Status OptimizeIdentityByOutput(map<string, NodeDef *> &nodedef_map, const string &curr_node_name,
                                  bool &clear_input_flag);
  Status GraphDefOptimizeSnapShot(domi::tensorflow::GraphDef *graph_def, map<string, NodeDef *> &nodedef_map,
                                  const vector<NodeDef *> &nodedef_to_optimize);
  Status GraphDefOptimizeDestroyTemporaryVariable(domi::tensorflow::GraphDef *graph_def,
                                                  domi::tensorflow::NodeDef *nodeCurrent);
  Status OptimizeSnapShot(domi::tensorflow::NodeDef *curr_mode_def, map<string, NodeDef *> &nodedef_map,
                          const std::pair<string, int> &input_data, const std::vector<string> &control_list);
  void OptimizeDestroyTemporaryVariable(domi::tensorflow::GraphDef *graph_def, domi::tensorflow::NodeDef *nodeCurrent,
                                        bool &clearInputFlag);
  void OptimizeTranspose(std::map<std::string, DelTransposeInfo> &transposeInfo);
  void SoftmaxAddAttr(GraphDef *graph_def);

  /**
  * @ingroup domi_omg
  * @brief Delete isolated nodes in graph
  */
  Status RemoveIsolateNode(ge::ComputeGraphPtr &graph);

  /**
   * @ingroup domi_omg
   * @brief Infer format for input ops.

   */
  domiTensorFormat_t InferInputFormats();

  /**
   * @ingroup domi_omg
   * @brief Get node format.

   */
  Status GetNodeFormat(const NodeDef *node, TfTranspose pred_transpose, domiTensorFormat_t &format,
                       set<const NodeDef *> &visited_node);

  /**
   * @ingroup domi_omg
   * @brief Get format transpose.

   */
  Status GetFormatTranspose(const NodeDef *transpose_node, TfTranspose &transpose_direc);
  Status TrimGraph(const domi::tensorflow::GraphDef &input_graph_def, domi::tensorflow::GraphDef *output_graph_def);
  Status TrimGraphByInput(const domi::tensorflow::GraphDef &input_graph_def,
                          domi::tensorflow::GraphDef *output_graph_def);
  Status TrimGraphByOutput(const domi::tensorflow::GraphDef &input_graph_def,
                           domi::tensorflow::GraphDef *output_graph_def);
  string NodeNameFromInput(const string &input_name);

  Status AddTensorDescToOpDesc(ge::OpDescPtr &op_desc, const domi::tensorflow::NodeDef *node);
  Status CheckoutInputNum(ge::OpDescPtr &op_desc, const domi::tensorflow::NodeDef *node);
  void UpdateInputTensor(ge::OpDescPtr &op_desc, const std::vector<ge::GeTensorDesc> &input_desc,
                         const size_t input_tensor_num);
  void UpdateOutputTensor(ge::OpDescPtr &op_desc, const std::vector<ge::GeTensorDesc> &output_desc,
                          size_t output_tensor_num);
  Status TransNodeToOpDesc(const domi::tensorflow::NodeDef *node_def, ge::OpDescPtr &op, const string &op_type);

  Status UppdateInputMap(shared_ptr<ge::ScopeGraph> &scope_graph, const ge::ScopeFusionOpInfo &info,
                         OpNodeContext &fusion_op_node_context, OpNodeContext &normal_op_node_context);
  Status UppdateOutputMap(shared_ptr<ge::ScopeGraph> &scope_graph, const ge::ScopeFusionOpInfo &info,
                          OpNodeContext &fusion_op_node_context, OpNodeContext &normal_op_node_context);
  void GetInputOutputTensorNum (ge::OpDescPtr &op_desc, size_t &input_tensor_num, size_t &output_tensor_num) const;
  Status CheckOpShapeDim(const domi::tensorflow::NodeDef *node_def, const std::set<int> &dims, bool &valid);
  Status CheckOpType(const domi::tensorflow::NodeDef *node_def, string &op_type);

  /**
   * @ingroup domi_omg
   * @brief Trans common decorate function to PartitionedCall.
   * @param [in] node_def:  Node of common function.
   * @param [out] op: result of PartitionedCall OpDesc.
   * @return 0: SUCCESS / Others: FAILED
   */
  Status DefunToPartitionedCall(const domi::tensorflow::NodeDef *node_def, ge::OpDescPtr &op);

  /**
   * @ingroup domi_omg
   * @brief Calling ParseParams method of fusion operator
   * @param op_parser，op parser of the fusion operator
   * @return SUCCESS
   * @return FAILED

   */
  Status FusionNodeParseParams(shared_ptr<OpParser> &op_parser,
                               const domi::tensorflow::NodeDef *node_def, ge::NodePtr &node);

  /**
   * @ingroup domi_omg
   * @brief Optimizing const nodes for custom operators
   * @param [in] graph_def graph object
   * @return true optimize successfully
   * @return false optimize failed
   *
   */
  Status OptimizeConstNodes4CustomOp(domi::tensorflow::GraphDef *graph_def);

  /**
   * @ingroup domi_omg
   * @brief Delete input from nodedef
   * @param [in] node_def Nodedef object
   * @param [in] remove_index_set Index collection of input nodes to be deleted
   * @return true remove successfully
   * @return false remove failed
   *
   */
  Status RemoveInputs(domi::tensorflow::GraphDef *graph_def,
                      domi::tensorflow::NodeDef *node_def,
                      const set<uint32_t> &remove_index_set,
                      const map<string, NodeDef *> &all_node_map);

  Status AddControlEdgeAfterRemoveInputs(domi::tensorflow::GraphDef *graph_def,
                                         domi::tensorflow::NodeDef *node_def,
                                         const map<string, NodeDef *> &all_node_map,
                                         const vector<string> &removed_inputs_vec);

  void RemoveInputAttr(domi::tensorflow::NodeDef *node_def, const map<string, vector<int>> &remove_inputs_map);

  /**
  * @ingroup domi_omg
  * @brief Parse the parameters in nodedef and construct Ge node.
  *        This function is a thread function，Parallel parse nodedef in tensorflow graph
  *        The member variables that need to be modified in this function should be locked
  * @param [in] parser TensorFlowModelParser
  * @param [in] graph  ge graph
  * @param [in] graphMutex ge graph lock
  * @param [in] scope_graph
  * @param [in] node_def Nodedef
  * @return SUCCESS
  * @return FAILED
  *
  */
  static Status ParseNodeDef(TensorFlowModelParser *parser, ge::ComputeGraphPtr &graph, std::mutex *graphMutex,
                             shared_ptr<ge::ScopeGraph> &scope_graph, const domi::tensorflow::NodeDef *node_def,
                             error_message::Context error_context);

  /**
  * @ingroup domi_omg
  * @brief adape op type
  * @param [in] node_def Nodedef
  * @param [in] isDatasetInit
  * @return SUCCESS adapt successfully
  * @return others adapt failed
  *
  */
  Status AdaptOpType(const domi::tensorflow::NodeDef *node_def, bool isDatasetInit);

  Status GetTensorflowGraphInOutMap(domi::tensorflow::GraphDef *graph_def);
  Status RemoveIsolateNode(domi::tensorflow::GraphDef *graph_def);
  static Status RecordFusionResult(std::shared_ptr<ge::ScopeGraph> &scope_graph,
                                   const domi::tensorflow::NodeDef *node,
                                   ge::OpDescPtr &op_def);

  Status GetFunctionProto(const string &file, domi::tensorflow::GraphDefLibrary &graph_def_library);

  Status SetOriginNodeContext(NodeDef *node_def, OpNodeContext &op_node_context,
                              const std::vector<std::pair<std::string, int32_t>> &inputs,
                              const std::vector<std::pair<std::string, int32_t>> &outputs);

  void GetFusionInputInfo(const string &fusion_op_name, OpNodeContext &fusion_context,
                          std::map<string, std::pair<std::string, std::pair<int32_t, int32_t>>> &remap_data_input,
                          std::map<string, std::vector<string>> &remap_ctrl_input,
                          std::set<string> &fusion_input_nodes);

  void GetFusionOutputInfo(const string &fusion_op_name, OpNodeContext &fusion_context,
      std::map<string, std::vector<std::pair<std::string, std::pair<int32_t, int32_t>>>> &remap_data_output,
      std::map<string, std::vector<string>> &remap_ctrl_output,
      std::set<string> &fusion_output_nodes);

  void UpdateInnerInputMap(const string &fusion_op_name, OpNodeContext &fusion_context,
                           const std::vector<std::string> &inner_nodes_name,
                           std::set<string> &fusion_input_nodes);

  void UpdateInnerOutputMap(const string &fusion_op_name, OpNodeContext &fusion_context,
                            const std::vector<std::string> &inner_nodes_name,
                            std::set<string> &fusion_output_nodes);

  Status UpdateInnerNodeContext(const string &fusion_op_name, const std::vector<std::string> &inner_nodes_name);

  Status AddFusionInnerNodeDef(shared_ptr<ge::ScopeGraph> &scope_graph,
                               const string &fusion_op_name,
                               vector<string> &node_name_list);

  Status AddFusionNodeDef(shared_ptr<ge::ScopeGraph> &scope_graph, vector<string> &node_name_list);

  static Status AddScopeInnerNode(TensorFlowModelParser *parser, ge::ComputeGraphPtr &graph,
                                  std::mutex *graph_mutex, const domi::tensorflow::NodeDef *node_def);

  void DumpNodeContext(const string &node_name, const OpNodeContext &ctx, const string &phase);
  void DumpAllNodeContext(const string &phase);

  Status ParseOpParams(const domi::tensorflow::NodeDef *node_def, ge::OpDescPtr &op, shared_ptr<OpParser> &op_parser);
  Status CheckAndUpdateInputDesc(ge::ComputeGraphPtr &compute_graph);

    /**
   * save <node_name, node_def>
   */
  unordered_map<string, const NodeDef *> nodedef_map_;

  /**
   * context, Input output relationship
   */
  unordered_map<string, OpNodeContext> op_node_context_map_;

  /**
   * Name of node of OP type, corresponding to node of DaVinci
   */
  std::unordered_map<std::string, ge::NodePtr> node_map_;

  /**
  * node_map_ Multithreaded write operation is involved, requiring lock protection
  */
  std::mutex nodeMapMutex_;

  /**
   * save <node_name, nodeDefList>
   */
  map<string, vector<const NodeDef *>> fusion_op_nodedef_map_;
  // Policy types of fusion operators,true:scope_pass match，false：prefix match
  map<string, bool> fusion_op_policy_;
  // The names of all children operators and the description of fusion operators
  unordered_map<string, ge::ScopeFusionOpInfo> fusion_op_children_;
  /**
   * save <node_name, {fusionOpName,description}>
   */
  map<string, vector<string>> fusion_op_type_map_;
  /**
   * save nodedef of the fusion operator
   */
  vector<domi::tensorflow::NodeDef *> fusion_nodedef_list;
  /**
   * control edge，{Key=NodeName,Value=index}
   */
  map<string, vector<int32_t>> edges_control_map;

  unordered_map<string, const domi::tensorflow::NodeDef *> framework_ops_;

  /**
   * save <node_name, op_type>
   */
  map<string, string> adaptedOpTypeMap_;

  // { node_name  <{input_node_name}, {output_node_name}> }
  map<string, std::pair<set<string>, set<string>>> node_inputs_outputs_map_;

  unordered_map<string, const ge::Operator *> scope_inner_node_map_;
};

/**
 * @ingroup domi_omg
 * @brief Tensorflow weight parse
 */
class PARSER_FUNC_VISIBILITY TensorFlowWeightsParser : public domi::WeightsParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief Parse weight data from file and save to graph
   * @param [in] file             Path of weight file after training
   * @param [in|out]              graph Save weight information after analysis
   * @return SUCCESS              parse successfully
   * @return PARAM_INVALID        param invalid
   * @return PARSE_WEIGHTS_FAILED parse failed
   */
  Status Parse(const char *file, ge::Graph &graph) override;

  Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) override;
};
}  // namespace domi
#endif  // PARSER_TENSORFLOW_TENSORFLOW_PARSER_H_
