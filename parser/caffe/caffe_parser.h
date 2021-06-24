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

#ifndef PARSER_CAFFE_CAFFE_PARSER_H_
#define PARSER_CAFFE_CAFFE_PARSER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "external/graph/operator.h"
#include "omg/parser/op_parser.h"
#include "omg/parser/model_parser.h"
#include "omg/parser/weights_parser.h"
#include "proto/caffe/caffe.pb.h"
#include "proto/om.pb.h"

namespace ge {
using domi::caffe::NetParameter;
using std::map;
using std::set;
using std::string;
using std::unordered_map;
using std::vector;
static std::map<std::vector<std::string>, std::vector<std::string>> params_share_map;

class PARSER_FUNC_VISIBILITY CaffeModelParser : public domi::ModelParser {
 public:
  CaffeModelParser() {}
  virtual ~CaffeModelParser() {}

  /**
   * @ingroup domi_omg
   * @brief Parse the relevant data from the model file and save it to graph
   * @param [in] file Path of model file
   * @param [in|out] graph graph for saving model information
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
   * @param [in] model_file  Path of model file
   * @param [out] json_file Converted JSON file path
   * @return SUCCESS parse successfully
   * @return others parse failed
   */
  Status ToJson(const char *model_file, const char *json_file) override;
  /**
   * @ingroup domi_omg
   * @brief Parse the relevant data from the model file and save it to graph
   * @param [in] graph_def input tensorflow model
   * @param [in|out] graph graph for saving model information
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) override;
  Status ParseProtoWithSubgraph(const google::protobuf::Message *root_proto, domi::GetGraphCallback callback,
                                ge::ComputeGraphPtr &graph) override;
  /*
   * @ingroup domi_omg
   * @brief Mapping CAFFE's datatype to GE's datatype
   * @param [in] type, datatype types of operators in CAFFE networks
   * @return ge::DataType
   */
  ge::DataType ConvertToGeDataType(const uint32_t type) override { return ge::DT_FLOAT; }

  Status ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) override {
    return domi::SUCCESS;
  }

 private:
  Status Parse(const char *file, ge::ComputeGraphPtr &graph);

  /**
   * @ingroup domi_omg
   * @brief Add the Layer in the model to the PreChecker
   * @param [in] net caffe net information
   * @return SUCCESS build successfully
   * @return FAILED build failed
   */
  Status PreCheck(const domi::caffe::NetParameter &net);

  /**
   * @ingroup domi_omg
   * @brief Parsing input related information from model files
   * @param [in] proto_message caffe net information
   * @param [in|out] net_input_name Used to store the acquired input name information
   * @param [in|out] net_input_data Used to store the acquired input data information
   * @return SUCCESS build successfully
   * @return FAILED build failed
   */
  Status ParseInput(domi::caffe::NetParameter &proto_message, bool &input_data_flag);

  /*
   * @ingroup domi_omg
   * @brief Parse model by custom proto and save info to operators
   * @param [in] model_path, file path of model(prototxt file)
   * @param [in] custom_proto, file path of custom proto
   * @param [in] caffe_proto, file path of caffe proto
   * @param [out] operators, operators saving custom info
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status CustomProtoParse(const char *model_path, const string &custom_proto, const string &caffe_proto,
                          std::vector<ge::Operator> &operators);

  /*
   * @ingroup domi_omg
   * @brief Parse model by custom proto and save info to operators
   * @param [in] model_path, file path of model(prototxt file)
   * @param [in] custom_proto_path, file path of custom proto
   * @param [in] custom_proto_name, custom proto name
   * @param [out] operators, operators saving custom info
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  Status ParseNetModelByCustomProto(const char *model_path, const string &custom_proto_path,
                                    const string &custom_proto_name, std::vector<ge::Operator> &operators);

  /*
   * @ingroup domi_omg
   * @brief Read caffe model and shield google warning
   * @param [in] model_path, file path of model(prototxt file)
   * @param [out] message, message saving custom info
   * @return SUCCESS read file successfully
   * @return FAILED read file failed
   */
  Status ReadModelWithoutWarning(const char *model_path, google::protobuf::Message *message);

  /*
   * @ingroup domi_omg
   * @brief Read caffe model and save it to message
   * @param [in] model_path, file path of model(prototxt file)
   * @param [out] message, message saving custom info
   * @return SUCCESS read file successfully
   * @return FAILED read file failed
   */
  Status ReadCaffeModelFromText(const char *model_path, google::protobuf::Message *message);

  /*
   * @ingroup domi_omg
   * @brief Parse layer message and save custom info to operators
   * @param [in] layer_descriptor, layer description of message
   * @param [in] message, message of model
   * @param [out] operators, operators saving custom info
   * @return SUCCESS parse layer successfully
   * @return FAILED parse layer failed
   */
  Status ParseLayerParameter(const google::protobuf::Descriptor *layer_descriptor,
                             const google::protobuf::Message *message, std::vector<ge::Operator> &operators);

  /*
   * @ingroup domi_omg
   * @brief Create custom operator by op_name and op_type
   * @param [in] op_name, name of operator
   * @param [in] op_type, type of operator
   * @param [in] message, message of model
   * @param [in] index, index of field
   * @param [out] operators, operators saving custom info
   * @return SUCCESS create operator successfully
   * @return FAILED create operator failed
   */
  Status CreateCustomOperator(std::string op_name, std::string op_type, const google::protobuf::Message *message,
                              int index, std::vector<ge::Operator> &operators);
  /**
   * @ingroup domi_omg
   * @brief Add blob information to the bottom_blobs_map and top_blobs_map_
   * @param [in] layer layer information
   * @param [in|out] inplace_blob_name_remapping save blob information
   * @return Status
   */
  Status AddBlobsToMap(const domi::caffe::LayerParameter &layer,
                       std::map<std::string, std::string> &inplace_blob_name_remapping);
  /**
   * @ingroup domi_omg
   * @brief Add node information to graph
   * @param [in] layer layer infromation
   * @param [in|out] graph graph for saving model information
   * @return SUCCESS add successfully
   * @return FAILED add failed
   */
  Status AddNode(const domi::caffe::LayerParameter &layer, ge::ComputeGraphPtr &graph);
  /**
   * @ingroup domi_omg
   * @brief Add edge information to graph
   * @param [in|out] graph graph for saving model information
   * @return SUCCESS add successfully
   * @return FAILED add failed
   */
  Status AddEdges(ge::ComputeGraphPtr &graph);

  /**
   * @ingroup domi_omg
   * @brief Add top name information to graph
   * @param [in|out] proto_message
   * @return SUCCESS add successfully
   * @return FAILED add failed
   */
  Status AddOutputTop(const domi::caffe::NetParameter &proto_message);

  /**
   * @ingroup domi_omg
   * @brief Check if the current layer is valid
   * @return true valid
   * @return false invalid
   */
  bool CheckValidLayer(const domi::caffe::LayerParameter &layer);

  /**
   * @ingroup domi_omg
   * @brief Check whether the top of the current layer is 'Inplace'
   * @return true is 'Inplace'
   * @return false not is 'Inplace'
   */
  bool IsInplaceTopBlob(const domi::caffe::LayerParameter &layer, const std::string &top_name);

  /**
   * @ingroup domi_omg
   * @brief Check whether the top of the current layer is user's specified output top
   * @return true yes
   * @return false no
   */
  bool IsOutputTop(const string &op_name, int32_t index);

  /**
   * @ingroup domi_omg
   * @brief Find a layer set with the same param
   * @param [in] Param name set of each layer
   * @param [in|out] Layer set of the same param
   * @return Status
   */
  Status FindShareParamLayers(const std::map<std::string, std::vector<std::string>> &);

  Status AddTensorDescToOpDesc(ge::OpDescPtr &op_desc, const domi::caffe::LayerParameter &layer);

  Status AddTensorDescToOpDescByIr(ge::OpDescPtr &op_desc, const domi::caffe::LayerParameter &layer,
                                   const string &op_type);

  Status AddUserOutNodesTop();

  std::string RemapTopNameByLayer(const domi::caffe::LayerParameter &layer, const std::string &top_name, int index);

  Status GetCustomOp(const domi::caffe::LayerParameter &layer, vector<ge::Operator> &operators);

  bool IsOpAttrEmpty(const ge::Operator &op, const std::string &type);

  Status ParseOpParam(const domi::caffe::LayerParameter &layer, ge::OpDescPtr &op,
                      std::shared_ptr<ge::OpParser> &op_parser);

  void SaveOrigionLayerTops(domi::caffe::LayerParameter &layer);

  Status ReorderInput(domi::caffe::NetParameter &net);

  void AddOutputInfoToContext(string layer_name, int32_t top_index);

  Status ParseOutputNodeTopInfo(const domi::caffe::NetParameter &proto_message);

  Status SaveDataLayerTops(const domi::caffe::LayerParameter &layer);

  std::map<std::string, ge::NodePtr> node_map;

  // key: blob name, value: layer name and index
  std::map<std::string, std::vector<std::pair<std::string, int32_t>>> bottom_blobs_map_;

  // key: blob name, value: layer name and index
  std::map<std::string, std::vector<std::pair<std::string, int32_t>>> top_blobs_map_;

  std::vector<ge::Operator> custom_operator_;
  std::map<std::string, std::vector<std::string>> layer_tops_map_;
};

/**
 * @ingroup domi_omg
 * @brief Caffe weight parser
 */
class PARSER_FUNC_VISIBILITY CaffeWeightsParser : public domi::WeightsParser {
 public:
  /**
   * @ingroup domi_omg
   * @brief Parse weight data from file and save to graph
   * @param [in] file             Path of weight file after training
   * @param [in|out]              graph Save weight information after parsing
   * @return SUCCESS              parse successfully
   * @return PARAM_INVALID        param invalid
   * @return PARSE_WEIGHTS_FAILED parse failed
   */
  Status Parse(const char *file, ge::Graph &graph) override;

  Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) override;

 private:
  Status CheckNodes(ge::ComputeGraphPtr &graph);
  /**
   * @ingroup domi_omg
   * @brief Convert netparameter to modedef and save in graph
   * @param [in] param Caffe network parameters to be converted
   * @param [in|out] graph Save weight information after parsing
   * @return SUCCESS parse successfully
   * @return FAILED parse failed
   */
  static Status ConvertNetParameter(const NetParameter &param, ge::ComputeGraphPtr &graph);

  Status Parse(const char *file, ge::ComputeGraphPtr &graph);

  Status ParseWeightByFusionProto(const char *model_path, const string &custom_proto_path,
                                  const string &custom_proto_name, ge::ComputeGraphPtr &graph);

  Status ParseLayerParameter(const google::protobuf::Descriptor *layer_descriptor,
                             const google::protobuf::Message *message,
                             ge::ComputeGraphPtr &graph);

  Status ConvertLayerParameter(const google::protobuf::Message *layer_message,
                               ge::ComputeGraphPtr &graph);

  Status CheckLayersSize(const google::protobuf::Message *message);

  Status ConvertLayerProto(const google::protobuf::Message *message,
                           google::protobuf::Message *layer);

  Status ParseLayerField(const google::protobuf::Reflection *reflection,
                         const google::protobuf::Message *message,
                         const google::protobuf::FieldDescriptor *field,
                         google::protobuf::Message *layer);

  Status ConvertBlobsProto(const google::protobuf::Message *message,
                           google::protobuf::Message *blobs);

  Status ConvertBlobShapeProto(const google::protobuf::Message *message,
                               google::protobuf::Message *dest_message);

  Status ConvertInnerProdcutProto(const google::protobuf::Message *message,
                                  google::protobuf::Message *dest_message);

  Status ConvertConvParamProto(const google::protobuf::Message *message,
                               google::protobuf::Message *dest_message);
  /**
   * @ingroup domi_omg
   * @brief Layer types to be ignored in weight resolution
   */
  static const set<string> skiped_layer_type_;
  std::map<std::string, int32_t> layer_name_record_map_;
};
}  // namespace domi

#endif  // PARSER_CAFFE_CAFFE_PARSER_H_
