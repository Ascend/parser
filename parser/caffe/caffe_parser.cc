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

#include "parser/caffe/caffe_parser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <algorithm>
#include "common/convert/message2operator.h"
#include "parser/common/convert/pb2json.h"
#include "parser/common/acl_graph_parser_util.h"
#include "common/op_map.h"
#include "common/util/error_manager/error_manager.h"
#include "common/string_util.h"
#include "external/graph/operator_factory.h"
#include "external/parser/caffe_parser.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include <google/protobuf/compiler/importer.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "omg/parser/op_parser.h"
#include "omg/parser/parser_factory.h"
#include "omg/parser/parser_inner_ctx.h"
#include "parser/caffe/caffe_custom_parser_adapter.h"
#include "parser/caffe/caffe_op_parser.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/pre_checker.h"
#include "parser/common/prototype_pass_manager.h"
#include "framework/omg/parser/parser_types.h"
#include "parser/common/model_saver.h"
#include "parser/common/acl_graph_parser_util.h"
#include "parser/common/proto_file_parser.h"
#include "register/op_registry.h"
#include "register/register_fmk_types.h"

using domi::caffe::LayerParameter;
using domi::caffe::NetParameter;
using domi::ParseParamByOpFunc;
using ge::caffe_op_map;
using ge::CaffeOpParser;
using ge::parser::ModelSaver;
using ge::OpParser;
using ge::OpParserFactory;
using ge::Pb2Json;
using ge::PreChecker;
using std::ifstream;

#define CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(val, errormsg)                                     \
  do {                                                                                          \
    if (val == nullptr) {                                                                       \
      GELOGE(ge::PARAM_INVALID, errormsg);                                                      \
      REPORT_INNER_ERROR("E19999", errormsg);                                                   \
      return ge::PARAM_INVALID;                                                                 \
    }                                                                                           \
  } while (0)

namespace ge {
graphStatus aclgrphParseCaffe(const char *model_file, const char *weights_file, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(model_file);
  GetParserContext().type = domi::CAFFE;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::CAFFE)));

  // load custom plugin so and proto
  AclGrphParseUtil acl_graph_parse_util;
  domi::Status status = acl_graph_parse_util.AclParserInitialize(options);
  if (status != domi::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AclParserInitialize failed, ret:%d.", status);
    GELOGE(GRAPH_FAILED, "[Parser][Initialize] failed, ret:%d.", status);
    return GRAPH_FAILED;
  }

  // Create an empty computegraph
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>("tmpGraph");
  GE_CHECK_NOTNULL(compute_graph);

  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::CAFFE);
  GE_CHECK_NOTNULL(model_parser);

  // parse caffe model_file and weights_file to GE graph
  ge::graphStatus ret = model_parser->Parse(model_file, graph);
  if (ret != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "parse param:model_file %s failed, graph:%s.", model_file, graph.GetName().c_str());
    GELOGE(ret, "[Parser][Param]ModelFile %s failed, graph:%s.", model_file, graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("Parser graph %s success.", graph.GetName().c_str());

  auto weights_parser = domi::WeightsParserFactory::Instance()->CreateWeightsParser(domi::CAFFE);
  GE_CHECK_NOTNULL(weights_parser);
  ret = weights_parser->Parse(weights_file, graph);
  if (ret != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "parse param:weights_file %s failed, graph:%s", weights_file, graph.GetName().c_str());
    GELOGE(ret, "[Parse][Param]WeightsFile %s failed. graph: %s", weights_file, graph.GetName().c_str());
    return ret;
  }
  GELOGI("Weights parse success. graph: %s", graph.GetName().c_str());
  std::map<AscendString, AscendString> parser_params;
  if (acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SetOutputNodeInfo failed, model file:%s graph:%s",
                      model_file, graph.GetName().c_str());
    GELOGE(ret, "[Invoke][SetOutputNodeInfo]Set graph %s default output node failed, model file:%s.",
           graph.GetName().c_str(), model_file);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

graphStatus aclgrphParseCaffe(const char *model_file, const char *weights_file,
                              const std::map<AscendString, AscendString> &parser_params, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(model_file);
  GetParserContext().type = domi::CAFFE;
  std::map<string, string> options;
  options.insert(std::pair<string, string>(string(ge::FRAMEWORK_TYPE), to_string(domi::CAFFE)));

  // load custom plugin so and proto
  AclGrphParseUtil acl_graph_parse_util;
  domi::Status status = acl_graph_parse_util.AclParserInitialize(options);
  if (status != domi::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AclParserInitialize failed, ret:%d.", status);
    GELOGE(GRAPH_FAILED, "[Parser][Initialize] failed ret:%d.", status);
    return GRAPH_FAILED;
  }

  string output_name;
  if (acl_graph_parse_util.ParseParamsBeforeGraph(parser_params, output_name) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Parse][Params]Before Graph failed.");
    return ge::FAILED;
  }
  // Create an empty computegraph
  string graph_name = output_name.empty() ? "tmpGraph" : output_name;
  ge::ComputeGraphPtr compute_graph = ge::parser::MakeShared<ge::ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(compute_graph);

  graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  auto model_parser = domi::ModelParserFactory::Instance()->CreateModelParser(domi::CAFFE);
  GE_CHECK_NOTNULL(model_parser);

  // parse caffe model_file and weights_file to GE graph
  ge::graphStatus ret = model_parser->Parse(model_file, graph);
  if (ret != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Parse param:model_file %s failed, graph:%s", model_file, graph.GetName().c_str());
    GELOGE(ret, "[Parser][Param]ModelFile %s failed, graph %s.", model_file, graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("Parser graph %s success.", graph.GetName().c_str());

  if (acl_graph_parse_util.ParseParamsAfterGraph(graph, parser_params) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "ParseParamsAfterGraph failed, graph:%s.", graph.GetName().c_str());
    GELOGE(ge::FAILED, "[Parser][Params] after graph failed, graph:%s.", graph.GetName().c_str());
    return ge::FAILED;
  }

  auto weights_parser = domi::WeightsParserFactory::Instance()->CreateWeightsParser(domi::CAFFE);
  GE_CHECK_NOTNULL(weights_parser);
  ret = weights_parser->Parse(weights_file, graph);
  if (ret != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "parse param:weights_file %s failed, graph: %s", weights_file, graph.GetName().c_str());
    GELOGE(ret, "[Parse][Param]WeightsFile %s failed. graph: %s", weights_file, graph.GetName().c_str());
    return ret;
  }
  GELOGI("Weights parse success. graph: %s", graph.GetName().c_str());

  if (acl_graph_parse_util.SetOutputNodeInfo(graph, parser_params) != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SetOutputNodeInfo failed, graph:%s", graph.GetName().c_str());
    GELOGE(ge::FAILED, "[Invoke][SetOutputNodeInfo]Set graph %s default output node failed.",
           graph.GetName().c_str());
    return ge::FAILED;
  }
  GELOGI("AclgrphParse graph %s success.", graph.GetName().c_str());
  return ge::SUCCESS;
}
} // namespace ge


namespace ge {
namespace {
const int32_t kAnchorIndexOne = 1;
const int32_t kAnchorIndexTwo = 2;
const int32_t kAnchorIndexThree = 3;
const int32_t kNumOne = 1;
const size_t kTensorNum = 2;
const int32_t kMinLineWorldSize = 3;
const int32_t kMaxIdentifier = 536870911; // 2^29 - 1
const int32_t kBase = 10;
const char *const kPython = "Python";
const char *const kProposalLayer = "ProposalLayer";
const char *const kDetectionOutput = "DetectionOutput";
const char *const kProjectRoot = "project_root";
const char *const kBeginningMessageType = "domi.caffe.NetParameter";
const char *const kLayerMessageType = "domi.caffe.LayerParameter";
const char *const kLayerName = "layer";
const char *const kLayersName = "layers";
const char *const kFieldName = "name";
const char *const kFieldType = "type";
const char *const kFieldBottom = "bottom";
const char *const kFieldTop = "top";
const char *const kFieldBlobs = "blobs";
const char *const kFieldShape = "shape";
const char *const kFieldConvParam = "convolution_param";
const char *const kFieldInnerPro = "inner_product_param";
const char *const kFieldDim = "dim";
const char *const kFieldBiasTerm = "bias_term";
const char *const kDevNull = "/dev/null";
const std::string kMessage = "message";
const std::string kLayerParameter = "LayerParameter";
const std::string kCloseBrace = "}";
const std::string kOptional = "optional";
const std::string kRepeated = "repeated";
const std::string kRequired = "required";
const std::string kCustom = "custom";
const std::string kBuiltin = "built-in";
std::vector<std::string> kAddTensorIrSkipNodes = {ge::parser::DATA, ge::parser::YOLODETECTIONOUTPUT,
                                                  ge::parser::NETOUTPUT};
const std::set<std::string> kCustomProtoLayerCommonField = {"name", "type"};
const std::set<std::string> kCaffeProtoLayerCommonField = {"name", "type", "bottom", "top", "phase",
                                                           "loss_weight", "param", "blobs", "propagate_down",
                                                           "include", "exclude"};
Status CheckPathValid(const char *model_path, const string &custom_proto, string &custom_proto_path,
                      string &custom_proto_name) {
  string path_model = ge::parser::RealPath(model_path);
  if (path_model.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19000", {"path", "errmsg"}, {model_path, strerror(errno)});
    GELOGE(FAILED, "[Check][Param]ModelPath %s is Invalid path of model", model_path);
    return FAILED;
  }

  custom_proto_name = kProjectRoot;
  auto pos = custom_proto.find_last_of("/\\");
  if (pos == string::npos) {
    custom_proto_path = "./";
    custom_proto_name += '/' + custom_proto;
  } else {
    custom_proto_path = custom_proto.substr(0, pos);
    custom_proto_name += '/' + custom_proto.substr(pos + 1);
  }
  GELOGI("Check validity of model file: %s and proto file: %s success.", model_path, custom_proto.c_str());

  return SUCCESS;
}
}  // namespace
   /*
      MultiLabelLMDB?The negligible layer for weight analysis in license plate recognition network of Safe city.
      Python: Currently, python custom layer only supports proposal,
      and there is no corresponding data in the proposal weight file, so Python layer is ignored.
   */
const set<string> CaffeWeightsParser::skiped_layer_type_ = {"Split",   "SoftmaxWithLoss", "Accuracy", "Data",
                                                            "Dropout", "MultiLabelLMDB",  "Python",   "AnnotatedData"};

Status CaffeModelParser::ParseInput(domi::caffe::NetParameter &proto_message, bool &input_data_flag) {
  if (proto_message.input_size() > 0) {
    GELOGI("This net exsit input.");

    if (proto_message.input_dim_size() > 0) {
      if (proto_message.input_shape_size() > 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E11001");
        GELOGE(FAILED, "[Check][Size]input_dim and input_shape can not both exist!");
        return FAILED;
      }
      int input_dim_size = proto_message.input_dim_size();

      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((input_dim_size / proto_message.input_size() != parser::DIM_DEFAULT_SIZE ||
                                      input_dim_size % proto_message.input_size() != 0),
                                     ErrorManager::GetInstance().ATCReportErrMessage(
                                         "E11003", {"input_dim_size", "input_size"},
                                         {std::to_string(input_dim_size), std::to_string(proto_message.input_size())});
                                     return FAILED,
                                     "[Check][Size]Model input_dim size[%d] is not 4 times of input size[%d].",
                                     input_dim_size, proto_message.input_size())

      for (int i = 0; i < proto_message.input_size(); i++) {
        domi::caffe::LayerParameter *layer = proto_message.add_layer();
        GE_CHECK_NOTNULL(layer);
        layer->set_name(proto_message.input(i));
        layer->set_type(ge::parser::INPUT_TYPE);
        layer->add_top(proto_message.input(i));

        domi::caffe::InputParameter *input_param = layer->mutable_input_param();
        GE_CHECK_NOTNULL(input_param);
        domi::caffe::BlobShape *shape = input_param->add_shape();
        GE_CHECK_NOTNULL(shape);

        for (int j = 0; j < parser::DIM_DEFAULT_SIZE; j++) {
          // Can guarantee that it will not cross the border
          shape->add_dim(static_cast<int64_t>(proto_message.input_dim(j + i * parser::DIM_DEFAULT_SIZE)));
        }
        input_data_flag = true;
      }
    } else if (proto_message.input_shape_size() > 0) {
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(proto_message.input_shape_size() != proto_message.input_size(),
          ErrorManager::GetInstance().ATCReportErrMessage("E11004", {"input_shape_size", "input_size"},
                                                          {std::to_string(proto_message.input_shape_size()),
                                                           std::to_string(proto_message.input_size())});
          return FAILED, "[Check][Size]caffe net input_shape size(%d) is not equal input size(%d).",
          proto_message.input_shape_size(), proto_message.input_size());

      for (int i = 0; i < proto_message.input_size(); i++) {
        int dim_size = proto_message.input_shape(i).dim_size();

        domi::caffe::LayerParameter *layer = proto_message.add_layer();
        GE_CHECK_NOTNULL(layer);
        layer->set_name(proto_message.input(i));
        layer->set_type(ge::parser::INPUT_TYPE);
        layer->add_top(proto_message.input(i));

        domi::caffe::InputParameter *input_param = layer->mutable_input_param();
        GE_CHECK_NOTNULL(input_param);
        domi::caffe::BlobShape *shape = input_param->add_shape();
        GE_CHECK_NOTNULL(shape);

        for (int j = 0; j < dim_size; j++) {
          // Can guarantee that it will not cross the border
          shape->add_dim(static_cast<int64_t>(proto_message.input_shape(i).dim(j)));
        }
        input_data_flag = true;
      }
    } else {
      const ge::ParserContext &ctx = ge::GetParserContext();
      std::map<std::string, std::vector<int64_t>> input_dims = ctx.input_dims;
      for (int i = 0; i < proto_message.input_size(); i++) {
        string name = proto_message.input(i);
        if (input_dims.count(name) == 0) {  // Input defined by model does not exist in input of external input
          REPORT_INPUT_ERROR("E11005", std::vector<std::string>({"input"}), std::vector<std::string>({name}));
          GELOGE(FAILED, "[Find][Dim]Model has no input shape.");
          return FAILED;
        }
        std::vector<int64_t> dims = input_dims.at(name);
        size_t dim_size = dims.size();

        domi::caffe::LayerParameter *layer = proto_message.add_layer();
        GE_CHECK_NOTNULL(layer);
        layer->set_name(name);
        layer->set_type(ge::parser::INPUT_TYPE);
        layer->add_top(proto_message.input(i));

        domi::caffe::InputParameter *input_param = layer->mutable_input_param();
        GE_CHECK_NOTNULL(input_param);
        domi::caffe::BlobShape *shape = input_param->add_shape();
        GE_CHECK_NOTNULL(shape);

        for (size_t j = 0; j < dim_size; j++) {
          shape->add_dim(dims.at(j));
        }
        input_data_flag = true;
      }
    }
  }
  return SUCCESS;
}


Status CaffeModelParser::ParseNetModelByCustomProto(const char *model_path, const string &custom_proto_path,
                                                    const string &custom_proto_name, vector<ge::Operator> &operators) {
  google::protobuf::compiler::DiskSourceTree source_tree;
  source_tree.MapPath(kProjectRoot, custom_proto_path);
  google::protobuf::compiler::Importer importer(&source_tree, nullptr);
  importer.Import(custom_proto_name.c_str());
  GELOGI("Import custom proto %s success.", custom_proto_path.c_str());

  const google::protobuf::Descriptor *descriptor = importer.pool()->FindMessageTypeByName(kBeginningMessageType);
  GE_CHECK_NOTNULL(descriptor);
  google::protobuf::DynamicMessageFactory factory;
  const google::protobuf::Message *proto = factory.GetPrototype(descriptor);
  GE_CHECK_NOTNULL(proto);
  google::protobuf::Message *message = proto->New();
  GE_CHECK_NOTNULL(message);

  if (ReadModelWithoutWarning(model_path, message) != SUCCESS) {
    delete message;
    GELOGE(FAILED, "[Invoke][ReadModelWithoutWarning] %s failed.", model_path);
    return FAILED;
  }

  GELOGI("Start to parse model file: %s.", model_path);
  const google::protobuf::Descriptor *layer_descriptor = importer.pool()->FindMessageTypeByName(kLayerMessageType);
  if (layer_descriptor == nullptr) {
    delete message;
    REPORT_INPUT_ERROR("E11032", std::vector<std::string>({"message_type", "name", "reason"}),
                       std::vector<std::string>({"model", "LayerParameter",
                           "Does not find domi.caffe.LayerParameter in google::protobuf::Descriptor"}));
    GELOGE(FAILED, "[Invoke][FindMessageTypeByName]Does not find domi.caffe.LayerParameter"
           "in google::protobuf::Descriptor");
    return FAILED;
  }

  if (ParseLayerParameter(layer_descriptor, message, operators) != SUCCESS) {
    delete message;
    GELOGE(FAILED, "[Parse][LayerParameter] failed, model path:%s.", model_path);
    return FAILED;
  }

  delete message;
  GELOGI("Parse model: %s by proto: %s success.", model_path, custom_proto_path.c_str());
  return SUCCESS;
}

Status CaffeModelParser::CustomProtoParse(const char *model_path, const string &custom_proto,
                                          const string &caffe_proto, vector<ge::Operator> &operators) {
  string custom_proto_path = ge::parser::RealPath(custom_proto.c_str());
  if (custom_proto_path.empty()) {
    GELOGW("Valid custom proto: %s does not exist, skip parsing custom proto", custom_proto.c_str());
    return SUCCESS;
  }

  string custom_proto_name;
  if (CheckPathValid(model_path, custom_proto, custom_proto_path, custom_proto_name) != SUCCESS) {
    GELOGE(FAILED, "[Check][PathValid] of model and proto failed, path:%s.", model_path);
    return FAILED;
  }

  GELOGI("Start to parse model: %s by custom proto: %s.", model_path, custom_proto.c_str());
  Status ret = ParseNetModelByCustomProto(model_path, custom_proto_path, custom_proto_name, operators);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Parse][NetModel]By CustomProto failed, path:%s.", model_path);
  }

  return ret;
}

Status CaffeModelParser::ReadModelWithoutWarning(const char *model_path, google::protobuf::Message *message) {
  int32_t copy_fd = mmDup(STDERR_FILENO);
  if (copy_fd < 0) {
    REPORT_CALL_ERROR("E19999", "Duplicate to file STDERR_FILENO failed, errmsg:%s", strerror(errno));
    GELOGE(FAILED, "[Invoke][Dup] failed:%d, reason:%s", copy_fd, strerror(errno));
    return FAILED;
  }

  int32_t fd = mmOpen(kDevNull, M_RDWR);
  if (fd < 0) {
    (void)mmClose(copy_fd);
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {kDevNull, strerror(errno)});
    GELOGE(FAILED, "[Open][File] %s failed. reason:%s", kDevNull, strerror(errno));
    return FAILED;
  }

  if (mmDup2(fd, STDERR_FILENO) < 0) {
    (void)mmClose(fd);
    (void)mmClose(copy_fd);
    REPORT_CALL_ERROR("E19999", "Duplicate to file STDERR_FILENO failed, errmsg:%s", strerror(errno));
    GELOGE(FAILED, "[Invoke][Dup2] Re-orient failed. reason:%s", strerror(errno));
    return FAILED;
  }

  if (ReadCaffeModelFromText(model_path, message) != SUCCESS) {
    (void)mmClose(fd);
    (void)mmClose(copy_fd);
    GELOGE(FAILED, "[Read][CaffeModel] From Text %s failed.", model_path);
    return FAILED;
  }

  if (mmDup2(copy_fd, STDERR_FILENO) < 0) {
    (void)mmClose(fd);
    (void)mmClose(copy_fd);
    REPORT_CALL_ERROR("E19999", "Duplicate to file STDERR_FILENO failed, errmsg:%s", strerror(errno));
    GELOGE(FAILED, "[Invoke][Dup2] Re-orient failed. reason:%s", strerror(errno));
    return FAILED;
  }
  (void)mmClose(fd);
  (void)mmClose(copy_fd);

  return SUCCESS;
}

Status CaffeModelParser::ReadCaffeModelFromText(const char *model_path, google::protobuf::Message *message) {
  GE_CHECK_NOTNULL(model_path);
  GE_CHECK_NOTNULL(message);
  GELOGI("Start to read model file: %s.", model_path);
  std::ifstream fs(model_path, std::ifstream::in);
  if (!fs.is_open()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {model_path, "ifstream open failed"});
    GELOGE(FAILED, "[Open][File] %s failed.", model_path);
    return FAILED;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  google::protobuf::TextFormat::Parser model_parser;
  model_parser.AllowUnknownField(true);
  if (!model_parser.Parse(&input, message)) {
    fs.close();
    ErrorManager::GetInstance().ATCReportErrMessage("E19005", {"file"}, {model_path});
    GELOGE(FAILED, "[Parse][ModelFile] %s failed.", model_path);
    return FAILED;
  }
  fs.close();
  GELOGI("Read model file: %s success.", model_path);

  return SUCCESS;
}

Status CaffeModelParser::ParseLayerParameter(const google::protobuf::Descriptor *layer_descriptor,
                                             const google::protobuf::Message *message,
                                             vector<ge::Operator> &operators) {
  auto field_name = layer_descriptor->FindFieldByName(kFieldName);
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field_name, "Does not find name in google::protobuf::Descriptor");
  auto field_type = layer_descriptor->FindFieldByName(kFieldType);
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field_type, "Does not find type in google::protobuf::Descriptor");

  const google::protobuf::Reflection *reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);
  for (auto &field : field_desc) {
    CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field, "Get FieldDescriptor failed in google::protobuf::Message");
    // Only care about layers
    if (field->name() != kLayerName) {
      continue;
    }
    if (!field->is_repeated()) {
      REPORT_INPUT_ERROR("E11032", std::vector<std::string>({"message_type", "name", "reason"}),
                         std::vector<std::string>({"model", field->name(), "LayerParameter should be repeated"}));
      GELOGE(FAILED, "[Check][Param] LayerParameter should be repeated.");
      return FAILED;
    }

    int field_size = reflection->FieldSize(*message, field);
    GELOGI("Total Layer num of model file is %d", field_size);
    for (int i = 0; i < field_size; ++i) {
      const google::protobuf::Message &layer_message = reflection->GetRepeatedMessage(*message, field, i);
      const google::protobuf::Reflection *layer_reflection = layer_message.GetReflection();
      CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(layer_reflection, "Get Reflection failed in google::protobuf::Message");
      GE_CHECK_NOTNULL(layer_reflection);

      string op_name = layer_reflection->GetString(layer_message, field_name);
      string op_type = layer_reflection->GetString(layer_message, field_type);
      if (domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(op_type) == nullptr) {
        continue;
      }
      if (CreateCustomOperator(op_name, op_type, &layer_message, i, operators) != SUCCESS) {
        GELOGE(FAILED, "[Create][CustomOperator] failed, name: %s, type: %s.", op_name.c_str(), op_type.c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CaffeModelParser::CreateCustomOperator(string op_name, string op_type, const google::protobuf::Message *message,
                                              int index, vector<ge::Operator> &operators) {
  if (op_name.empty() || op_type.empty()) {
    REPORT_INNER_ERROR("E19999", "[Check][Param]Name or type of layer is empty, name: %s, type: %s.",
                       op_name.c_str(), op_type.c_str());
    GELOGE(FAILED, "[Check][Param]Name or type of layer is empty, name: %s, type: %s.",
           op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  GELOGI("Start to create new operator, name: %s, type: %s, index: %d.", op_name.c_str(), op_type.c_str(), index);
  ge::Operator ops(op_name, op_type);
  if (ops.GetName() != op_name) {
    REPORT_INNER_ERROR("E19999", "Create Operator failed, name: %s, type: %s, index: %d.",
                       op_name.c_str(), op_type.c_str(), index);
    GELOGE(FAILED, "[Create][Operator] failed, name: %s, type: %s, index: %d.",
           op_name.c_str(), op_type.c_str(), index);
    return FAILED;
  }

  if (Message2Operator::ParseOperatorAttrs(message, 1, ops) != SUCCESS) {
    GELOGE(FAILED, "[Parse][OperatorAttrs] of %s failed.", op_name.c_str());
    return FAILED;
  }

  operators.emplace_back(ops);
  GELOGI("Create new operator success, name: %s, type: %s, index: %d.", op_name.c_str(), op_type.c_str(), index);

  return SUCCESS;
}

void CaffeModelParser::AddOutputInfoToContext(string layer_name, int32_t top_index) {
  auto iter_node_name = ge::GetParserContext().out_nodes_map.find(layer_name);
  if (iter_node_name != ge::GetParserContext().out_nodes_map.end()) {
    iter_node_name->second.emplace_back(top_index);
  } else {
    std::vector<int32_t> index_v;
    index_v.emplace_back(top_index);
    ge::GetParserContext().out_nodes_map.emplace(layer_name, index_v);
  }
  ge::GetParserContext().user_out_nodes.push_back(std::make_pair(layer_name, top_index));
}

Status CaffeModelParser::ParseOutputNodeTopInfo(const domi::caffe::NetParameter &proto_message) {
  if (ge::GetParserContext().user_out_nodes_top_vec.empty()) {
    return SUCCESS;
  }

  ge::GetParserContext().out_nodes_map.clear();
  ge::GetParserContext().user_out_nodes.clear();
  int32_t layer_count = proto_message.layer_size();
  const std::vector<string> &user_out_nodes_top_vec =
      ge::GetParserContext().user_out_nodes_top_vec;

  for (const auto &top_name : user_out_nodes_top_vec) {
    bool find_node_falg = false;
    string layer_name;
    int32_t top_index = 0;
    for (int32_t i = layer_count - 1; i >= 0; --i) {
      domi::caffe::LayerParameter &layer =
          const_cast<domi::caffe::LayerParameter &>(proto_message.layer(i));

      for (int j = 0; j < layer.top_size(); ++j) {
        string top_blob_name = layer.top(j);
        if (top_blob_name != top_name) {
          continue;
        }

        find_node_falg = true;
        layer_name.assign(layer.name());
        top_index = static_cast<int32_t>(j);
        break;
      }
      if (find_node_falg) {
        break;
      }
    }
    if (!find_node_falg || layer_name.empty()) {
        REPORT_INPUT_ERROR("E11017", std::vector<std::string>({"opname"}), std::vector<std::string>({top_name}));
        GELOGE(PARAM_INVALID, "[Check][Param]Cannot find top_name[%s], which is invalid", top_name.c_str());
        return PARAM_INVALID;
    }
    GELOGD("Node[%s] find top_name[%s], top_index[%d]", layer_name.c_str(), top_name.c_str(), top_index);
    AddOutputInfoToContext(layer_name, top_index);
  }
  return SUCCESS;
}

Status CaffeModelParser::AddBlobsToMap(const domi::caffe::LayerParameter &layer,
                                       std::map<std::string, std::string> &inplace_blob_name_remapping) {
  if (layer.top_size() <= 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E11037", {"opname"}, {layer.name()});
    GELOGE(FAILED, "[Check][Size]The output size of layer %s needs to be greater than zero.", layer.name().c_str());
    return FAILED;
  }

  // Need to check if the input is the output of 'inplace'
  for (int i = 0; i < layer.bottom_size(); ++i) {
    std::string blob_name = layer.bottom(i);
    auto iter = inplace_blob_name_remapping.find(blob_name);
    if (iter != inplace_blob_name_remapping.end()) {
      blob_name = iter->second;
    }
    bottom_blobs_map_[blob_name].emplace_back(std::make_pair(layer.name(), i));
  }

  // Handling 'inplace' scenarios
  for (int j = 0; j < layer.top_size(); ++j) {
    std::string top_blob_name = layer.top(j);
    if (IsInplaceTopBlob(layer, top_blob_name)) {
      std::string remapped_blob_name = RemapTopNameByLayer(layer, top_blob_name, j);
      inplace_blob_name_remapping[top_blob_name] = remapped_blob_name;
      top_blob_name = remapped_blob_name;
    }
    top_blobs_map_[top_blob_name].emplace_back(std::make_pair(layer.name(), j));
  }

  return SUCCESS;
}

bool CaffeModelParser::IsOpAttrEmpty(const ge::Operator &op, const std::string &type) {
  const std::map<std::string, std::string> attrs = op.GetAllAttrNamesAndTypes();

  if (type == kCustom) {
    for (const auto &attr : attrs) {
      if (kCustomProtoLayerCommonField.count(attr.first) == 0) {
        GELOGI("Custom op[%s] attr name[%s] exists, not empty.", op.GetName().c_str(), attr.first.c_str());
        return false;
      }
    }
  } else if (type == kBuiltin) {
    for (const auto &attr : attrs) {
      if (kCaffeProtoLayerCommonField.count(attr.first) == 0) {
        GELOGI("Built-in op[%s] attr name[%s] exists, not empty.", op.GetName().c_str(), attr.first.c_str());
        return false;
      }
    }
  }

  return true;
}

Status CaffeModelParser::GetCustomOp(const domi::caffe::LayerParameter &layer, vector<ge::Operator> &operators) {
  string op_type = layer.type();
  string op_name = layer.name();

  bool is_search_built_in_layer = false;
  for (ge::Operator &custom_op : custom_operator_) {
    if (custom_op.GetName() == layer.name() && custom_op.GetOpType() == op_type) {
      if (IsOpAttrEmpty(custom_op, kCustom)) {
        GELOGW("Custom op attr is empty, should try to get op params from built-in layer.");
        is_search_built_in_layer = true;
      } else {
        operators.emplace_back(custom_op);
        GELOGI("Find custom op success.");
        return SUCCESS;
      }
      break;
    }
  }

  if (custom_operator_.empty()) {
    GELOGW("Custom operator is empty, should try to get op params from built-in layer.");
    is_search_built_in_layer = true;
  }

  if (is_search_built_in_layer) {
    const google::protobuf::Message *layer_message = reinterpret_cast<const google::protobuf::Message *>(&layer);
    Status status = CreateCustomOperator(op_name, op_type, layer_message, 0, operators);
    if (status != SUCCESS || operators.empty()) {
      GELOGE(status, "[Create][CustomOperator] failed, name: %s, type: %s.", op_name.c_str(), op_type.c_str());
      return FAILED;
    }
    if (IsOpAttrEmpty(operators[0], kBuiltin)) {
      GELOGW("Custom and built-in op attr param is empty, name: %s, type: %s.", op_name.c_str(), op_type.c_str());
    }
    GELOGI("Search built-in layer success.");
  }
  return SUCCESS;
}

Status CaffeModelParser::ParseOpParam(const domi::caffe::LayerParameter &layer, ge::OpDescPtr &op,
                                      std::shared_ptr<OpParser> &op_parser) {
  GE_CHECK_NOTNULL(op);
  GE_CHECK_NOTNULL(op_parser);
  string op_type = layer.type();

  Status status = FAILED;
  ParseParamByOpFunc parse_param_func = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(op_type);
  if (parse_param_func == nullptr) {
    // Parsing weight information through opparser
    status = op_parser->ParseParams(&layer, op);
  } else {
    // The custom op defined by customer deals with parse params separately
    std::shared_ptr<ge::CaffeCustomParserAdapter> caffe_custom_op_parser =
            std::dynamic_pointer_cast<ge::CaffeCustomParserAdapter>(op_parser);
    vector<ge::Operator> custom_operator;
    status = GetCustomOp(layer, custom_operator);
    if (status != SUCCESS || custom_operator.empty()) {
      REPORT_CALL_ERROR("E19999", "Get CustomOp failed for op:%s(%s)", layer.name().c_str(), op_type.c_str());
      GELOGE(status, "[Get][CustomOp]failed for op [%s], optype [%s]",
             layer.name().c_str(), op_type.c_str());
      return status;
    }
    status = caffe_custom_op_parser->ParseParams(custom_operator[0], op);
  }

  if (status != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Parse param for op:%s(%s) failed", layer.name().c_str(), op_type.c_str());
    GELOGE(status, "[Parse][Params] for op [%s] fail, optype [%s]", layer.name().c_str(), op_type.c_str());
    return status;
  }

  return SUCCESS;
}

Status CaffeModelParser::AddNode(const domi::caffe::LayerParameter &layer, ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  // Release in node destructor
  string op_type;

  op_type = layer.type();
  // User defined duplicate name operator processing
  auto m_iter = ge::GetParserContext().op_conf_map.find(op_type);
  // User specified configuration item found
  if (m_iter != ge::GetParserContext().op_conf_map.end()) {
    op_type = m_iter->second;
  }
  // General layer layer, search optype
  auto iter = caffe_op_map.find(op_type);
  if (iter == caffe_op_map.end()) {
    if (op_type == kDetectionOutput) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11008");
      GELOGE(FAILED, "[Check][Type] Op type 'DetectionOutput' is confused. Suggest you modify the model file "
             "and use a explicit type, such as 'FSRDetectionOutput' or 'SSDDetectionOutput'.");
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage("E11009", {"opname", "optype"}, {layer.name(), op_type});
      GELOGE(FAILED, "[Check][Type]Unsupport op[%s] optype[%s], you should customize the op at first.",
             layer.name().c_str(), op_type.c_str());
    }

    return FAILED;
  }
  op_type = iter->second;

  GELOGD("Caffe layer name:%s, layer type %s", layer.name().c_str(), op_type.c_str());
  // create OpParser
  std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::CAFFE);
  GE_CHECK_NOTNULL(factory);
  std::shared_ptr<OpParser> op_parser = factory->CreateOpParser(op_type);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(op_parser == nullptr,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E11009", {"opname", "optype"},
                                                                                 {layer.name(), op_type});
                                 return FAILED, "op_parser is null, op_type: %s.",
                                 op_type.c_str());

  ge::OpDescPtr op;
  // Process change of tensordesc initialization of opdesc,
  // The previous process tensordesc was constructed according to the graph structure in the builder stage
  // The current process requires tensordesc to determine before the opdesc of the operator is added to the graph
  GE_RETURN_IF_ERROR(AddTensorDescToOpDescByIr(op, layer, op_type));
  GELOGI("After AddTensorDescToOpDescByIr op[%s] type[%s] have input size: %zu, output size: %zu",
         op->GetName().c_str(), op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());
  // op parser execute
  GE_RETURN_IF_ERROR(ParseOpParam(layer, op, op_parser));
  GELOGI("After op parser op[%s] type[%s] have input size: %zu, output size: %zu", op->GetName().c_str(),
         op->GetType().c_str(), op->GetInputsSize(), op->GetOutputsSize());

  // Caffe has also been plug-in at present. Here it is directly set to NCHW
  // Set input and output format
  GELOGI("Enter caffe parser. op name:%s, type:%s", op->GetName().c_str(), op->GetType().c_str());
  // inputDescsPtr and outputDescsPtr are guaranteed not to be nullptr
  auto inputDescsPtr = op->GetAllInputsDescPtr();
  auto outputDescsPtr = op->GetAllOutputsDescPtr();
  ge::Format format = ge::FORMAT_NCHW;

  for (auto &inputDescPtr : inputDescsPtr) {
    GE_CHECK_NOTNULL(inputDescPtr);
    inputDescPtr->SetFormat(format);
    inputDescPtr->SetOriginFormat(format);
  }
  for (auto &outputDescPtr : outputDescsPtr) {
    GE_CHECK_NOTNULL(outputDescPtr);
    outputDescPtr->SetFormat(format);
    outputDescPtr->SetOriginFormat(format);
  }

  ge::NodePtr node = graph->AddNode(op);
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "AddNode failed, op name:%s, type:%s", op->GetName().c_str(), op->GetType().c_str());
    GELOGE(FAILED, "[Add][Node] failed, op name:%s, type:%s", op->GetName().c_str(), op->GetType().c_str());
    return FAILED;
  }

  // Caffe's reshape is different from IR definition, which has only one input.
  // In caffe process, after constructing reshape according to IR, the second input of reshape is empty.
  // So a constant node needs to be added to reshape as the second input.
  // AddConstInput is a function defined in caffe_op_parser, override in caffe_reshape_parser.
  std::shared_ptr<CaffeOpParser> caffe_op_parser = std::static_pointer_cast<CaffeOpParser>(op_parser);
  GE_CHECK_NOTNULL(caffe_op_parser);
  Status status;
  status = caffe_op_parser->AddConstInput(node);
  if (status != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddConstInput failed for node:%s", node->GetOpDesc()->GetName().c_str());
    GELOGE(FAILED, "[Add][ConstInput] to node %s fail.", node->GetOpDesc()->GetName().c_str());
    return status;
  }

  node_map[layer.name()] = node;
  return SUCCESS;
}

Status CaffeModelParser::AddTensorDescToOpDesc(ge::OpDescPtr &op_desc, const domi::caffe::LayerParameter &layer) {
  GE_CHECK_NOTNULL(op_desc);
  // Data node input and output tensordesc added in parserparam
  if (op_desc->GetType() == ge::parser::DATA) {
    return SUCCESS;
  }

  for (int i = 0; i < layer.bottom_size(); i++) {
    ge::GeTensorDesc input_tensor;
    GE_RETURN_IF_ERROR(op_desc->AddInputDesc(input_tensor));
  }
  GELOGD("AddTensorInputDescToOpDesc, op name: %s, type: %s, input num: %d", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), layer.bottom_size());
  // Output number
  int32_t output_tensor_num = layer.top_size();
  GELOGD("AddTensorOutputDescToOpDesc, op name: %s, type: %s, output num: %d", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), output_tensor_num);
  for (int32_t j = 0; j < output_tensor_num; j++) {
    ge::GeTensorDesc output_tensor;
    GE_RETURN_IF_ERROR(op_desc->AddOutputDesc(output_tensor));
  }

  // yolo v2 YoloDetectionOutput
  if (op_desc->GetType() == ge::parser::YOLODETECTIONOUTPUT) {
    ge::GeTensorDesc input_tensor;
    GE_RETURN_IF_ERROR(op_desc->AddInputDesc(input_tensor));
    GE_RETURN_IF_ERROR(op_desc->AddInputDesc(input_tensor));
    GELOGD(
        "Current op type is YOLODETECTIONOUTPUT, add 2 additional inputs"
        "while it's original input num is: %d",
        layer.bottom_size());
  }
  return SUCCESS;
}

Status CaffeModelParser::AddTensorDescToOpDescByIr(ge::OpDescPtr &op_desc, const domi::caffe::LayerParameter &layer,
                                                   const string &op_type) {
  if (std::find(kAddTensorIrSkipNodes.begin(), kAddTensorIrSkipNodes.end(), op_type) != kAddTensorIrSkipNodes.end()) {
    op_desc = ge::parser::MakeShared<ge::OpDesc>(layer.name(), op_type);
    GE_CHECK_NOTNULL(op_desc);
    Status ret = AddTensorDescToOpDesc(op_desc, layer);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Add][TensorDesc]To OpDesc failed for op[%s] type[%s].", layer.name().c_str(), op_type.c_str());
    }
    return ret;
  }

  // Get opDesc by ir
  string layer_name = layer.name();
  ge::Operator op_factory = ge::OperatorFactory::CreateOperator(layer_name, op_type);
  if (op_factory.GetName() != layer.name()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10501", {"opname", "optype"}, {layer_name, op_type});
    GELOGE(FAILED, "[Invoke][CreateOperator]IR for op[%s] optype[%s] is not registered.",
           layer_name.c_str(), op_type.c_str());
    return FAILED;
  } else {
    op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_factory);
    GE_CHECK_NOTNULL(op_desc);
    auto valid_input_size = layer.bottom_size();
    auto blob_size = layer.blobs_size();
    GELOGI("After GetOpDescFromOperator op[%s] type[%s] have all input size: %zu, "
           "caffe_input_size:%d blob_size %d output size: %zu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           op_desc->GetAllInputsSize(), valid_input_size,
           blob_size, op_desc->GetOutputsSize());
    bool update_in_turn = (static_cast<int64_t >(op_desc->GetAllInputsSize()) == (valid_input_size + blob_size));
    for (int i = 0; i < valid_input_size; i++) {
      ge::GeTensorDesc input_tensor;
      std::string input_name;
      ge::graphStatus ret = ge::GRAPH_SUCCESS;
      // Below cases are supported fow now when there are optional inputs
      // x means optional, o means requierd input
      // a. ooxxx, number of o and x>=layer.bottom_size+layer.blobs_size>=number of o
      // b. oxoxoxox, layer.bottom_size+layer.blobs_size=number of o
      // c. oxoxoxox, layer.bottom_size+layer.blobs_size=number of o and x
      if (update_in_turn) {
        ret = op_desc->UpdateInputDesc(op_desc->GetInputNameByIndex(static_cast<uint32_t>(i)), input_tensor);
      } else {
        if (static_cast<size_t>(i) >= op_desc->GetInputsSize()) {
          ret = op_desc->UpdateInputDesc(static_cast<uint32_t>(i), input_tensor);
        } else {
          input_name = op_desc->GetValidInputNameByIndex(static_cast<uint32_t>(i));
          ret = op_desc->UpdateInputDesc(input_name, input_tensor);
        }
      }
      GELOGI("op [%s], type[%s], update input(%d) with name %s %s", op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), i, input_name.c_str(), ret == ge::GRAPH_SUCCESS ? "success" : "failed");
    }

    for (int i = 0; i < layer.top_size(); i++) {
      ge::GeTensorDesc output_tensor;
      auto ret = op_desc->UpdateOutputDesc(op_desc->GetOutputNameByIndex(static_cast<uint32_t>(i)), output_tensor);
      GELOGI("op [%s], type[%s], update output(%d) with name %s %s",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             i, op_desc->GetOutputNameByIndex(i).c_str(),
             ret == ge::GRAPH_SUCCESS ? "success" : "failed");
    }
  }
  return SUCCESS;
}

Status CaffeModelParser::AddEdges(ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  // Traversal input
  for (auto b_iter = bottom_blobs_map_.begin(); b_iter != bottom_blobs_map_.end(); b_iter++) {
    // Find the top blob corresponding to the bottom blob
    auto t_iter = top_blobs_map_.find(b_iter->first);
    // Unable to find the output corresponding to the input, error reported
    if (t_iter == top_blobs_map_.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11012", {"bottom_blob", "layer", "bottom_index"},
                                                      {b_iter->first, b_iter->second[0].first,
                                                       std::to_string(b_iter->second[0].second)});
      GELOGE(FAILED, "[Find][Blob]Unknown bottom blob '%s', in layer '%s', bottom index:%d.", b_iter->first.c_str(),
             b_iter->second[0].first.c_str(), b_iter->second[0].second);
      return PARAM_INVALID;
    }

    vector<pair<string, int32_t>> &top_blob_layers = t_iter->second;
    vector<pair<string, int32_t>> &bottom_blob_layers = b_iter->second;
    // 1.Traversal output, all input layers of the current blob
    for (auto &top_blob_layer_pair : top_blob_layers) {
      // 2.Traversal input, all output layers of the current blob
      for (auto &bottom_blob_layer_pair : bottom_blob_layers) {
        // Find the layer for this output
        auto top_node_iter = node_map.find(top_blob_layer_pair.first);
        // Find the layer for this input
        auto bottom_node_iter = node_map.find(bottom_blob_layer_pair.first);
        if (top_node_iter != node_map.end() && bottom_node_iter != node_map.end()) {
          // Output node top_node_iter->second,
          // Output index top_blob_layer_pair.second
          // input node bottom_node_iter->second
          // input index bottom_blob_layer_pair.second
          GELOGD("Start add edge: From %s:%d To %s:%d.", top_node_iter->second->GetName().c_str(),
                 top_blob_layer_pair.second, bottom_node_iter->second->GetName().c_str(),
                 bottom_blob_layer_pair.second);
          ge::OutDataAnchorPtr out_archor_ptr = top_node_iter->second->GetOutDataAnchor(top_blob_layer_pair.second);
          GE_CHECK_NOTNULL(out_archor_ptr);
          ge::InDataAnchorPtr in_archor_ptr = bottom_node_iter->second->GetInDataAnchor(bottom_blob_layer_pair.second);
          GE_CHECK_NOTNULL(in_archor_ptr);
          GE_IF_BOOL_EXEC(ge::GraphUtils::AddEdge(out_archor_ptr, in_archor_ptr) != ge::GRAPH_SUCCESS,
                          REPORT_CALL_ERROR("E19999", "Add edge between %s and %s failed",
                                            top_node_iter->second->GetName().c_str(),
                                            bottom_node_iter->second->GetName().c_str());
                          GELOGE(INTERNAL_ERROR, "[Invoke][AddEdge]Add link failed from op[%s] to op[%s].",
                                 top_node_iter->second->GetName().c_str(), bottom_node_iter->second->GetName().c_str());
                          return INTERNAL_ERROR;);
          auto op_desc = bottom_node_iter->second->GetOpDesc();
          GE_CHECK_NOTNULL(op_desc);
          auto out_op_desc = top_node_iter->second->GetOpDesc();
          GE_CHECK_NOTNULL(out_op_desc);
          (void) op_desc->UpdateInputDesc((static_cast<uint32_t>(in_archor_ptr->GetIdx())),
                                          out_op_desc->GetOutputDesc(static_cast<uint32_t>(out_archor_ptr->GetIdx())));
        }
        GE_IF_BOOL_EXEC(top_node_iter == node_map.end(),
                        ErrorManager::GetInstance().ATCReportErrMessage("E11014", {"opname"},
                                                                        {top_blob_layer_pair.first});
                        GELOGE(INTERNAL_ERROR, "[Find][TopLayer] %s failed.", top_blob_layer_pair.first.c_str());
                        return ge::FAILED;)
        GE_IF_BOOL_EXEC(top_node_iter == node_map.end(),
                        ErrorManager::GetInstance().ATCReportErrMessage("E11015", {"opname"},
                                                                        {bottom_blob_layer_pair.first});
                        GELOGE(INTERNAL_ERROR, "[Find][BottomLayer] %s failed.", bottom_blob_layer_pair.first.c_str());
                        return ge::FAILED;)
      }
    }
  }

  return SUCCESS;
}

bool CaffeModelParser::IsOutputTop(const string &op_name, const int32_t index) {
  bool ret = false;
  auto iter = ge::GetParserContext().out_nodes_map.find(op_name);
  if (iter != ge::GetParserContext().out_nodes_map.end()) {
    std::vector<int32_t> tmp_index_v;
    for (int32_t id : iter->second) {
      if (index == id) {
        ret = true;
      } else {
        tmp_index_v.emplace_back(id);
      }
    }
    // To prevent specifying network output again in the build phase, need to delete the output node in the map list.
    if (ret) {
      ge::GetParserContext().out_nodes_map.erase(op_name);
      ge::GetParserContext().out_nodes_map.emplace(op_name, tmp_index_v);
    }
  }
  return ret;
}

Status CaffeModelParser::AddUserOutNodesTop() {
  int32_t index = 0;
  const std::vector<std::pair<std::string, int32_t>> &user_out_nodes = ge::GetParserContext().user_out_nodes;
  int net_output_num = user_out_nodes.size();
  for (const auto &out_pair : user_out_nodes) {
    auto layer_iter = layer_tops_map_.find(out_pair.first);
    GELOGI("Add to output, node name: %s", out_pair.first.c_str());
    if (layer_iter != layer_tops_map_.end()) {
      if (static_cast<uint32_t>(out_pair.second) >= (layer_iter->second).size()) {
        ErrorManager::GetInstance().ATCReportErrMessage(
            "E11016", {"opname", "outputindex", "totlaloutputindex", "inputindex", "totlalinputindex"},
            {out_pair.first.c_str(), std::to_string(out_pair.second), std::to_string((layer_iter->second).size()),
             std::to_string(index), std::to_string(net_output_num)});
        GELOGE(INTERNAL_ERROR, "[Check][Size]Add op %s to NetOutput faild, current node output index:%d should < %zu. "
               "NetOutput input_index:%d should < %u.", out_pair.first.c_str(), out_pair.second,
               (layer_iter->second).size(), index, net_output_num);
        return INTERNAL_ERROR;
      }

      string top_name = layer_iter->second[out_pair.second];
      auto top_node_iter = node_map.find(out_pair.first);
      if (top_node_iter != node_map.end()) {
        ge::GetParserContext().out_top_names.push_back(top_name);
        GELOGI("The top of out node [%s] is [%s]", out_pair.first.c_str(), top_name.c_str());
      }
      ++index;
    } else {
      ErrorManager::GetInstance().ATCReportErrMessage("E11017", {"opname"}, {out_pair.first});
      GELOGE(PARAM_INVALID, "[Find][Node]Can not find out_node:%s, you should check --out_nodes.",
             out_pair.first.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status CaffeModelParser::AddOutputTop(const domi::caffe::NetParameter &proto_message) {
  for (int32_t i = 0; i < proto_message.layer_size(); i++) {
    const domi::caffe::LayerParameter &layer = proto_message.layer(i);

    if (!CheckValidLayer(layer)) {
      continue;
    }

    for (int i = 0; i < layer.top_size(); i++) {
      string top = layer.top(i);
      string top_origin = top;
      // Handling 'inplace' scenarios
      if (IsInplaceTopBlob(layer, top)) {
        top = RemapTopNameByLayer(layer, top, i);
      }

      auto t_iter = top_blobs_map_.find(top);

      GE_RETURN_WITH_LOG_IF_FALSE(t_iter != top_blobs_map_.end(),
                                  "[Check][Param]Failed to find top: %s, layer name:%s", top.c_str(),
                                  layer.name().c_str());

      // Find the bottom blob corresponding to the top blob
      auto b_iter = bottom_blobs_map_.find(t_iter->first);
      if (b_iter != bottom_blobs_map_.end() && !IsOutputTop(layer.name(), i)) {
        continue;
      }

      // If not found, add to the output side of the output
      // Find the layer for this output
      auto top_node_iter = node_map.find(layer.name());
      GELOGI("output in top_blob: %s", layer.name().c_str());
      if (top_node_iter != node_map.end()) {
        ge::GetParserContext().out_top_names.push_back(top_origin);
        ge::GetParserContext().default_out_nodes.push_back(std::make_pair(layer.name(), (int32_t)i));
        GELOGI("The top of out node [%s] is [%s]", layer.name().c_str(), top_origin.c_str());
      }
    }
  }

  return SUCCESS;
}

bool CaffeModelParser::CheckValidLayer(const domi::caffe::LayerParameter &layer) {
  if (layer.include_size() != 0) {
    bool filter_flag = false;
    for (int32_t j = 0; j < layer.include_size(); j++) {
      // Determine whether there is a data node for train in a Caffe model
      if (layer.include(j).phase() == domi::caffe::TRAIN) {
        filter_flag = true;
        break;
      }
    }

    if (filter_flag) {
      // If the phase of the data node's include is train, the data node ignores
      return false;
    }
  }

  return true;
}

bool CaffeModelParser::IsInplaceTopBlob(const domi::caffe::LayerParameter &layer, const std::string &top_name) {
  for (auto &bottom_name : layer.bottom()) {
    if (top_name == bottom_name) {
      return true;
    }
  }
  return false;
}

std::string CaffeModelParser::RemapTopNameByLayer(const domi::caffe::LayerParameter &layer, const std::string &top_name,
                                                  int index) {
  return (top_name + "_" + layer.name() + "_" + std::to_string(index));
}

Status CaffeModelParser::PreCheck(const domi::caffe::NetParameter &net) {
  // Add layer in the model to PreChecker and check the general parameters
  PreChecker::Instance().SetModelName(net.name());
  for (int i = 0; i < net.layer_size(); i++) {
    const LayerParameter &layer = net.layer(i);

    // Skip training related layers and python layers
    if (!CheckValidLayer(layer) || layer.type() == kPython) {
      continue;
    }

    // validate opname
    string mode = "^[A-Za-z0-9./_-]+$";
    if (!ge::parser::ValidateStr(layer.name(), mode)) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11018", {"opname"}, {layer.name()});
      GELOGE(ge::FAILED, "[Invoke][ValidateStr]Parse caffe pbtxt validate op[%s] failed, opname can only contain "
             "'a-z' 'A-Z' '0-9' '-' '.' '_' '/'", layer.name().c_str());
      return ge::FAILED;
    }

    GE_RETURN_WITH_LOG_IF_ERROR(PreChecker::Instance().AddOp(&layer, layer.name(), layer.type()),
                                "[Invoke][AddOp]Add layer to PreChecker failed, layer name: %s.",
                                layer.name().c_str());
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(PreChecker::Instance().CheckName(&layer) != SUCCESS, return FAILED,
                                   "[Invoke][CheckName]Check op[%s] failed, name repeat in caffe prototxt.",
                                   layer.name().c_str());
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(PreChecker::Instance().CheckType(&layer) != SUCCESS, return FAILED,
                                   "[Invoke][CheckType]Check op[%s]'s optype failed, type is not supported.",
                                   layer.name().c_str());
  }

  return SUCCESS;
}

Status CaffeModelParser::ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  bool has_error = false;

  GE_CHK_BOOL_RET_STATUS(data != nullptr, FAILED, "[Check][Param]model data  is nullptr.");
  GE_CHK_BOOL_RET_STATUS(graph != nullptr, FAILED, "[Check][Param]graph is nullptr.");

  domi::caffe::NetParameter proto_message;

  // Get Caffe network model information
  if (!ge::parser::ReadProtoFromMem(data, static_cast<int>(size), &proto_message)) {
    GELOGE(FAILED, "[Read][ProtoFromMem] ret fail");
    return FAILED;
  }

  GE_CHK_BOOL_RET_STATUS(!(proto_message.layer_size() == 0 && proto_message.layers_size() > 0), FAILED,
      "[Check][Size]The model file is consisted of layers-structure which is deprecated in caffe and unsupport in OMG."
      "It is recommended to convert layers-structure to layer-structure by caffe tool.");
  GE_CHK_BOOL_RET_STATUS((proto_message.layer_size() != 0), FAILED,
                         "[Check][Size]net layer num is zero, prototxt file may be invalid.");

  GE_RETURN_WITH_LOG_IF_ERROR(ProtoTypePassManager::Instance().Run(&proto_message, domi::CAFFE),
                              "Run ProtoType Pass Failed");
  // Set network name
  GE_IF_BOOL_EXEC((proto_message.has_name()), graph->SetName(proto_message.name()));

  // Add layer in the model to PreChecker, and perform general checks
  GE_RETURN_IF_ERROR(PreCheck(proto_message));
  has_error = PreChecker::Instance().HasError();

  if (ReorderInput(proto_message) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Reorder][Input] failed.");
    return INTERNAL_ERROR;
  }

  bool input_data_flag = false;

  // Process input of type input
  CHECK_FALSE_EXEC(ParseInput(proto_message, input_data_flag) == SUCCESS, has_error = true;
                   GELOGE(FAILED, "[Parse][Input] ret fail."));

  int32_t layer_count = proto_message.layer_size();
  std::map<std::string, std::string> inplace_blob_name_remapping;
  // Map of operator name and occurrence times
  std::map<std::string, int32_t> layer_name_map;

  // <layername,paramnames>
  std::map<std::string, std::vector<std::string>> layer_params_map;
  // same param name set <paramnames,layernames>
  // std::map<std::vector<std::string>, std::vector<std::string>> params_share_map;
  for (int32_t i = 0; i < layer_count; i++) {
    domi::caffe::LayerParameter &layer = const_cast<domi::caffe::LayerParameter &>(proto_message.layer(i));

    GE_CHK_BOOL_EXEC_INFO(CheckValidLayer(layer), continue,
                          "[Check][Layer]layer phase is train, skip this layer, name:%s, type:%s.",
                          layer.name().c_str(), layer.type().c_str());

    CHECK_FALSE_EXEC(!((layer.type() == ge::parser::DATA_TYPE) && (input_data_flag == true)), has_error = true;
                     REPORT_INNER_ERROR("E19999", "net %s has input and data layer simultaneously, check invalid."
                                        "layer name:%s, layer type:%s", proto_message.name().c_str(),
                                        layer.name().c_str(), layer.type().c_str());
                     GELOGE(FAILED, "[Check][Layer]net %s has input and data layer simultaneously, check invalid."
                            "layer name:%s, layer type:%s", proto_message.name().c_str(),
                            layer.name().c_str(), layer.type().c_str()));

    // All layer names cannot be duplicate
    // 20181208 Modified to support the existence of duplicate operators in Caffe model
    GE_IF_BOOL_EXEC(layer_name_map.find(layer.name()) != layer_name_map.end(),
                    // duplicate operator modification
                    string new_name = layer.name() + "_same_" + std::to_string(layer_name_map[layer.name()]);
                    // Times accumulation of duplicate operators
                    layer_name_map[layer.name()]++;
                    // Set the name in proto and layer
                    domi::caffe::LayerParameter *duplicate_name_layer = proto_message.mutable_layer(i);
                    duplicate_name_layer->set_name(new_name); layer.set_name(new_name);)

    // Insert the new operator name, the number of times of duplicate name is recorded as 1
    layer_name_map.insert(std::make_pair(layer.name(), kNumOne));

    // Do not exit immediately when there is an error, wait until all errors are collected before exiting
    Status ret = AddNode(layer, graph);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Add][Node]failed for layer:%s.", layer.name().c_str());
      has_error = true;
      continue;
    }

    // parse ParamSpec
    std::vector<string> v_param_names;
    for (int i = 0; i < layer.param_size(); i++) {
      const domi::caffe::ParamSpec &param = layer.param(i);
      GE_IF_BOOL_EXEC((param.has_name()), v_param_names.emplace_back(param.name()));
    }

    // Save the layer with param name parameter to map
    GE_IF_BOOL_EXEC((v_param_names.size() > 0), layer_params_map.emplace(layer.name(), v_param_names));

    GE_RETURN_WITH_LOG_IF_ERROR(AddBlobsToMap(layer, inplace_blob_name_remapping),
                                "[Add][Blobs]To Map ret fail, layer:%s.", layer.name().c_str());
  }
  // Find a layer with the same param name and save it to graph
  GE_RETURN_WITH_LOG_IF_ERROR(FindShareParamLayers(layer_params_map),
                              "[Find][ShareParamLayers] by Caffe parser ret fail.");

  // Exit if an error occurs
  GE_IF_BOOL_EXEC(has_error, return FAILED);

  GE_CHK_BOOL_RET_STATUS(top_blobs_map_.size() > 0, FAILED, "[Check][Size]current net has no output!");

  GE_RETURN_WITH_LOG_IF_ERROR(AddEdges(graph), "[Add][Edges] failed by Caffe parser, graph:%s.",
                              graph->GetName().c_str());

  if (!(ge::GetParserContext().user_out_nodes.empty())) {
    GE_RETURN_WITH_LOG_IF_ERROR(AddUserOutNodesTop(), "[Add][UserOutNodesTop] by Caffe parser failed.");
  } else {
    GE_RETURN_WITH_LOG_IF_ERROR(AddOutputTop(proto_message), "[Add][OutputTop] by Caffe parser failed.");
  }
  GE_RETURN_WITH_LOG_IF_ERROR(graph->TopologicalSorting(),
                              "[Call][TopologicalSorting] by Caffe parser failed, graph:%s.",
                              graph->GetName().c_str());

  auto nodes = graph->GetDirectNode();
  GELOGI("graph node size = %zu.", nodes.size());
  for (auto &node : nodes) {
    GELOGI("node name = %s.", node->GetName().c_str());
    for (auto &out_node : node->GetOutDataNodes()) {
      GELOGI("out node name = %s.", out_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status CaffeModelParser::Parse(const char *model_path, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(model_path);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  Status ret = Parse(model_path, compute_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parser][Model] %s for graph %s failed.", model_path, graph.GetName().c_str());
    return ret;
  }

  GELOGI("Parser model for graph %s success.", graph.GetName().c_str());
  return SUCCESS;
}

void CaffeModelParser::SaveOrigionLayerTops(domi::caffe::LayerParameter &layer) {
  string name = layer.name();
  vector<string> tops;
  for (auto top : layer.top()) {
    tops.push_back(top);
  }
  auto it = layer_tops_map_.find(name);
  if (it == layer_tops_map_.end()) {
    layer_tops_map_[name] = tops;
  }
  return;
}

Status CaffeModelParser::SaveDataLayerTops(const domi::caffe::LayerParameter &layer) {
  string name = layer.name();
  if (node_map.find(name) == node_map.end()) {
    REPORT_INNER_ERROR("E19999", "layer:%s not find in node_map after AddNode, exist error before", name.c_str());
    GELOGE(FAILED, "[Find][Node]Node can not be found by layer name: %s", name.c_str());
    return FAILED;
  }

  ge::NodePtr node = node_map[name];
  GE_CHECK_NOTNULL(node);

  if (node->GetType() == ge::parser::DATA) {
    if (layer.top_size() != 1) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11035", {"opname", "size"},
        {name, std::to_string(layer.top_size())});
      GELOGE(FAILED, "[Check][Type]Data layer[%s] top size must be 1, real size: %d", name.c_str(), layer.top_size());
      return FAILED;
    }

    string top_name = layer.top(0);
    auto data_top_names = ge::GetParserContext().data_top_names;
    if (find(data_top_names.begin(), data_top_names.end(), top_name) != data_top_names.end()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E11036", {"topname"}, {top_name});
      GELOGE(FAILED, "[Check][Node]Different data node can not have same top name: %s.", top_name.c_str());
      return FAILED;
    }
    ge::GetParserContext().data_top_names.push_back(top_name);
  }

  return SUCCESS;
}

Status CaffeModelParser::Parse(const char *model_path, ge::ComputeGraphPtr &graph) {
  bool has_error = false;
  GE_CHECK_NOTNULL(model_path);
  GE_CHECK_NOTNULL(graph);
  GELOGI("Caffe Parse model file %s", model_path);

  PreChecker::Instance().Clear();

  domi::caffe::NetParameter proto_message;

  // Get Caffe network model information
  if (ReadModelWithoutWarning(model_path, &proto_message) != SUCCESS) {
    GELOGE(FAILED, "[Read][Model] from text ret fail, model path: %s.", model_path);
    return FAILED;
  }

  // parse network model by custom proto and get custom operators
  string custom_proto_path = ge::GetParserContext().custom_proto_path + "custom.proto";
  string caffe_proto_path = ge::GetParserContext().caffe_proto_path + "caffe.proto";
  Status result = CustomProtoParse(model_path, custom_proto_path, caffe_proto_path, custom_operator_);
  if (result != SUCCESS) {
    GELOGE(FAILED, "[Parse][Model] by custom proto failed, model path: %s.", model_path);
    return FAILED;
  }

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      proto_message.layer_size() == 0 && proto_message.layers_size() > 0,
      ErrorManager::GetInstance().ATCReportErrMessage("E11021", {"realpath"}, {model_path});
      return FAILED,
             "[Check][Size]The model file[%s] is consisted of layers-structure which is deprecated in Caffe "
             "and unsupported in ATC. The \"layers\" should be changed to \"layer\".",
             model_path);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((proto_message.layer_size() == 0),
                                 ErrorManager::GetInstance().ATCReportErrMessage("E11022");
                                 return FAILED, "[Check][Size]net layer num is zero, prototxt file may be invalid.");

  GE_RETURN_WITH_LOG_IF_ERROR(ProtoTypePassManager::Instance().Run(&proto_message, domi::CAFFE),
                              "Run ProtoType Pass Failed");
  // Set network name
  GE_IF_BOOL_EXEC((proto_message.has_name() && !proto_message.name().empty()), graph->SetName(proto_message.name()));

  // Add layer in the model to PreChecker, and perform general checks
  GE_RETURN_IF_ERROR(PreCheck(proto_message));

  if (PreChecker::Instance().HasError()) {
    REPORT_INNER_ERROR("E19999", "Precheck failed. Please read check report.");
    GELOGE(INTERNAL_ERROR, "[Has][Error]Precheck failed. Please read check report.");
    return FAILED;
  }

  if (ReorderInput(proto_message) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Reorder][Input] failed.");
    return INTERNAL_ERROR;
  }

  bool input_data_flag = false;

  // Process input of type input
  CHECK_FALSE_EXEC(ParseInput(proto_message, input_data_flag) == SUCCESS, has_error = true;
                   GELOGE(FAILED, "[Parse][Input] ret fail."));

  int32_t layer_count = proto_message.layer_size();

  if (!ge::GetParserContext().user_out_nodes_top_vec.empty()) {
    GELOGW("The out_put info has top_name items.");
    GE_RETURN_WITH_LOG_IF_ERROR(ParseOutputNodeTopInfo(proto_message),
                                "[Parse][OutputNodeTopInfo] failed.");
    ge::GetParserContext().user_out_nodes_top_vec.clear();
  }

  std::map<std::string, std::string> inplace_blob_name_remapping;
  // Map of operator name and occurrence times
  std::map<std::string, int32_t> layer_name_map;

  GetParserContext().data_top_names.clear();
  // <layername,paramnames>
  std::map<std::string, std::vector<std::string>> layer_params_map;
  // same param name set <paramnames,layernames>
  for (int32_t i = 0; i < layer_count; i++) {
    domi::caffe::LayerParameter &layer = const_cast<domi::caffe::LayerParameter &>(proto_message.layer(i));
    SaveOrigionLayerTops(layer);
    GE_CHK_BOOL_EXEC_INFO(CheckValidLayer(layer), continue,
                          "[Check][Layer]layer phase is train, skip this layer, name:%s, type:%s.",
                          layer.name().c_str(), layer.type().c_str());

    CHECK_FALSE_EXEC(!((layer.type() == ge::parser::DATA_TYPE) && (input_data_flag == true)), has_error = true;
                     GELOGE(FAILED, "[Check][Layer]net %s has input and data layer simultaneously, check invalid."
                            "layer name:%s, layer type:%s", proto_message.name().c_str(),
                            layer.name().c_str(), layer.type().c_str()));

    // All layer names cannot be duplicate
    // Modified to support the existence of duplicate operators in Caffe model
    GE_IF_BOOL_EXEC(layer_name_map.find(layer.name()) != layer_name_map.end(),
                    // duplicate operator modification
                    string new_name = layer.name() + "_same_" + std::to_string(layer_name_map[layer.name()]);
                    // Times accumulation of duplicate operators
                    layer_name_map[layer.name()]++;
                    // Set the name in proto and layer
                    domi::caffe::LayerParameter *duplicate_name_layer = proto_message.mutable_layer(i);
                    duplicate_name_layer->set_name(new_name); layer.set_name(new_name);)

    // Insert the new operator name, the number of times of duplicate name is recorded as 1
    layer_name_map.insert(std::make_pair(layer.name(), kNumOne));

    // Do not exit immediately when there is an error, wait until all errors are collected before exiting
    Status ret = AddNode(layer, graph);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Add][Node] fail, layer:%s.", layer.name().c_str());
      has_error = true;
      continue;
    }

    // parse ParamSpec
    std::vector<string> v_param_names;
    for (int i = 0; i < layer.param_size(); i++) {
      const domi::caffe::ParamSpec &param = layer.param(i);
      GE_IF_BOOL_EXEC((param.has_name()), v_param_names.emplace_back(param.name()));
    }

    // Save the layer with param name parameter to map
    GE_IF_BOOL_EXEC((v_param_names.size() > 0), layer_params_map.emplace(layer.name(), v_param_names));

    GE_RETURN_WITH_LOG_IF_ERROR(AddBlobsToMap(layer, inplace_blob_name_remapping),
                                "[Add][blobs] to map ret fail, layer:%s.", layer.name().c_str());
    if (SaveDataLayerTops(layer) != SUCCESS) {
      GELOGE(FAILED, "[Save][DataLayerTops] failed, layer:%s.", layer.name().c_str());
      return FAILED;
    }
  }
  // Find a layer with the same param name and save it to graph
  GE_RETURN_WITH_LOG_IF_ERROR(FindShareParamLayers(layer_params_map),
                              "[Find][ShareParamLayers] ret fail.");

  // Exit if an error occurs
  GE_IF_BOOL_EXEC(has_error, return FAILED);

  GE_CHK_BOOL_RET_STATUS(top_blobs_map_.size() > 0, FAILED, "[Check][Size]current net has no output!");

  GE_RETURN_WITH_LOG_IF_ERROR(AddEdges(graph), "[Add][Edges] fail, graph:%s.", graph->GetName().c_str());

  if (!(ge::GetParserContext().user_out_nodes.empty())) {
    GE_RETURN_WITH_LOG_IF_ERROR(AddUserOutNodesTop(), "[Add][UserOutNodesTop] failed.");
  } else {
    GE_RETURN_WITH_LOG_IF_ERROR(AddOutputTop(proto_message), "[Add][OutputTop] failed.");
  }
  GE_RETURN_WITH_LOG_IF_ERROR(graph->TopologicalSorting(), "[Call][TopologicalSorting] failed, graph:%s.",
                              graph->GetName().c_str());

  auto nodes = graph->GetDirectNode();
  GELOGI("graph node size = %zu.", nodes.size());
  for (auto &node : nodes) {
    GELOGI("node name = %s.", node->GetName().c_str());
    for (auto &out_node : node->GetOutDataNodes()) {
      GELOGI("out node name = %s.", out_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status CaffeModelParser::FindShareParamLayers(const std::map<std::string, std::vector<std::string>> &layer_params_map) {
  for (auto p_iter = layer_params_map.begin(); p_iter != layer_params_map.end(); ++p_iter) {
    for (auto p2_iter = p_iter; p2_iter != layer_params_map.end(); ++p2_iter) {
      if (p_iter->first != p2_iter->first && p_iter->second == p2_iter->second) {
        if (params_share_map.find(p_iter->second) == params_share_map.end()) {  // Unsaved layer
          vector<string> tmp_v;
          tmp_v.push_back(p_iter->first);
          tmp_v.push_back(p2_iter->first);
          params_share_map.emplace(p_iter->second, tmp_v);
        } else {
          vector<string>::iterator iter =
              find(params_share_map[p_iter->second].begin(), params_share_map[p_iter->second].end(), p2_iter->first);
          if (iter == params_share_map[p_iter->second].end()) {
            params_share_map[p_iter->second].push_back(p2_iter->first);
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status CaffeModelParser::ToJson(const char *model_file, const char *json_file) {
  GE_CHK_BOOL_RET_STATUS(model_file != nullptr, FAILED, "[Check][Param]model_file is nullptr.");
  GE_CHK_BOOL_RET_STATUS(json_file != nullptr, FAILED, "[Check][Param]json_file is nullptr.");
  domi::caffe::NetParameter net;
  nlohmann::json j;

  GE_RETURN_WITH_LOG_IF_FALSE(ReadModelWithoutWarning(model_file, &net) == SUCCESS,
                              "[Read][Model]Without Warning failed, Please Check file:%s.", model_file);
  Pb2Json::Message2Json(net, set<string>(), j, true);
  return ModelSaver::SaveJsonToFile(json_file, j);
}

Status CaffeModelParser::ReorderInput(domi::caffe::NetParameter &net) {
  int layer_size = net.layer_size();
  for (int i = 0; i < layer_size; ++i) {
    domi::caffe::LayerParameter *layer = net.mutable_layer(i);
    const std::vector<domi::RemoveInputConfigure> &move_input_vec =
      domi::OpRegistry::Instance()->GetRemoveInputConfigure(layer->type());
    if (move_input_vec.empty()) {
      continue;
    }
    for (const auto &it : move_input_vec) {
      if (it.moveType == domi::OMG_INPUT_REORDER) {
        auto inputs = layer->bottom();
        if (static_cast<size_t>(inputs.size()) != it.input_order.size()) {
          REPORT_INNER_ERROR("E19999", "Size of input is mismatched, check invalid,"
                             "new order size is %zu, input size is %d.", it.input_order.size(), inputs.size());
          GELOGE(INTERNAL_ERROR, "[Check][Size]Size of input is mismatched, new order size is %zu, input size is %d.",
                 it.input_order.size(), inputs.size());
          return INTERNAL_ERROR;
        }
        for (size_t j = 0; j < it.input_order.size(); ++j) {
          int new_index = it.input_order[j];
          if (new_index < 0 || new_index >= inputs.size()) {
            REPORT_INNER_ERROR("E19999", "New order of %s has invalid index %d, which is out of range, "
                               "inputs size:%d.", layer->name().c_str(), new_index, inputs.size());
            GELOGE(INTERNAL_ERROR, "[Check][Param]New order of %s has invalid index %d, which is out of range, "
                   "inputs size:%d.", layer->name().c_str(), new_index, inputs.size());
            return INTERNAL_ERROR;
          }
          layer->set_bottom(j, inputs[new_index]);
        }
        GELOGI("The input sequence of the node has been rearranged, node name:%s.", layer->name().c_str());
      }
    }
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  if (data == nullptr) {
    REPORT_INNER_ERROR("E19999", "param data is nullptr.");
    GELOGE(PARAM_INVALID, "[Check][Param]Caffe weights data is nullptr");
    return PARAM_INVALID;
  }

  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "param graph is nullptr.");
    GELOGE(PARAM_INVALID, "[Check][Param]Caffe weights graph is nullptr");
    return PARAM_INVALID;
  }

  // Resolve proto file to netparameter
  NetParameter proto;
  bool success = ge::parser::ReadProtoFromArray(reinterpret_cast<const char *>(data), static_cast<int>(size), &proto);
  if (!success) {
    REPORT_CALL_ERROR("E19999", "ReadProtoFromArray failed.");
    GELOGE(domi::PARSE_WEIGHTS_FAILED, "[Read][Proto] from Memory fail");
    return domi::PARSE_WEIGHTS_FAILED;
  }

  // Convert netparameter to opdef and save to graph
  Status status = ConvertNetParameter(proto, graph);
  GE_IF_BOOL_EXEC(status != SUCCESS, GELOGE(FAILED, "[Convert][NetParameter] failed, status=%d", status);
                  return domi::PARSE_WEIGHTS_FAILED;);

  return SUCCESS;
}

Status CaffeWeightsParser::Parse(const char *file, ge::Graph &graph) {
  ErrorManager::GetInstance().SetStage(error_message::kModelCompile, error_message::kParser);
  GE_CHECK_NOTNULL(file);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);

  Status ret = Parse(file, compute_graph);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parser][Weight] %s for graph %s failed.", file, graph.GetName().c_str());
    return ret;
  }

  GELOGI("Parser weight for graph %s success.", graph.GetName().c_str());
  return SUCCESS;
}

Status CaffeWeightsParser::Parse(const char *file, ge::ComputeGraphPtr &graph) {
  if (file == nullptr) {
    REPORT_INNER_ERROR("E19999", "param file is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param]Caffe weights parse fail, Parameter file invalid");
    return PARAM_INVALID;
  }

  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "param graph is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param]Caffe weights parse fail, Parameter graph invalid");
    return PARAM_INVALID;
  }

  GELOGI("Parse weights file:%s", file);

  string caffe_proto_path = ge::GetParserContext().caffe_proto_path + "caffe.proto";
  string custom_proto_path = ge::GetParserContext().custom_proto_path + "custom.proto";
  ProtoFileParser proto_file_parser;

  GELOGD("caffe_proto_path:%s custom_proto_path:%s", caffe_proto_path.c_str(), custom_proto_path.c_str());
  string fusion_proto_file;
  string custom_proto_file = ge::parser::RealPath(custom_proto_path.c_str());
  if (custom_proto_file.empty()) {
    GELOGW("custom_proto_path:%s is not existed", custom_proto_path.c_str());
    fusion_proto_file = caffe_proto_path;
  } else {
    if (proto_file_parser.CombineProtoFile(caffe_proto_path.c_str(), custom_proto_path.c_str(),\
        fusion_proto_file) != SUCCESS) {
      REPORT_INNER_ERROR("E19999", "CombineProtoFile failed, caffe_proto_path:%s, custom_proto_path:%s.",
                         caffe_proto_path.c_str(), custom_proto_path.c_str());
      GELOGE(FAILED, "[Invoke][CombineProtoFile]Create tmp fusion proto file from caffe and custom proto failed.");
      return FAILED;
    }
  }

  string fusion_proto_path = ge::parser::RealPath(fusion_proto_file.c_str());
  GELOGI("Get fusion proto file[%s]-[%s].", fusion_proto_file.c_str(), fusion_proto_path.c_str());
  if (fusion_proto_path.empty()) {
    REPORT_INNER_ERROR("E19999", "Fusion proto file path [%s] is not real existed.",
                       fusion_proto_file.c_str());
    GELOGE(FAILED, "[Invoke][RealPath]Fusion proto file path [%s]-[%s] is not real existed.",
           fusion_proto_file.c_str(), fusion_proto_path.c_str());
    return FAILED;
  }

  string fusion_proto_name;
  if (CheckPathValid(file, fusion_proto_file, fusion_proto_path, fusion_proto_name) != SUCCESS) {
    GELOGE(FAILED, "[Check][PathValid] of weight file[%s] and tmp proto[%s] failed.", file,
           fusion_proto_file.c_str());
    return FAILED;
  }

  GELOGI("Start to parse weight: %s by fusion proto: %s.", file, fusion_proto_file.c_str());
  Status status = ParseWeightByFusionProto(file, fusion_proto_path, fusion_proto_name, graph);
  if (status != SUCCESS) {
    GELOGE(FAILED, "[Invoke][ParseWeightByFusionProto] failed. ret:%u", status);
    return status;
  }

  status = CheckNodes(graph);
  if (status != SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Nodes] failed, status=%u", status);
    return domi::PARSE_WEIGHTS_FAILED;
  }

  return SUCCESS;
}

Status CaffeWeightsParser::ParseWeightByFusionProto(const char *weight_path, const string &fusion_proto_path,
                                                    const string &fusion_proto_name, ge::ComputeGraphPtr &graph) {
  google::protobuf::compiler::DiskSourceTree source_tree;
  source_tree.MapPath(kProjectRoot, fusion_proto_path);
  google::protobuf::compiler::Importer importer(&source_tree, nullptr);
  importer.Import(fusion_proto_name.c_str());
  GELOGI("Import fusion proto %s success, proto_name %s.", fusion_proto_path.c_str(), fusion_proto_name.c_str());

  const google::protobuf::Descriptor *descriptor = importer.pool()->FindMessageTypeByName(kBeginningMessageType);
  if (descriptor == nullptr) {
    REPORT_INPUT_ERROR("E11032", std::vector<std::string>({"message_type", "name", "reason"}),
                       std::vector<std::string>({"weight", "NetParameter",
                           "Does not find domi.caffe.NetParameter in google::protobuf::Descriptor."}));
    GELOGE(FAILED, "[Invoke][FindMessageTypeByName]Does not find domi.caffe.NetParameter in "
           "google::protobuf::Descriptor, which may be caused by problematic fusion proto.");
    return FAILED;
  }
  google::protobuf::DynamicMessageFactory factory;
  const google::protobuf::Message *proto = factory.GetPrototype(descriptor);
  GE_CHECK_NOTNULL(proto);
  google::protobuf::Message *message = proto->New();
  GE_CHECK_NOTNULL(message);

  if (!ge::parser::ReadProtoFromBinaryFile(weight_path, message)) {
    delete message;
    message = nullptr;
    REPORT_CALL_ERROR("E19999", "ReadProtoFromBinaryFile based on fusion proto failed from weight file:%s.",
                      weight_path);
    GELOGE(FAILED, "[Invoke][ReadProtoFromBinaryFile] %s failed.", weight_path);
    return FAILED;
  }

  GELOGI("Start to parse weight file: %s.", weight_path);
  const google::protobuf::Descriptor *layer_descriptor = importer.pool()->FindMessageTypeByName(kLayerMessageType);
  if (layer_descriptor == nullptr) {
    delete message;
    message = nullptr;
    REPORT_INPUT_ERROR("E11032", std::vector<std::string>({"message_type", "name", "reason"}),
                       std::vector<std::string>({"weight", "NetParameter",
                           "Does not find domi.caffe.LayerParameter in google::protobuf::Descriptor"}));
    GELOGE(FAILED,
           "[Invoke][FindMessageTypeByName]Does not find domi.caffe.LayerParameter in google::protobuf::Descriptor");
    return FAILED;
  }

  if (CheckLayersSize(message) != SUCCESS) {
    delete message;
    message = nullptr;
    return FAILED;
  }

  if (ParseLayerParameter(layer_descriptor, message, graph) != SUCCESS) {
    delete message;
    message = nullptr;
    REPORT_CALL_ERROR("E19999", "ParseLayerParameter failed failed from weight file:%s.", weight_path);
    GELOGE(FAILED, "[Parse][LayerParameter] failed.");
    return FAILED;
  }

  delete message;
  message = nullptr;
  GELOGI("Parse weight: %s by proto: %s success.", weight_path, fusion_proto_path.c_str());
  return SUCCESS;
}

Status CaffeWeightsParser::ParseLayerParameter(const google::protobuf::Descriptor *layer_descriptor,
                                               const google::protobuf::Message *message,
                                               ge::ComputeGraphPtr &graph) {
  auto field_name = layer_descriptor->FindFieldByName(kFieldName);
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field_name, "Does not find name in google::protobuf::Descriptor");
  auto field_type = layer_descriptor->FindFieldByName(kFieldType);
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field_type, "Does not find type in google::protobuf::Descriptor");

  const google::protobuf::Reflection *reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);

  NetParameter tmp_net;
  for (auto &field : field_desc) {
    CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field, "Get FieldDescriptor failed in google::protobuf::Message");
    // Only care about layers
    GE_CHECK_NOTNULL(field);
    if (field->name() != kLayerName) {
      continue;
    }
    if (!field->is_repeated()) {
      REPORT_INPUT_ERROR("E11032", std::vector<std::string>({"message_type", "name", "reason"}),
                         std::vector<std::string>({"weight", field->name(), "LayerParameter should be repeated"}));
      GELOGE(FAILED, "[Check][Param] LayerParameter should be repeated, field:%s.", field->name().c_str());
      return FAILED;
    }

    int field_size = reflection->FieldSize(*message, field);
    GELOGI("Total Layer num of model file is %d", field_size);
    for (int i = 0; i < field_size; ++i) {
      const google::protobuf::Message &layer_message = reflection->GetRepeatedMessage(*message, field, i);

      LayerParameter *layer = tmp_net.add_layer();
      if (ConvertLayerProto(&layer_message, layer) != SUCCESS) {
         GELOGE(FAILED, "[Invoke][ConvertLayerProto] Convert message to layer proto failed.");
         return FAILED;
      }

      const string &layer_name = layer->name();
      if (skiped_layer_type_.find(layer->type()) != skiped_layer_type_.end()) {
        GELOGI("Skip layer %s", layer_name.c_str());
        continue;
      }

      GELOGI("Parse layer %s", layer_name.c_str());
      auto ret = ConvertLayerParameter(layer, graph);
      if (ret != SUCCESS) {
        return ret;
      }
    }
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ConvertLayerProto(const google::protobuf::Message *message,
                                             google::protobuf::Message *layer) {
  const google::protobuf::Reflection *layer_reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(layer_reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  layer_reflection->ListFields(*message, &field_desc);

  for (auto &field : field_desc) {
    GE_CHECK_NOTNULL(field);
    if (ParseLayerField(layer_reflection, message, field, layer) != SUCCESS) {
      GELOGE(FAILED, "[Invoke][ParseLayerField] Parse field %s failed.", field->name().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ParseLayerField(const google::protobuf::Reflection *reflection,
                                           const google::protobuf::Message *message,
                                           const google::protobuf::FieldDescriptor *field,
                                           google::protobuf::Message *layer) {
  GELOGD("Start to parse field: %s.", field->name().c_str());
  domi::caffe::LayerParameter *layer_proto = reinterpret_cast<domi::caffe::LayerParameter *>(layer);
  string filed_name = field->name();
#define CASE_FIELD_NAME(kName, method)                                 \
  if (filed_name == kField##kName) {                                   \
    string value = reflection->GetString(*message, field);             \
    GELOGD("Parse result(%s : %s)", filed_name.c_str(), value.c_str());\
    layer_proto->set_##method(value);                                  \
    return SUCCESS;                                                    \
  }
  CASE_FIELD_NAME(Name, name);
  CASE_FIELD_NAME(Type, type);
#undef CASE_FIELD_NAME
#define CASE_FIELD_NAME_REPEATED(kName, method)                        \
  if (filed_name == kField##kName) {                                   \
    int field_size = reflection->FieldSize(*message, field);           \
    for (int i = 0; i < field_size; ++i) {                             \
      string value = reflection->GetRepeatedString(*message, field, i);\
      layer_proto->add_##method(value);                                \
    }                                                                  \
    return SUCCESS;                                                    \
  }
  CASE_FIELD_NAME_REPEATED(Bottom, bottom);
  CASE_FIELD_NAME_REPEATED(Top, top);
#undef CASE_FIELD_NAME_REPEATED
  if (filed_name == kFieldBlobs) {
    int field_size = reflection->FieldSize(*message, field);
    for (int i = 0; i < field_size; ++i) {
      BlobProto *item_message = layer_proto->add_blobs();
      const google::protobuf::Message &sub_message = reflection->GetRepeatedMessage(*message, field, i);
      if (ConvertBlobsProto(&sub_message, item_message) != SUCCESS) {
        GELOGE(FAILED, "[Invoke][ConvertBlobsProto] ParseLayerField of field: %s failed.", field->name().c_str());
        return FAILED;
      }
    }
    return SUCCESS;
  }
  if (filed_name == kFieldConvParam) {
    const google::protobuf::Message &sub_message = reflection->GetMessage(*message, field);
    ConvolutionParameter *conv_param = layer_proto->mutable_convolution_param();
    ConvertConvParamProto(&sub_message, conv_param);
  }
  if (filed_name == kFieldInnerPro) {
    const google::protobuf::Message &sub_message = reflection->GetMessage(*message, field);
    InnerProductParameter *inner_product = layer_proto->mutable_inner_product_param();
    ConvertInnerProdcutProto(&sub_message, inner_product);
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ConvertBlobsProto(const google::protobuf::Message *message,
                                             google::protobuf::Message *blobs) {
  const google::protobuf::Reflection *blobs_reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(blobs_reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  blobs_reflection->ListFields(*message, &field_desc);

  domi::caffe::BlobProto *blobs_proto = reinterpret_cast<domi::caffe::BlobProto *>(blobs);

  for (auto &field : field_desc) {
    GE_CHECK_NOTNULL(field);
    string feild_name = field->name();
#define CASE_BLOBS_FIELD_NAME_REPEATED(kName, method, valuetype, name)                \
  if (feild_name == #kName) {                                                         \
    int field_size = blobs_reflection->FieldSize(*message, field);                    \
    for (int i = 0; i < field_size; ++i) {                                            \
      valuetype value = blobs_reflection->GetRepeated##method(*message, field, i);    \
      blobs_proto->add_##name(value);                                                 \
    }                                                                                 \
    continue;                                                                         \
  }
    CASE_BLOBS_FIELD_NAME_REPEATED(data, Float, float,  data);
    CASE_BLOBS_FIELD_NAME_REPEATED(diff, Float, float,  diff);
    CASE_BLOBS_FIELD_NAME_REPEATED(double_data, Double, double,  double_data);
    CASE_BLOBS_FIELD_NAME_REPEATED(double_diff, Double, double,  double_diff);
    CASE_BLOBS_FIELD_NAME_REPEATED(int32_data, Int32, int32_t,  int32_data);
    CASE_BLOBS_FIELD_NAME_REPEATED(uint64_data, UInt64, uint64_t,  uint64_data);
#undef CASE_BLOBS_FIELD_NAME_REPEATED
#define CASE_BLOBS_FIELD_NAME(kName, method, valuetype, name)                        \
  if (feild_name == #kName) {                                                        \
    valuetype value = blobs_reflection->Get##method(*message, field);                \
    blobs_proto->set_##name(value);                                                  \
    continue;                                                                        \
  }
    CASE_BLOBS_FIELD_NAME(int8_data, String, string,  int8_data);
    CASE_BLOBS_FIELD_NAME(num, Int32, int32_t,  num);
    CASE_BLOBS_FIELD_NAME(channels, Int32, int32_t,  channels);
    CASE_BLOBS_FIELD_NAME(height, Int32, int32_t,  height);
    CASE_BLOBS_FIELD_NAME(width, Int32, int32_t,  width);
#undef CASE_BLOBS_FIELD_NAME
    if (feild_name == kFieldShape) {
      const google::protobuf::Message &sub_message = blobs_reflection->GetMessage(*message, field);
      domi::caffe::BlobShape *blob_shape = blobs_proto->mutable_shape();
      ConvertBlobShapeProto(&sub_message, blob_shape);
    }
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ConvertBlobShapeProto(const google::protobuf::Message *message,
                                                 google::protobuf::Message *dest_message) {
  const google::protobuf::Reflection *reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);

  domi::caffe::BlobShape *shape_proto = reinterpret_cast<domi::caffe::BlobShape *>(dest_message);

  for (auto &field : field_desc) {
    if (field->name() != kFieldDim) {
      continue;
    }
    int field_size = reflection->FieldSize(*message, field);
    for (int i = 0; i < field_size; ++i) {
      int64_t value = reflection->GetRepeatedInt64(*message, field, i);
      shape_proto->add_dim(value);
    }
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ConvertConvParamProto(const google::protobuf::Message *message,
                                                 google::protobuf::Message *dest_message) {
  const google::protobuf::Reflection *reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);

  domi::caffe::ConvolutionParameter *conv_param_proto =
      reinterpret_cast<domi::caffe::ConvolutionParameter *>(dest_message);

  for (auto &field : field_desc) {
    if (field->name() != kFieldBiasTerm) {
      continue;
    }
    bool value = reflection->GetBool(*message, field);
    conv_param_proto->set_bias_term(value);
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ConvertInnerProdcutProto(const google::protobuf::Message *message,
                                                    google::protobuf::Message *dest_message) {
  const google::protobuf::Reflection *reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);

  domi::caffe::InnerProductParameter *inner_product_proto =
      reinterpret_cast<domi::caffe::InnerProductParameter *>(dest_message);

  for (auto &field : field_desc) {
    if (field->name() != kFieldBiasTerm) {
      continue;
    }
    bool value = reflection->GetBool(*message, field);
    inner_product_proto->set_bias_term(value);
  }
  return SUCCESS;
}

Status CaffeWeightsParser::CheckLayersSize(const google::protobuf::Message *message) {
  const google::protobuf::Reflection *reflection = message->GetReflection();
  CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(reflection, "Get Reflection failed in google::protobuf::Message");
  vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);

  int num_layer = 0;
  int num_layers = 0;

  for (auto &field : field_desc) {
    CAFFE_CHECK_NULL_AND_REPROT_ERRORMSG(field, "Get FieldDescriptor failed in google::protobuf::Message");
    // Only care about layers
    if (field->name() != kLayerName && field->name() != kLayersName) {
      continue;
    }
    if (!field->is_repeated()) {
      REPORT_INPUT_ERROR("E11032", std::vector<std::string>({"message_type", "name", "reason"}),
                         std::vector<std::string>({"weight", field->name(), "LayerParameter should be repeated"}));
      GELOGE(FAILED, "[Check][Param] LayerParameter should be repeated. field:%s", field->name().c_str());
      return FAILED;
    }

    int field_size = reflection->FieldSize(*message, field);
    if (field->name() == kLayerName) {
      num_layer = field_size;
    } else {
      num_layers = field_size;
    }
  }

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(num_layer == 0 && num_layers > 0,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E11023");
                                 return FAILED,
                                 "[Check][Param]The weight file is consisted of layers-structure which is deprecated "
                                 "in Caffe and unsupported in ATC. The \"layers\" should be changed to \"layer\".");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((num_layer == 0), ErrorManager::GetInstance().ATCReportErrMessage("E11024");
                                 return FAILED,
                                 "[Check][Param] Weight layer num is zero, weight file may be invalid.");

  return SUCCESS;
}

Status CaffeWeightsParser::ConvertLayerParameter(const google::protobuf::Message *layer_message,
                                                 ge::ComputeGraphPtr &graph) {
  vector<string> need_share_layers;
  const domi::caffe::LayerParameter *layer = reinterpret_cast<const domi::caffe::LayerParameter *>(layer_message);
  const string &layer_name = layer->name();
  const string &layer_type = layer->type();
  for (auto p_iter = params_share_map.begin(); p_iter != params_share_map.end(); ++p_iter) {
    if (find(p_iter->second.begin(), p_iter->second.end(), layer_name) != p_iter->second.end()) {
      GELOGI("layer:%s need share weights !", layer_name.c_str());
      need_share_layers = p_iter->second;
    }
  }

  if (need_share_layers.size() == 0) {
    need_share_layers.push_back(layer_name);
  }

  for (auto share_iter = need_share_layers.begin(); share_iter != need_share_layers.end(); ++share_iter) {
    // Find created nodes
    string layer_name = *share_iter;
    GE_IF_BOOL_EXEC(layer_name_record_map_.find(layer_name) != layer_name_record_map_.end(),
                    string temp_layer_name = layer_name;
                    // duplicate operator modification
                    layer_name = temp_layer_name + "_same_" + std::to_string(layer_name_record_map_[temp_layer_name]);
                    // Times accumulation of duplicate operators
                    layer_name_record_map_[temp_layer_name]++;
                    // Set the name in proto and layer
                    )
    ge::NodePtr node = graph->FindNode(layer_name);
    layer_name_record_map_.insert(std::make_pair(layer_name, kNumOne));
    if (node == nullptr) {
      // If there are redundant layers in the weight file, they should be skipped rather than returned with an error.
      GELOGI("Layer %s not found in graph", layer_name.c_str());
      continue;
    }

    // The weight processing also needs to judge the duplicate operator, which is reserved here and processed later.
    auto iter = caffe_op_map.find(layer_type);
    if (iter == caffe_op_map.end()) {
      GELOGW("Unrecognized layer type %s , layer name: %s, layer ignored.", layer_type.c_str(), layer_name.c_str());
      continue;
    }
    GELOGD("Caffe layer name: %s , layer type: %s.", layer_name.c_str(), layer_type.c_str());
    string op_type = iter->second;

    // create OpParser
    std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::CAFFE);
    GE_CHECK_NOTNULL(factory);
    std::shared_ptr<OpParser> op_parser = factory->CreateOpParser(op_type);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        (op_parser.get() == nullptr),
        REPORT_INPUT_ERROR("E11009", std::vector<std::string>({"opname", "optype"}),
                           std::vector<std::string>({layer_name, op_type}));
        return FAILED,
        "[Create][OpParser] failed for Op[%s], optype is %s", layer_name.c_str(), op_type.c_str());

    // Parsing weight information through op parser
    Status status = op_parser->ParseWeights(layer_message, node);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        (status != SUCCESS),
        REPORT_CALL_ERROR("E19999", "Parse weight for op:%s(%s) failed", layer_name.c_str(), op_type.c_str());
        return status,
        "[Parse][Weights] for op[%s] failed", layer_name.c_str());
  }
  return SUCCESS;
}

Status CaffeWeightsParser::CheckNodes(ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  for (const ge::NodePtr &node : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &in_anchor_ptr : node->GetAllInDataAnchors()) {
      if (op_desc->GetType() == ge::parser::DATA || op_desc->GetType() == ge::parser::CONSTANT) {
        continue;
      }
      auto index = in_anchor_ptr->GetIdx();
      auto input_desc = op_desc->MutableInputDesc(index);
      if (in_anchor_ptr->GetPeerAnchors().empty() && input_desc != nullptr) {
        if (layer_name_record_map_.find(node->GetName()) == layer_name_record_map_.end()) {
          ErrorManager::GetInstance().ATCReportErrMessage("E11029", {"opname"}, {node->GetName()});
          GELOGE(ge::GRAPH_FAILED, "[Find][Node] Op[%s] in model file does not exist in weight file.",
                 node->GetName().c_str());
          PreChecker::Instance().RefreshErrorMessageByName(node->GetName(), PreChecker::PARAM_INVALID,
                                                           "Node does not exist in weight file.");
        } else {
          REPORT_INNER_ERROR("E19999", "Op:%s(%s)'s input %d is not linked, check invalid",
                             node->GetName().c_str(), node->GetType().c_str(), in_anchor_ptr->GetIdx());
          GELOGE(ge::GRAPH_FAILED, "[Check][Param] Op[%s]'s input %d is not linked.", node->GetName().c_str(),
                 in_anchor_ptr->GetIdx());
          string check_msg = "input " + to_string(in_anchor_ptr->GetIdx()) + "is not linked in weight file";
          PreChecker::Instance().RefreshErrorMessageByName(node->GetName(), PreChecker::PARAM_INVALID, check_msg);
        }
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CaffeWeightsParser::ConvertNetParameter(const NetParameter &param, ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  int num_layer = param.layer_size();
  int num_layers = param.layers_size();

  // Operator name and occurrence map, handle duplicate operators
  std::map<std::string, int32_t> layer_name_map;

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(num_layer == 0 && num_layers > 0,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E11023");
                                 return FAILED, "[Check][Param] The weight file is consisted of layers-structure "
                                 "which is deprecated in Caffe and unsupported in ATC. "
                                 "The \"layers\" should be changed to \"layer\".");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((num_layer == 0), ErrorManager::GetInstance().ATCReportErrMessage("E11024");
                                 return FAILED, "weight layer num is zero, weight file may be invalid.");

  for (int i = 0; i < num_layer; ++i) {
    const LayerParameter &layer = param.layer(i);
    const string &layer_name = layer.name();

    // Skip some layer types
    if (skiped_layer_type_.find(layer.type()) != skiped_layer_type_.end()) {
      GELOGI("Skip layer %s", layer_name.c_str());
      continue;
    }

    GELOGI("Parse layer %s", layer_name.c_str());

    vector<string> need_share_layers;

    for (auto p_iter = params_share_map.begin(); p_iter != params_share_map.end(); ++p_iter) {
      if (find(p_iter->second.begin(), p_iter->second.end(), layer_name) != p_iter->second.end()) {
        GELOGI("Layer: %s need share weights !", layer_name.c_str());
        need_share_layers = p_iter->second;
      }
    }

    if (need_share_layers.size() == 0) {
      need_share_layers.push_back(layer_name);
    }

    for (auto share_iter = need_share_layers.begin(); share_iter != need_share_layers.end(); ++share_iter) {
      // Find created nodes
      string layer_name = *share_iter;
      GE_IF_BOOL_EXEC(layer_name_map.find(layer_name) != layer_name_map.end(), string temp_layer_name = layer_name;
                      // duplicate operator modification
                      layer_name = temp_layer_name + "_same_" + std::to_string(layer_name_map[temp_layer_name]);
                      // Times accumulation of duplicate operators
                      layer_name_map[temp_layer_name]++;
                      // Set the name in proto and layer
                      )
      ge::NodePtr node = graph->FindNode(layer_name);
      layer_name_map.insert(std::make_pair(layer_name, kNumOne));
      if (node == nullptr) {
        // If there are redundant layers in the weight file, they should be skipped rather than returned with an error.
        GELOGI("Layer %s not found in graph", layer_name.c_str());
        continue;
      }

      // The weight processing also needs to judge the duplicate operator, which is reserved here and processed later.
      auto iter = caffe_op_map.find(layer.type());
      if (iter == caffe_op_map.end()) {
        GELOGW("Unrecognized layer type %s , layer name: %s, layer ignored.", layer.type().c_str(), layer_name.c_str());
        continue;
      }
      GELOGD("Caffe layer name: %s , layer type: %s.", layer_name.c_str(), layer.type().c_str());
      string op_type = iter->second;

      // create OpParser
      std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::CAFFE);
      GE_CHECK_NOTNULL(factory);
      std::shared_ptr<OpParser> op_parser = factory->CreateOpParser(op_type);

      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
          (op_parser.get() == nullptr),
          REPORT_INPUT_ERROR("E11009", std::vector<std::string>({"opname", "optype"}),
                             std::vector<std::string>({layer_name, op_type}));
          return FAILED, "[Create][OpParser] failed for Op[%s], optype is %s", layer_name.c_str(), op_type.c_str());

      // Parsing weight information through op parser
      Status status = op_parser->ParseWeights(&layer, node);
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
          (status != SUCCESS),
          REPORT_CALL_ERROR("E19999", "Parse weight for op:%s(%s) failed", layer_name.c_str(), op_type.c_str());
          return status, "[Parse][Weights] for op[%s] failed", layer_name.c_str());
    }
  }

  return SUCCESS;
}

Status CaffeModelParser::ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) {
  return SUCCESS;
}
Status CaffeModelParser::ParseProtoWithSubgraph(const google::protobuf::Message *root_proto,
                                                domi::GetGraphCallback callback,
                                                ge::ComputeGraphPtr &graph) {
  return SUCCESS;
}
}  // namespace ge

namespace domi {
  REGISTER_MODEL_PARSER_CREATOR(CAFFE, ge::CaffeModelParser);
  REGISTER_WEIGHTS_PARSER_CREATOR(CAFFE, ge::CaffeWeightsParser);
}
