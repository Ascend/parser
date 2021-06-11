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

#include "parser/caffe/caffe_data_parser.h"
#include <unordered_map>
#include <utility>
#include "framework/omg/parser/parser_types.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "parser/common/op_parser_factory.h"

using namespace ge::parser;

namespace ge {
Status CaffeDataParser::GetOutputDesc(const string &name, int dim_size, const std::vector<int64_t> &input_dims,
                                      ge::OpDescPtr &op) {
  GE_CHECK_NOTNULL(op);
  GELOGI("The input dim size is %zu in layer %s.", input_dims.size(), name.c_str());

  // Caffe default data type is float32
  GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetInt(op, DATA_ATTR_NAME_DATA_TYPE, ge::DT_FLOAT)),
                  GELOGW("SetInt failed for op %s.", op->GetName().c_str()););  // no need to return

  // Initialize input and output description of OP according to input_dims information
  GE_RETURN_WITH_LOG_IF_ERROR(ParseShape(input_dims, op), "[Parse][Shape] failed for data layer %s", name.c_str());

  return SUCCESS;
}

Status CaffeDataParser::ParseParams(const Message *op_src, ge::OpDescPtr &op) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);
  const domi::caffe::LayerParameter *layer = DOMI_DYNAMIC_CAST<const domi::caffe::LayerParameter *>(op_src);
  GE_CHECK_NOTNULL(layer);
  GELOGD("Caffe layer name = %s, layer type= %s, parse params", layer->name().c_str(), layer->type().c_str());

  if (layer->type() == ge::parser::INPUT_TYPE) {
    GE_CHK_STATUS_RET(ParseParamsForInput(layer, op), "[Parse][Params] failed, Caffe layer name = %s, "
                      "layer type= %s", layer->name().c_str(), layer->type().c_str());
  } else if(layer->type() == ge::parser::DUMMY_DATA) {
    GE_CHK_STATUS_RET(ParseParamsForDummyData(layer, op), "[Parse][Params] failed, Caffe layer name = %s, "
                      "layer type= %s", layer->name().c_str(), layer->type().c_str());
  } else {
    REPORT_INNER_ERROR("E19999", "layer:%s(%s) type is not %s or %s, check invalid",
                       layer->name().c_str(), layer->type().c_str(),
                       ge::parser::INPUT_TYPE.c_str(), ge::parser::DUMMY_DATA.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Caffe prototxt has no optype [Input]");
    return FAILED;
  }
  return SUCCESS;
}

Status CaffeDataParser::ParseParamsForInput(const domi::caffe::LayerParameter *layer, ge::OpDescPtr &op) {
  if (layer->has_input_param()) {
    const domi::caffe::InputParameter &input_param = layer->input_param();
    if (input_param.shape_size() == 0) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E11027", {"layername", "layertype"}, {layer->name(), layer->type()});
      GELOGE(PARAM_INVALID, "[Check][Param]input_param shape size is zero, check invalid, "
             "caffe layer name [%s], layer type [%s].", layer->name().c_str(), layer->type().c_str());
      return FAILED;
    }
    for (int i = 0; i < input_param.shape_size(); i++) {
      const domi::caffe::BlobShape &blob_shape = input_param.shape(i);
      vector<int64_t> shape;
      std::map<string, vector<int64_t>> &shape_map = GetParserContext().input_dims;
      std::vector<int64_t> model_dims;
      for (auto &blob_shape_dim_temp : blob_shape.dim()) {
        model_dims.push_back(blob_shape_dim_temp);
      }
      string name = layer->name();
      GE_IF_BOOL_EXEC(shape_map.count(name) != 0, model_dims = shape_map.at(name));
      GE_CHK_STATUS_RET(GetOutputDesc(name, model_dims.size(), model_dims, op),
                        "[Get][OutputDesc] failed in layer %s", name.c_str());
    }
  } else {
    // Get from external input
    const ge::ParserContext &ctx = GetParserContext();
    std::map<std::string, std::vector<int64_t>> input_dims = ctx.input_dims;
    string name = layer->name();
    auto search = input_dims.find(name);
    if (search == input_dims.end()) {
      REPORT_INPUT_ERROR("E11005", std::vector<std::string>({"input"}), std::vector<std::string>({layer->name()}));
      GELOGE(PARAM_INVALID, "[Check][Param] Caffe prototxt has no input_param or user "
             "should set --input_shape in atc parameter, caffe layer name [%s], layer type [%s].",
             layer->name().c_str(), layer->type().c_str());
      return FAILED;
    }
    std::vector<int64_t> dims = search->second;
    GE_CHK_STATUS_RET(GetOutputDesc(name, dims.size(), dims, op),
                      "[Get][OutputDesc] failed in layer %s.", name.c_str());
  }
  return SUCCESS;
}

Status CaffeDataParser::ParseParamsForDummyData(const domi::caffe::LayerParameter *layer, ge::OpDescPtr &op) {
  if (layer->has_dummy_data_param()) {
    const domi::caffe::DummyDataParameter &dummy_data_param = layer->dummy_data_param();
    if (dummy_data_param.shape_size() == 0) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E11027", {"layername", "layertype"}, {layer->name(), layer->type()});
      GELOGE(PARAM_INVALID, "[Check][Param] input_param shape size is zero, caffe layer name [%s], layer type [%s].",
             layer->name().c_str(), layer->type().c_str());
      return FAILED;
    }
    for (int i = 0; i < dummy_data_param.shape_size(); i++) {
      const domi::caffe::BlobShape &blob_shape = dummy_data_param.shape(i);

      vector<int64_t> shape;
      std::map<string, vector<int64_t>> &shape_map = GetParserContext().input_dims;
      std::vector<int64_t> model_dims;
      for (auto &blob_shape_dim_temp : blob_shape.dim()) {
        model_dims.push_back(blob_shape_dim_temp);
      }

      string name = layer->name();
      GE_IF_BOOL_EXEC(shape_map.count(name) != 0, model_dims = shape_map.at(name));
      GE_CHK_STATUS_RET(GetOutputDesc(name, model_dims.size(), model_dims, op),
                        "[Get][OutputDesc] failed in layer %s", name.c_str());
    }
  } else {
    // Get from external input
    const ge::ParserContext &ctx = GetParserContext();
    std::map<std::string, std::vector<int64_t>> input_dims = ctx.input_dims;
    string name = layer->name();
    auto search = input_dims.find(name);
    if (search == input_dims.end()) {
      REPORT_INPUT_ERROR("E11005", std::vector<std::string>({"input"}), std::vector<std::string>({layer->name()}));
      GELOGE(PARAM_INVALID, "[Check][Param] Caffe prototxt has no input_param or user "
             "should set --input_shape in atc parameter, caffe layer name [%s], layer type [%s].",
             layer->name().c_str(), layer->type().c_str());
      return FAILED;
    }
    std::vector<int64_t> dims = search->second;
    GE_CHK_STATUS_RET(GetOutputDesc(name, dims.size(), dims, op),
                      "[Get][OutputDesc] failed in layer %s.", name.c_str());
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(CAFFE, DATA, CaffeDataParser);
}  // namespace ge
