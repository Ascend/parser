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

#include "parser/common/register_tbe.h"
#include <map>
#include <memory>
#include <string>
#include "parser/common/acl_graph_parser_util.h"
#include "common/op_map.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_fusion_custom_parser_adapter.h"

namespace ge {
using PARSER_CREATOR_FN = std::function<std::shared_ptr<OpParser>(void)>;

FMK_FUNC_HOST_VISIBILITY OpRegistrationTbe *OpRegistrationTbe::Instance() {
  static OpRegistrationTbe instance;
  return &instance;
}

bool OpRegistrationTbe::Finalize(const OpRegistrationData &reg_data, bool is_train) {
  static std::map<domi::FrameworkType, std::map<std::string, std::string> *> op_map = {{CAFFE, &caffe_op_map}};
  if (is_train) {
    op_map[domi::TENSORFLOW] = &tensorflow_train_op_map;
  } else {
    op_map[domi::TENSORFLOW] = &tensorflow_op_map;
  }

  if (op_map.find(reg_data.GetFrameworkType()) != op_map.end()) {
    std::map<std::string, std::string> *fmk_op_map = op_map[reg_data.GetFrameworkType()];
    auto ori_optype_set = reg_data.GetOriginOpTypeSet();
    for (auto &tmp : ori_optype_set) {
      if ((*fmk_op_map).find(tmp) != (*fmk_op_map).end()) {
        GELOGW("Op type does not need to be changed, om_optype:%s, orignal type:%s.", (*fmk_op_map)[tmp].c_str(),
               tmp.c_str());
        continue;
      } else {
        (*fmk_op_map)[tmp] = reg_data.GetOmOptype();
        GELOGD("First register in parser initialize, original type: %s, om_optype: %s, imply type: %s.", tmp.c_str(),
               reg_data.GetOmOptype().c_str(), TypeUtils::ImplyTypeToSerialString(reg_data.GetImplyType()).c_str());
      }
    }
  }

  bool ret = RegisterParser(reg_data);
  return ret;
}

bool OpRegistrationTbe::RegisterParser(const OpRegistrationData &reg_data) {
  if (reg_data.GetFrameworkType() == domi::TENSORFLOW) {
    std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
    if (factory == nullptr) {
      REPORT_CALL_ERROR("E19999", "Get OpParserFactory failed.");
      GELOGE(INTERNAL_ERROR, "[Get][OpParserFactory] for tf failed.");
      return false;
    }
    if (reg_data.GetParseParamFn() != nullptr || reg_data.GetParseParamByOperatorFn() != nullptr) {
      bool is_registed = factory->OpParserIsRegistered(reg_data.GetOmOptype());
      if (is_registed) {
        GELOGW("Parse param func has already register for op:%s.", reg_data.GetOmOptype().c_str());
        return false;
      }
      std::shared_ptr<TensorFlowCustomParserAdapter> tf_parser_adapter =
          ge::parser::MakeShared<TensorFlowCustomParserAdapter>();
      if (tf_parser_adapter == nullptr) {
        REPORT_CALL_ERROR("E19999", "Create TensorFlowCustomParserAdapter failed.");
        GELOGE(PARAM_INVALID, "[Create][TensorFlowCustomParserAdapter] failed.");
        return false;
      }
      OpParserRegisterar registerar __attribute__((unused)) = OpParserRegisterar(
          domi::TENSORFLOW, reg_data.GetOmOptype(), [=]() -> std::shared_ptr<OpParser> { return tf_parser_adapter; });
    }
    if (reg_data.GetFusionParseParamFn() != nullptr || reg_data.GetFusionParseParamByOpFn() != nullptr) {
      bool is_registed = factory->OpParserIsRegistered(reg_data.GetOmOptype(), true);
      if (is_registed) {
        GELOGW("Parse param func has already register for fusion op:%s.", reg_data.GetOmOptype().c_str());
        return false;
      }
      GELOGI("Register fusion custom op parser: %s", reg_data.GetOmOptype().c_str());
      std::shared_ptr<TensorFlowFusionCustomParserAdapter> tf_fusion_parser_adapter =
          ge::parser::MakeShared<TensorFlowFusionCustomParserAdapter>();
      if (tf_fusion_parser_adapter == nullptr) {
        REPORT_CALL_ERROR("E19999", "Create TensorFlowFusionCustomParserAdapter failed.");
        GELOGE(PARAM_INVALID, "[Create][TensorFlowFusionCustomParserAdapter] failed.");
        return false;
      }
      OpParserRegisterar registerar __attribute__((unused)) = OpParserRegisterar(
          domi::TENSORFLOW, reg_data.GetOmOptype(),
          [=]() -> std::shared_ptr<OpParser> { return tf_fusion_parser_adapter; }, true);
    }
  } else {
    std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(reg_data.GetFrameworkType());
    if (factory == nullptr) {
      REPORT_CALL_ERROR("E19999", "Get OpParserFactory for %s failed.",
                        TypeUtils::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      GELOGE(INTERNAL_ERROR, "[Get][OpParserFactory] for %s failed.",
             TypeUtils::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      return false;
    }
    bool is_registed = factory->OpParserIsRegistered(reg_data.GetOmOptype());
    if (is_registed) {
      GELOGW("Parse param func has already register for op:%s.", reg_data.GetOmOptype().c_str());
      return false;
    }

    PARSER_CREATOR_FN func = CustomParserAdapterRegistry::Instance()->GetCreateFunc(reg_data.GetFrameworkType());
    if (func == nullptr) {
      REPORT_CALL_ERROR("E19999", "Get custom parser adapter failed for fmk type %s.",
                        TypeUtils::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      GELOGE(INTERNAL_ERROR, "[Get][CustomParserAdapter] failed for fmk type %s.",
             TypeUtils::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      return false;
    }
    OpParserFactory::Instance(reg_data.GetFrameworkType())->RegisterCreator(reg_data.GetOmOptype(), func);
    GELOGD("Register custom parser adapter for op %s of fmk type %s success.", reg_data.GetOmOptype().c_str(),
           TypeUtils::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
  }
  return true;
}
}  // namespace ge
