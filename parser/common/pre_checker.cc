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

#include "parser/common/pre_checker.h"
#include <nlohmann/json.hpp>
#include "common/model_saver.h"
#include "common/op_map.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/model_saver.h"
#include "omg/parser/parser_inner_ctx.h"
#include "register/op_registry.h"

namespace ge {
// Keys in JSON file
namespace {
const char *const kKeyName = "name";
const char *const kKeyResult = "result";
const char *const kKeyTotal = "total";
const char *const kKeyPass = "pass";
const char *const kKeyFail = "fail";
const char *const kKeyOp = "op";
const char *const kKeyOpName = "name";
const char *const kKeyOpType = "type";
const char *const kKeyOpResult = "result";
const char *const kKeyCause = "cause";
const char *const kKeyCauseCode = "code";
const char *const kKeyCauseMessage = "message";

// Checking result and support warning later
const char *const kResultSuccess = "success";
const char *const kResultFailed = "failed";
}  // namespace

PreChecker::PreChecker() : fmk_op_types_(nullptr) { Init(); }

void PreChecker::Init() {
  model_name_.clear();
  op_map_.clear();
  ops_.clear();
  fmk_op_types_ = nullptr;

  // Currently only Caffe and tensorflow are supported
  domi::FrameworkType fmk_type = GetParserContext().type;
  if (fmk_type == domi::CAFFE)
    fmk_op_types_ = &caffe_op_map;
  else if (fmk_type == domi::TENSORFLOW)
    fmk_op_types_ = &tensorflow_op_map;
  else
    return;
}

PreChecker::~PreChecker() {}

FMK_FUNC_HOST_VISIBILITY PreChecker &PreChecker::Instance() {
  static PreChecker instance;
  return instance;
}

FMK_FUNC_HOST_VISIBILITY void PreChecker::SetModelName(const string &name) { model_name_ = name; }

FMK_FUNC_HOST_VISIBILITY Status PreChecker::AddOp(OpId id, const string &name, const string &type) {
  GE_RETURN_WITH_LOG_IF_TRUE(op_map_.find(id) != op_map_.end(),
                             "[Check][Param] Id already exists, name:%s, type:%s.", name.c_str(), type.c_str());

  Info info;
  info.id = id;
  info.name = name;
  info.type = type;
  op_map_[id] = info;
  ops_.push_back(id);

  return SUCCESS;
}

Status PreChecker::CheckName(OpId id) {
  auto iter = op_map_.find(id);
  GE_RETURN_WITH_LOG_IF_TRUE(iter == op_map_.end(), "[Check][Param] Id does not exist.");

  Info &info = iter->second;
  for (auto &v : op_map_) {
    // If the name is duplicate, an error is logged
    if (id != v.first && info.name == v.second.name) {
      Cause cause;
      cause.code = NAME_REPEATED;
      cause.message = "The name is repeated.";

      GELOGI("Name %s repeated.", info.name.c_str());
      ErrorManager::GetInstance().ATCReportErrMessage("E19009", {"opname"}, {info.name});
      GE_RETURN_WITH_LOG_IF_ERROR(AddCause(id, cause), "[Add][Cause] failed.");
      GE_RETURN_WITH_LOG_IF_ERROR(AddCause(v.first, cause), "[Add][Cause] failed.");
      break;
    }
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY Status PreChecker::CheckType(OpId id, bool is_tensorflow) {
  auto iter = op_map_.find(id);
  GE_RETURN_WITH_LOG_IF_TRUE(iter == op_map_.end(), "[Check][Param] Id does not exist.");

  Info &info = iter->second;
  string type = info.type;

  // If the user explicitly specifies the mapping relationship of the operator type through
  // the -- OP_name_map parameter, the type specified by the user is used.
  auto op_map_iter = GetParserContext().op_conf_map.find(type);
  if (op_map_iter != GetParserContext().op_conf_map.end()) {
    type = op_map_iter->second;
  }

  // Judge whether the type is supported
  GE_RETURN_WITH_LOG_IF_ERROR(CheckTypeSupported(info.id, type, info.name, is_tensorflow),
                              "[Invoke][CheckTypeSupported] failed, type:%s, name:%s.",
                              type.c_str(), info.name.c_str());

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY Status PreChecker::AddCause(OpId id, ErrorCode code, const string &msg) {
  Cause cause;
  cause.code = code;
  cause.message = msg;
  return AddCause(id, cause);
}

FMK_FUNC_HOST_VISIBILITY void PreChecker::RefreshErrorMessageByName(const string &op_name, ErrorCode code,
                                                                    const string &msg) {
  for (const auto &op : op_map_) {
    if (op.second.name == op_name) {
      AddCause(op.second.id, code, msg);
      return;
    }
  }
  GELOGW("Node [%s] not founded in prechecking list.", op_name.c_str());
}

Status PreChecker::AddCause(OpId id, const Cause &cause) {
  auto iter = op_map_.find(id);
  GE_RETURN_WITH_LOG_IF_TRUE(iter == op_map_.end(), "[Check][Param] Id does not exist.");

  Info &info = iter->second;

  // Avoid adding repeatedly
  for (Cause &c : info.causes) {
    if (c.code == cause.code && c.message == cause.message) {
      return SUCCESS;
    }
  }

  info.causes.push_back(cause);

  return SUCCESS;
}

void PreChecker::Clear() { Init(); }

Status PreChecker::Clear(OpId id, const string &message) {
  auto iter = op_map_.find(id);
  GE_RETURN_WITH_LOG_IF_TRUE(iter == op_map_.end(), "[Check][Param] Id does not exist.");

  Info &info = iter->second;
  info.causes.clear();

  // Set additional information
  if (message != "") {
    Cause cause;
    cause.code = ErrorCode::OK;
    cause.message = message;
    GE_RETURN_WITH_LOG_IF_ERROR(AddCause(id, cause), "[Add][Cause] failed.");
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY bool PreChecker::HasError() {
  for (auto id : ops_) {
    if (HasError(id)) {
      return true;
    }
  }

  return false;
}

Status PreChecker::Save(string file) {
  uint32_t fail_num = 0;
  for (auto id : ops_) {
    if (HasError(id)) {
      fail_num++;
    }
  }

  // Initialization model related JSON information
  nlohmann::json model;
  model[kKeyName] = model_name_;
  model[kKeyResult] = HasError() ? kResultFailed : kResultSuccess;
  model[kKeyTotal] = ops_.size();
  model[kKeyPass] = ops_.size() - fail_num;
  model[kKeyFail] = fail_num;

  // Constructing JSON information of operators in order of network
  for (auto id : ops_) {
    auto iter = op_map_.find(id);
    GE_CHK_BOOL_RET_STATUS(iter != op_map_.end(), FAILED, "[Check][Param] don't find this op.");
    Info &info = iter->second;

    // Initialization operator general information
    nlohmann::json op = {{kKeyOpName, info.name}, {kKeyOpType, info.type}};
    op[kKeyOpResult] = HasError(id) ? kResultFailed : kResultSuccess;

    // handle causes
    for (const Cause &cause : info.causes) {
      nlohmann::json cause_j = {{kKeyCauseCode, cause.code}, {kKeyCauseMessage, cause.message}};
      op[kKeyCause].push_back(cause_j);
    }

    model[kKeyOp].push_back(op);
  }

  // Save JSON data to a file
  GE_RETURN_WITH_LOG_IF_ERROR(ge::parser::ModelSaver::SaveJsonToFile(file.c_str(), model),
                              "[Invoke][SaveJsonToFile]Save failed, file:%s.", file.c_str());

  return SUCCESS;
}

Status PreChecker::CheckTypeSupported(OpId id, const string &type, const string &name, bool is_tensorflow) {
  // Currently only partial framework type checking is supported
  if (fmk_op_types_ == nullptr) {
    std::string op_type;
    if (!domi::OpRegistry::Instance()->GetOmTypeByOriOpType(type, op_type)) {
      Cause cause;
      cause.code = TYPE_UNSUPPORTED;
      cause.message = "The type is not supported.";
      GELOGI("Check op[%s]'s type[%s] failed, it is not supported.", name.c_str(), type.c_str());
      if (!is_tensorflow) {
        ErrorManager::GetInstance().ATCReportErrMessage("E19010", {"opname", "optype"}, {name, type});
      }
      GE_RETURN_WITH_LOG_IF_ERROR(AddCause(id, cause), "[Add][Cause] failed.");
    }
    return SUCCESS;
  }

  // Log error if type not found
  if (fmk_op_types_->find(type) == fmk_op_types_->end()) {
    Cause cause;
    cause.code = TYPE_UNSUPPORTED;
    cause.message = "The type is not supported.";

    GELOGI("Check op[%s]'s type[%s] failed, it is not supported.", name.c_str(), type.c_str());
    if (!is_tensorflow) {
      ErrorManager::GetInstance().ATCReportErrMessage("E19010", {"opname", "optype"}, {name, type});
    }
    GE_RETURN_WITH_LOG_IF_ERROR(AddCause(id, cause), "[Add][Cause] failed.");
  }

  return SUCCESS;
}

bool PreChecker::HasError(OpId id) {
  auto iter = op_map_.find(id);
  GE_RETURN_WITH_LOG_IF_TRUE(iter == op_map_.end(), "[Check][Param] Id does not exist.");

  Info &info = iter->second;
  for (const Cause &cause : info.causes) {
    if (cause.code != ErrorCode::OK) {
      return true;
    }
  }

  return false;
}
}  // namespace ge
