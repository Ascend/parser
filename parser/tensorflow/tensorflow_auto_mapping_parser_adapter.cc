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

#include "tensorflow_auto_mapping_parser_adapter.h"

#include "framework/omg/parser/parser_types.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"
#include "register/register.h"


using domi::TENSORFLOW;
using namespace ge::parser;

using ge::parser::PLACEHOLDERWITHDEFAULT;

namespace ge {
namespace {
const char *const kTfAttrT = "T";
}  // namespace

Status TensorFlowAutoMappingParserAdapter::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  if (op_src == nullptr) {
    GELOGE(PARAM_INVALID, "Op src is null");
    return PARAM_INVALID;
  }
  const NodeDef *node = reinterpret_cast<const NodeDef *>(op_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  if (op_dest == nullptr) {
    GELOGE(FAILED, "Op dest is null");
    return PARAM_INVALID;
  }

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  Status ret = domi::AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Tensorflow auto mapping parser params failed");
    return FAILED;
  }
  op.BreakConnect();

  // add dynamic input/output
  if (op_dest->GetType() == IDENTITYN) {
    uint32_t dynamic_tensor_num = 0;
    domi::tensorflow::AttrValue attr_num;
    if (!(TensorFlowUtil::FindAttrValue(node, kTfAttrT, attr_num))) {
      GELOGW("In NodeDef %s dynamic attr [%s] is not exist.", op_dest->GetName().c_str(), kTfAttrT);
    }
    dynamic_tensor_num = attr_num.list().type_size();

    GE_CHK_STATUS_RET(op_dest->AddDynamicInputDesc("x", dynamic_tensor_num), "AddDynamicInputDesc failed");
    GE_CHK_STATUS_RET(op_dest->AddDynamicOutputDesc("y", dynamic_tensor_num), "AddDynamicInputDesc failed");
    GELOGI("add dynamic intput and output for op [%s], type[%s], number:%u", op_dest->GetName().c_str(),
           op_dest->GetType().c_str(), dynamic_tensor_num);
  }

  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, PLACEHOLDERWITHDEFAULT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, EXPANDDIMS, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, SIZE, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, SHAPE, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, GUARANTEECONST, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, BROADCASTARGS, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, PREVENTGRADIENT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, RANK, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, BROADCASTGRADIENTARGS, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, STOPGRADIENT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, DESTROYTEMPORARYVARIABLE, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, SNAPSHOT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, EMPTY, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, IDENTITYN, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, CONTROLTRIGGER, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, SWITCH, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, LOOPCOND, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, NEXTITERATION, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, REFNEXTITERATION, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, EXIT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, REFEXIT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, CONSTANT, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, PARALLELCONCATSTART, TensorFlowAutoMappingParserAdapter);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, BITCAST, TensorFlowAutoMappingParserAdapter);
}
