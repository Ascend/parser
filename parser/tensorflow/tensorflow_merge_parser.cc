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

#include "parser/tensorflow/tensorflow_merge_parser.h"

#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"

using domi::TENSORFLOW;
using ge::parser::MERGE;

namespace ge {
Status TensorFlowMergeParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op_desc);

  const NodeDef *node = reinterpret_cast<const NodeDef *>(op_src);
  domi::tensorflow::AttrValue attr_num;
  if (!(TensorFlowUtil::FindAttrValue(node, ATTR_NAME_N, attr_num))) {
    GELOGW("In NodeDef %s dynamic attr [%s] is not exist.", op_desc->GetName().c_str(), ATTR_NAME_N.c_str());
  }
  int32_t input_tensor_num = attr_num.i();

  // add dynamic input
  graphStatus ret = op_desc->AddDynamicInputDesc("x", input_tensor_num);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add Dynamic InputDesc name:x to node:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "Add dynamic input:x for node:%s failed.", op_desc->GetName().c_str());
    return FAILED;
  }
  GELOGI("add dynamic input for Merge op [%s], num:%d", op_desc->GetName().c_str(), input_tensor_num);

  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, MERGE, TensorFlowMergeParser);
}
