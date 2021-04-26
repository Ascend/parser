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

#include "parser/common/op_def/fill_op.h"
#include "parser/tensorflow/tensorflow_parser_register.h"
#include "framework/omg/parser/parser_types.h"

using ge::parser::ALPHA_DEFAULT_VALUE;
using ge::parser::BETA_DEFAULT_VALUE;
using ge::parser::FILL;

namespace ge {
/*
node {
    name: "model_with_buckets/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/zeros"
    op: "Fill"
    input: "model_with_buckets/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/concat"
    input: "model_with_buckets/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/zeros/Const"
    device: "/device:GPU:2"
    attr {
        key: "T"
        value {
            type: DT_FLOAT
        }
    }
}
*/
domi::Status ParseParams(const NodeDef *node, FillOperator *op) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(op);
  op->Name(node->name());

  domi::tensorflow::DataType data_type;
  GE_RETURN_IF_ERROR(TensorFlowUtil::ParseDataType(node, TENSORFLOW_ATTR_T, data_type));
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(data_type);
  CHECK_FALSE_EXEC(
      type != ge::DataType::DT_UNDEFINED,
      REPORT_CALL_ERROR("E19999", "Data type %s of node %s is not supported",
                        DataType_Name(data_type).c_str(),
                        node->name().c_str());
      GELOGE(PARAM_INVALID, "Data type %s of node %s is not supported.", DataType_Name(data_type).c_str(),
          node->name().c_str());
      return PARAM_INVALID);

  op->DataType(type);

  op->Alpha(ge::parser::ALPHA_DEFAULT_VALUE);
  op->Beta(ge::parser::BETA_DEFAULT_VALUE);

  return domi::SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(FILL, FillOperator).SetParseParamsFn(ParseParams);
}  // namespace ge
