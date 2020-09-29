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

#ifndef DOMI_COMMON_OP_DEF_IR_PB_CONVERTER_H
#define DOMI_COMMON_OP_DEF_IR_PB_CONVERTER_H

#include "framework/common/fmk_error_codes.h"
#include "common/op_def/op_schema.h"
#include "parser/common/op_def/operator.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "proto/om.pb.h"

namespace ge {
domi::Status ConvertToOpDesc(const ParserOperator &op, ge::OpDescPtr op_def);

domi::Status ConvertFromOpDesc(const ge::OpDescPtr op_def, ParserOperator &op);
}  // namespace ge

#endif  // DOMI_COMMON_OP_DEF_IR_PB_CONVERTER_H
