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

#include "common/op/ge_op_utils.h"
#include "common/op_def/ir_pb_converter.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"

#include "parser/tensorflow/tensorflow_identity_parser.h"

using domi::TENSORFLOW;
using ge::parser::IDENTITY;
using ge::parser::READVARIABLEOP;

namespace ge {
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, IDENTITY, TensorFlowIdentityParser);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, READVARIABLEOP, TensorFlowIdentityParser);
}  // namespace ge
