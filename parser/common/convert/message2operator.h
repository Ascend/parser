/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef PARSER_MESSAGE2OPERATOR_H
#define PARSER_MESSAGE2OPERATOR_H

#include "external/ge/ge_api_error_codes.h"
#include "external/graph/operator.h"
#include "google/protobuf/message.h"

namespace ge {
class Message2Operator {
 public:
  static Status ParseOperatorAttrs(const google::protobuf::Message *message, int depth, ge::Operator &ops);

 private:
  static Status ParseField(const google::protobuf::Reflection *reflection, const google::protobuf::Message *message,
                           const google::protobuf::FieldDescriptor *field, int depth, ge::Operator &ops);

  static Status ParseRepeatedField(const google::protobuf::Reflection *reflection,
                                   const google::protobuf::Message *message,
                                   const google::protobuf::FieldDescriptor *field, int depth, ge::Operator &ops);
};
}  // namespace ge
#endif  // PARSER_MESSAGE2OPERATOR_H
