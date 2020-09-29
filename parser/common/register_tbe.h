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

#ifndef PARSER_COMMON_REGISTER_TBE_H_
#define PARSER_COMMON_REGISTER_TBE_H_

#include "register/op_registry.h"

namespace ge {
class OpRegistrationTbe {
 public:
  static OpRegistrationTbe *Instance();

  bool Finalize(const OpRegistrationData &reg_data, bool is_train = false);

 private:
  bool RegisterParser(const OpRegistrationData &reg_data);
};
}  // namespace ge

#endif  // PARSER_COMMON_REGISTER_TBE_H_