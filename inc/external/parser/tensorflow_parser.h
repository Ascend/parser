/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020~2022. All rights reserved.
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

#ifndef INC_EXTERNAL_ACL_PARSER_TENSORFLOW_H_
#define INC_EXTERNAL_ACL_PARSER_TENSORFLOW_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"

namespace ge {
PARSER_FUNC_VISIBILITY graphStatus aclgrphParseTensorFlow(const char *model_file, ge::Graph &graph);
PARSER_FUNC_VISIBILITY graphStatus aclgrphParseTensorFlow(const char *model_file,
    const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph);
}  // namespace ge

#endif  // INC_EXTERNAL_ACL_PARSER_TENSORFLOW_H_