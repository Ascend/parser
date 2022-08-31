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

#ifndef INC_EXTERNAL_ACL_GRAPH_CAFFE_H_
#define INC_EXTERNAL_ACL_GRAPH_CAFFE_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"
#include "parser_common.h"

namespace ge {
PARSER_FUNC_VISIBILITY graphStatus aclgrphParseCaffe(const char *model_file, const char *weights_file,
                                                     ge::Graph &graph);

PARSER_FUNC_VISIBILITY graphStatus aclgrphParseCaffe(const char *model_file, const char *weights_file,
                                                     const std::map<ge::AscendString, ge::AscendString> &parser_params,
                                                     ge::Graph &graph);
}  // namespace ge

#endif  // INC_EXTERNAL_ACL_GRAPH_CAFFE_H_
