/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef PARSER_COMMON_AUTO_MAPPING_SUBGRAPH_IO_INDEX_FUNC_H_
#define PARSER_COMMON_AUTO_MAPPING_SUBGRAPH_IO_INDEX_FUNC_H_

#include <functional>
#include "external/graph/graph.h"
#include "external/register/register_error_codes.h"

namespace ge {
domi::Status AutoMappingSubgraphIndexByDataNodeAndOutputNodesInfo(
    const ge::Graph &graph,
    const std::function<domi::Status(int data_index, int &parent_input_index)> &input,
    const std::function<domi::Status(int netoutput_index, int &parent_output_index)> &output);
} // namespace ge
#endif  // PARSER_COMMON_AUTO_MAPPING_SUBGRAPH_IO_INDEX_FUNC_H_
