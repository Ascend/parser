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

#ifndef GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_H_
#define GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_H_

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

#include <map>
#include <vector>
#include "proto/onnx/ge_onnx.pb.h"
#include "external/register/register_error_codes.h"
#include "framework/omg/parser/parser_types.h"
#include "parser/onnx/onnx_util.h"

using Status = domi::Status;
using namespace ge::parser;

namespace ge {
class PARSER_FUNC_VISIBILITY SubgraphAdapter {
 public:
  /// @brief parse params
  /// @param [in/out] parent_op               parent op
  /// @param [in/out] onnx_graph_tasks        onnx graph task
  /// @param [in/out] name_to_onnx_graph      map name to onnx graph
  /// @return SUCCESS                         parse success
  /// @return FAILED                          Parse failed
  virtual Status AdaptAndFindAllSubgraphs(ge::onnx::NodeProto *parent_op,
                                          std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                          std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph) {
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_H_
