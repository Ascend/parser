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

#ifndef GE_PARSER_TESTS_UT_PARSER_H_
#define GE_PARSER_TESTS_UT_PARSER_H_

#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/compute_graph.h"

namespace ge {
struct MemBuffer {
  void *data;
  uint32_t size;
};

class ParerUTestsUtils {
 public:
  static void ClearParserInnerCtx();
  static MemBuffer* MemBufferFromFile(const char *path);
};
namespace ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace ut
}  // namespace ge

#endif  // GE_PARSER_TESTS_UT_PARSER_H_
