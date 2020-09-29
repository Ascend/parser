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

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "framework/omg/parser/parser_types.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "omg/omg_inner_types.h"

using std::map;
using std::string;
using std::unordered_map;
using std::vector;

namespace ge {
class ParserGraphOptimizer {
 public:
  explicit ParserGraphOptimizer(ge::ComputeGraphPtr graph, domi::FrameworkType type = domi::TENSORFLOW)
      : graph_(graph), fmktype_(type), local_fmk_op_flag_(false) {}

  ~ParserGraphOptimizer() {}

  domi::Status Optimize();

  domi::Status OptimizeAfterCal();

  domi::Status FusionFmkop();

  inline bool IsHCOMOp(const string &op_type) {
    return (op_type == ge::parser::HCOMALLREDUCE) || (op_type == ge::parser::HCOMALLGATHER) ||
           (op_type == ge::parser::HCOMBROADCAST) || (op_type == ge::parser::HCOMSEND) ||
           (op_type == ge::parser::HCOMRECEIVE) || (op_type == "HcomReduceScatter");
  }

  void SetLocalFmkopFlag(bool isLocalFmkopFlag) { local_fmk_op_flag_ = isLocalFmkopFlag; }

  const bool GetLocalFmkopFlag() const { return local_fmk_op_flag_; }

  void SetFuncBinPath(std::string isFuncBinPath) { func_bin_path_ = isFuncBinPath; }
  const std::string GetFuncBinPath() const { return func_bin_path_; }

  domi::Status InsertHWCK2FZ(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor,
                             enum ge::Format srcOutFormat, enum ge::DataType srcOutDatatype,
                             enum ge::Format dstInFormat, enum ge::DataType dstInDatatype);

  domi::Status Insert4DTo5DTransOp(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor,
                                   enum ge::Format src_out_format, enum ge::DataType src_out_data_type,
                                   enum ge::Format dst_in_format, enum ge::DataType dst_in_data_type);

  domi::Status InsertFZ2HWCK(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor,
                             enum ge::Format srcOutFormat, enum ge::DataType srcOutDatatype,
                             enum ge::Format dstInFormat, enum ge::DataType dstInDatatype);

  domi::Status Insert5DTo4DTransOp(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor,
                                   enum ge::Format src_out_format, enum ge::DataType src_out_data_type,
                                   enum ge::Format dst_in_format, enum ge::DataType dst_in_data_type);

  ge::OpDescPtr CreateCastOp(enum ge::DataType input_datatype, enum ge::DataType output_datatype, ge::Format format);

  ge::OpDescPtr CreatePermuteOp(enum ge::Format input_format, enum ge::Format output_format);

  ge::OpDescPtr CreateTransDataOp(enum ge::Format input_format);

  domi::Status NewNodeAddEdges(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor, ge::NodePtr first,
                               ge::NodePtr second, ge::NodePtr third);

  domi::Status InsertVar5DTo4D(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor,
                               enum ge::Format srcOutFormat, enum ge::DataType srcOutDatatype,
                               enum ge::Format dstInFormat, enum ge::DataType dstInDatatype);

  ge::OpDescPtr CreateTranslateOp(enum ge::Format inFormat, ge::DataType inDatatype, enum ge::Format outFormat,
                                  ge::DataType outDatatype);

 private:
  ge::ComputeGraphPtr graph_;
  domi::FrameworkType fmktype_;
  // local fmkop flag
  bool local_fmk_op_flag_;
  std::string func_bin_path_;

  domi::Status FindFmkNodeCluser(unordered_map<string, vector<ge::NodePtr>> &node_cluser_Map);

  domi::Status MarkForFusion(unordered_map<string, vector<ge::NodePtr>> &node_cluser_Map);

  domi::Status UpdateGraph(vector<ge::NodePtr> &nodes);

  domi::Status InsertNode(ge::ComputeGraphPtr sub_graph, vector<ge::NodePtr> &nodes,
                          vector<ge::InDataAnchorPtr> &input_anchors, vector<ge::OutDataAnchorPtr> &output_anchors,
                          map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                          vector<ge::InControlAnchorPtr> &input_control_anchors,
                          vector<ge::OutControlAnchorPtr> &output_control_anchors,
                          unordered_map<string, ge::NodePtr> &node_map);

  domi::Status LinkInnerAnchor(unordered_map<string, ge::NodePtr> &node_map);

  domi::Status RebuildOutputAnchors(vector<ge::OutDataAnchorPtr> &output_anchors, ge::OpDescPtr fusion_op_desc);

  domi::Status RebuildInputAnchors(vector<ge::InDataAnchorPtr> &input_anchors, ge::OpDescPtr fusion_op_desc);

  domi::Status RebuildFusionNode(vector<ge::InDataAnchorPtr> &input_anchors,
                                 vector<ge::OutDataAnchorPtr> &output_anchors,
                                 map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                                 vector<ge::InControlAnchorPtr> &input_control_anchors,
                                 vector<ge::OutControlAnchorPtr> &output_control_anchors, ge::NodePtr fusion_node);

  domi::Status MakeTfProtoDef();
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
