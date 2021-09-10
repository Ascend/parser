#include <gtest/gtest.h>
#include <iostream>
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_builder_utils.h"
#include "common/util.h"

#include "tensorflow/iterator_fusion_pass.h"
#include "parser/common/acl_graph_parser_util.h"
#define private public
#include "tensorflow/graph_optimizer.h"
#undef private

namespace ge {
class UtestGraphOptimizer : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
  ComputeGraphPtr MakeGraph() {
  ge::ut::GraphBuilder builder("graph");
  std::string name = "graph";
  std::string original_type ;

  original_type = "IteratorV2";
  auto data1 = builder.AddNode(name + "_"+original_type, ge::parser::FRAMEWORKOP, 1, 1);
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);

  original_type = "IteratorGetNext";
  auto data2 = builder.AddNode(name + "_"+original_type+"2", ge::parser::FRAMEWORKOP, 1, 2); 
  ge::AttrUtils::SetStr(data2->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);

  string nodefStr;
  AttrUtils::SetZeroCopyBytes(
  data2->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
  Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(nodefStr.data()), nodefStr.length()));

  original_type = "IteratorGetNext";
  auto data3 = builder.AddNode(name + "_"+original_type+"3", ge::parser::FRAMEWORKOP, 2, 1);
  ge::AttrUtils::SetStr(data3->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type); 

  AttrUtils::SetZeroCopyBytes(
  data3->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
  Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(nodefStr.data()), nodefStr.length()));
  
  builder.AddDataEdge(data1, 0, data2, 0);
  builder.AddDataEdge(data2, 0, data3, 0);
  builder.AddDataEdge(data2, 1, data3, 1);
  return builder.GetGraph();
  }
}

TEST_F(UtestGraphOptimizer, graph_optimizer) {
  ge::ComputeGraphPtr graph = MakeGraph();
  ge::IteratorFusionPass iteratorFusionPass(domi::TENSORFLOW);
  EXPECT_NE(iteratorFusionPass.Run(graph),ge::SUCCESS);
}

TEST_F(UtestGraphOptimizer, graph_optimizer_output) {
  ge::ComputeGraphPtr graph = MakeGraph();
  domi::FrameworkType type = domi::TENSORFLOW;
  ge::ParserGraphOptimizer parserGraphOptimizer(graph,type);
  vector<ge::InDataAnchorPtr> input_anchors;
  vector<ge::OutDataAnchorPtr> output_anchors;
  ge::OpDescPtr fusion_op_desc;
  EXPECT_NE(parserGraphOptimizer.RebuildInputAnchors(input_anchors,fusion_op_desc),ge::SUCCESS);
  EXPECT_NE(parserGraphOptimizer.RebuildOutputAnchors(output_anchors,fusion_op_desc),ge::SUCCESS);
}
}
