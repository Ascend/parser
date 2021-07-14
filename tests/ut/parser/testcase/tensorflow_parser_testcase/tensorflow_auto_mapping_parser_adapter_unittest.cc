/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include <gtest/gtest.h>
#include <iostream>
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_auto_mapping_parser_adapter.h"
#include "framework/omg/parser/parser_factory.h"
#include "graph/operator_reg.h"
#include "external/graph/types.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"


namespace ge {
class UtestTensorflowAutoMappingParserAdapter : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

};


TEST_F(UtestTensorflowAutoMappingParserAdapter, success) {
  auto parser = TensorFlowAutoMappingParserAdapter();

  domi::tensorflow::NodeDef arg_node;
  arg_node.set_name("size");
  arg_node.set_op("Size");
  auto attr = arg_node.mutable_attr();
  domi::tensorflow::AttrValue value;
  value.set_type(domi::tensorflow::DataType::DT_HALF);
  (*attr)["out_type"] = value;

  auto op_desc = ge::parser::MakeShared<ge::OpDesc>("size", "Size");
  auto ret = parser.ParseParams(reinterpret_cast<Message *>(&arg_node), op_desc);
  EXPECT_EQ(ret, ge::SUCCESS);


  auto ret2 = ge::AttrUtils::SetBool(op_desc, "test_fail", true);
  EXPECT_EQ(ret2, true);
  EXPECT_EQ(ge::AttrUtils::HasAttr(op_desc, "test_fail"), true);

  ret = parser.ParseParams(reinterpret_cast<Message *>(&arg_node), op_desc);
  EXPECT_EQ(ret, ge::FAILED);
}


} // namespace ge