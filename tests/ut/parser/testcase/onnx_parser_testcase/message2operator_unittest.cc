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

#include "common/convert/message2operator.h"

#include <gtest/gtest.h>

#include "proto/onnx/ge_onnx.pb.h"

namespace ge {
class UtestMessage2Operator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestMessage2Operator, message_to_operator_success) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  attribute->set_f(1.0);
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMessage2Operator, message_to_operator_fail) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->add_double_data(1.00);

  ge::Operator op_src("add", "Add");
  auto ret = Message2Operator::ParseOperatorAttrs(attribute, 6, op_src);
  EXPECT_EQ(ret, FAILED);

  ret = Message2Operator::ParseOperatorAttrs(attribute, 1, op_src);
  EXPECT_EQ(ret, FAILED);
}
}  // namespace ge