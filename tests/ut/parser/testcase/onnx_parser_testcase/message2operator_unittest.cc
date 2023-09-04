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

#include <fstream>
#include <gtest/gtest.h>

#include "proto/onnx/ge_onnx.pb.h"
#include "parser/common/convert/pb2json.h"

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

TEST_F(UtestMessage2Operator, pb2json_one_field_json) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json_depth_max) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(1);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true, 21);
}

TEST_F(UtestMessage2Operator, pb2json_one_field_json_type) {
  ge::onnx::NodeProto input_node;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("attribute");
  attribute->set_type(onnx::AttributeProto::AttributeType(1));
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  attribute_tensor->set_data_type(3);
  attribute_tensor->add_dims(4);
  attribute_tensor->set_raw_data("\007");
  Json json;
  ge::Pb2Json::Message2Json(input_node, std::set<std::string>{}, json, true);
}

TEST_F(UtestMessage2Operator, enum_to_json_success) {
  std::ifstream f("../tests/ut/parser/testcase/onnx_parser_testcase/om_json/enumjson1.json");
  Json json = Json::parse(f);
  ge::Pb2Json::EnumJson2Json(json);
  std::cout << json.dump(4) << std::endl;
}

}  // namespace ge