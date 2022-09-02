/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef DOMI_COMMON_OP_OPERATOR_H
#define DOMI_COMMON_OP_OPERATOR_H

#include <string>
#include <unordered_map>
#include <vector>
#include "framework/common/fmk_types.h"
#include "common/tuple.h"
#include "graph/ge_tensor.h"
#include "proto/om.pb.h"
namespace ge {
struct OpAttribute {
  OpAttribute(const std::string &name, const domi::AttrDef &value) : name_(name), value_(value) {}
  const std::string name_;
  domi::AttrDef value_;
};

class FMK_FUNC_HOST_VISIBILITY ParserOperator {
 public:
  explicit ParserOperator(const std::string &type);
  ParserOperator() {}

  virtual ~ParserOperator() = default;

  ParserOperator &Input(const ParserOperator &in_op, uint32_t index = 0);

  ParserOperator &Attr(const OpAttribute &op_attr);

  ParserOperator &AttrVector(std::string key, std::vector<int32_t> &value);
  ParserOperator &AttrVector(std::string key, std::vector<int64_t> &value);

  virtual ParserOperator &Name(const std::string &name);

  ParserOperator &Type(const std::string &type);

  ParserOperator &InputTensorDesc(const ge::GeTensorDesc &input_tensordesc);

  ParserOperator &OutputTensorDesc(const ge::GeTensorDesc &output_tensordesc);

  ParserOperator &Attr_bt(const std::string &name, const std::string &value);

 // Register "optional" attribute with default value.
  ParserOperator &Attr(const std::string &name, const uint32_t &value);
  ParserOperator &Attr(const std::string &name, const std::vector<uint32_t> &value);
  ParserOperator &Attr(const std::string &name, const ge::Tuple<uint32_t> &value);

  ParserOperator &Attr(const std::string &name, const int64_t &value);
  ParserOperator &Attr(const std::string &name, const std::vector<int64_t> &value);
  ParserOperator &Attr(const std::string &name, const ge::Tuple<int64_t> &value);

  ParserOperator &Attr(const std::string &name, const bool &value);
  ParserOperator &Attr(const std::string &name, const std::vector<bool> &value);
  ParserOperator &Attr(const std::string &name, const ge::Tuple<bool> &value);

  ParserOperator &Attr(const std::string &name, const float &value);
  ParserOperator &Attr(const std::string &name, const std::vector<float> &value);
  ParserOperator &Attr(const std::string &name, const ge::Tuple<float> &value);

  ParserOperator &Attr(const std::string &name, const std::string &value);
  ParserOperator &Attr(const std::string &name, const std::vector<std::string> &value);
  ParserOperator &Attr(const std::string &name, const ge::Tuple<std::string> &value);

  const std::string &GetName() const { return name_; }

  const std::string &GetType() const { return type_; }

  const std::vector<std::string> &GetInputs() const { return inputs_; }

  const std::vector<ge::GeTensorDesc> &GetInputTensorDesc() const { return input_descs_; }

  const std::vector<ge::GeTensorDesc> &GetOutputTensorDesc() const { return output_descs_; }

  const std::unordered_map<std::string, OpAttribute> GetOpAttrs() const { return op_attrs_; }

  bool HasAttr(const std::string &name) const { return op_attrs_.find(name) != op_attrs_.end(); }

  int64_t GetIntAttr(const std::string &name) const;

  uint32_t GetUintAttr(const std::string &name) const;

  float GetFloatAttr(const std::string &name) const;

  bool GetBoolAttr(const std::string &name) const;

  std::string GetStringAttr(const std::string &name) const;

  ge::IntTuple GetIntTupleAttr(const std::string &name) const;

  ge::UintTuple GetUintTupleAttr(const std::string &name) const;

  ge::FloatTuple GetFloatTupleAttr(const std::string &name) const;

  ge::BoolTuple GetBoolTupleAttr(const std::string &name) const;

  ge::StringTuple GetStringTupleAttr(const std::string &name) const;

 private:
  std::string name_;
  std::string type_;
  std::vector<std::string> inputs_;
  std::unordered_map<std::string, OpAttribute> op_attrs_;
  std::vector<ge::GeTensorDesc> input_descs_;
  std::vector<ge::GeTensorDesc> output_descs_;
};
}  // namespace domi
#endif  // DOMI_COMMON_OP_OPERATOR_H
