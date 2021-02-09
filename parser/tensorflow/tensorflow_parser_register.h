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

// Copyright (c) <2018>, <Huawei Technologies Co., Ltd>
#ifndef PARSER_TENSORFLOW_TENSORFLOW_PARSER_REGISTER_H_
#define PARSER_TENSORFLOW_TENSORFLOW_PARSER_REGISTER_H_

#include <functional>
#include <memory>
#include <string>
#include "common/util.h"
#include "framework/omg/parser/op_parser.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/op_def/operator.h"
#include "parser/common/acl_graph_parser_util.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "proto/tensorflow/node_def.pb.h"

using domi::tensorflow::NodeDef;

namespace ge {
class PARSER_FUNC_VISIBILITY TensorflowFinalizeable {
 public:
  virtual bool Finalize() = 0;
  virtual ~TensorflowFinalizeable() {}
};

class PARSER_FUNC_VISIBILITY TensorflowReceiver {
 public:
  TensorflowReceiver(TensorflowFinalizeable &f) { f.Finalize(); }
  ~TensorflowReceiver() {}
};

namespace tensorflow_parser {
template <typename Param>
class TensorflowParserBuilder;

class PARSER_FUNC_VISIBILITY TensorflowWeightParserBuilder : public TensorflowFinalizeable {
 public:
  virtual ~TensorflowWeightParserBuilder() {}
};

template <typename Param>
class TensorflowOpParserAdapter;

template <typename Param>
class PARSER_FUNC_VISIBILITY TensorflowParserBuilder : public TensorflowWeightParserBuilder {
 public:
  using ParseParamsFn = std::function<domi::Status(const domi::tensorflow::NodeDef *, Param *)>;

  explicit TensorflowParserBuilder(const std::string &davinci_optype) : davinci_optype_(davinci_optype) {}

  ~TensorflowParserBuilder() {}

  TensorflowParserBuilder &SetParseParamsFn(ParseParamsFn parse_params_fn) {
    parse_params_fn_ = parse_params_fn;
    return *this;
  }

  bool Finalize() override {
    auto op_parser_adapter = ge::parser::MakeShared<TensorflowOpParserAdapter<Param>>(*this);
    if (op_parser_adapter == nullptr) {
      GELOGE(FAILED, "Op parser adapter is null.");
    }
    // register to OpParserFactory
    OpParserRegisterar registerar __attribute__((unused)) = OpParserRegisterar(
      domi::TENSORFLOW, davinci_optype_, [=] { return std::shared_ptr<OpParser>(op_parser_adapter); });
    return true;
  }

 private:
  std::string davinci_optype_;  // op type in davinci model

  ParseParamsFn parse_params_fn_;

  friend class TensorflowOpParserAdapter<Param>;
};

template <typename Param>
class PARSER_FUNC_VISIBILITY TensorflowOpParserAdapter : public TensorFlowOpParser {
  using ParseParamsFn = std::function<domi::Status(const domi::tensorflow::NodeDef *, Param *)>;

 public:
  TensorflowOpParserAdapter(TensorflowParserBuilder<Param> builder) { parse_params_fn_ = builder.parse_params_fn_; }

  ~TensorflowOpParserAdapter() {}

  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override {
    const domi::tensorflow::NodeDef *node = static_cast<const domi::tensorflow::NodeDef *>(op_src);
    GE_CHECK_NOTNULL(node);
    std::shared_ptr<Param> param = ge::parser::MakeShared<Param>();
    if (param == nullptr) {
      GELOGE(domi::FAILED, "Param is null");
      return domi::FAILED;
    }
    GE_RETURN_IF_ERROR(parse_params_fn_(node, param.get()));
    param.get()->Name(node->name());
    std::shared_ptr<ParserOperator> op_param = std::static_pointer_cast<ParserOperator>(param);
    ConvertToOpDesc(*op_param, op_dest);

    return domi::SUCCESS;
  }

 private:
  ParseParamsFn parse_params_fn_;
};
}  // namespace tensorflow_parser

#define DOMI_REGISTER_TENSORFLOW_PARSER(name, param_clazz) \
  DOMI_REGISTER_TENSORFLOW_PARSER_UNIQ_HELPER(__COUNTER__, name, param_clazz)
#define DOMI_REGISTER_TENSORFLOW_PARSER_UNIQ_HELPER(ctr, name, param_clazz) \
  DOMI_REGISTER_TENSORFLOW_PARSER_UNIQ(ctr, name, param_clazz)
#define DOMI_REGISTER_TENSORFLOW_PARSER_UNIQ(ctr, name, param_clazz)                  \
  static TensorflowReceiver register_tensorflow_parser##ctr __attribute__((unused)) = \
    tensorflow_parser::TensorflowParserBuilder<param_clazz>(name)
}  // namespace ge

#endif  // PARSER_TENSORFLOW_TENSORFLOW_PARSER_REGISTER_H_
