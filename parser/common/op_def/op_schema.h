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

#ifndef DOMI_COMMON_OP_SCHEMA_H
#define DOMI_COMMON_OP_SCHEMA_H

#include <string>
#include <unordered_map>
#include <vector>
#include "common/tuple.h"
#include "graph/op_desc.h"
#include "proto/om.pb.h"
#include "framework/common/fmk_types.h"

namespace ge {
enum class AttributeType {
  UNDEFINED,
  INT,
  UINT,
  BOOL,
  FLOAT,
  STRING,
  BYTES,

  INTLIST,
  UINTLIST,
  BOOLLIST,
  FLOATLIST,
  STRINGLIST
};

class OpSchema;

class OpSchemaRegistry;

class FMK_FUNC_HOST_VISIBILITY OpSchema {
 public:
  // Formal parameter options.
  enum FormalParameterOption {
    // The input formal parameter is single and not optional.
    // Number of this input is 1.
    Single = 0,
    // The input formal parameter is single and optional.
    // Number of this input is 0 or 1.
    Optional = 1,
    // The input formal parameter is variadic.
    // Number of this input is [1, n].
    Variadic = 2,
  };

  // Formal parameter represenation, including input/output name, typeStr,
  // description, and type constraints.
  class FormalParameter {
   public:
    // Constructor.
    FormalParameter() = default;

    explicit FormalParameter(const std::string &name, FormalParameterOption param_option = Single);

    ~FormalParameter();

    // Get formal parameter name.
    const std::string &Name() const;

    // Get the parameter option, it could be Single, Optional or Variadic.
    FormalParameterOption Option() const;

   private:
    friend class OpSchema;

    // Formal parameter name.
    std::string name_;

    // Formal parameter option.
    FormalParameterOption param_option_;
  };

  explicit OpSchema(const std::string &name);

  ~OpSchema();

  OpSchema &Input(const std::string &name, FormalParameterOption param_option = Single);

  OpSchema &Output(const std::string &name, FormalParameterOption param_option = Single);

  struct Attribute {
    Attribute(const std::string &name, AttributeType type, bool required)
        : name_(name), type_(type), required_(required) {}

    Attribute(const std::string &name, AttributeType type, domi::AttrDef default_value)
        : name_(name), type_(type), required_(false), default_value_(default_value) {}

    const std::string name_;
    AttributeType type_;
    bool required_;
    domi::AttrDef default_value_;
  };

  OpSchema &Attr(const Attribute &attr);

// Register "optional" attribute with default value.
#define ATTR_SETTER_WITH_DEFAULT_VALUE(TypeName)                                                           \
  OpSchema &Attr(const std::string &name, AttributeType type, const TypeName &default_value);              \
  OpSchema &Attr(const std::string &name, AttributeType type, const std::vector<TypeName> &default_value); \
  OpSchema &Attr(const std::string &name, AttributeType type, const Tuple<TypeName> &default_value);

  ATTR_SETTER_WITH_DEFAULT_VALUE(uint32_t)
  ATTR_SETTER_WITH_DEFAULT_VALUE(int64_t)
  ATTR_SETTER_WITH_DEFAULT_VALUE(bool)
  ATTR_SETTER_WITH_DEFAULT_VALUE(float)
  ATTR_SETTER_WITH_DEFAULT_VALUE(std::string)

  // Register "required" attribute without default value.
  OpSchema &AttrRequired(const std::string &name, AttributeType type);

  bool HasDefaultAttr(const std::string &name) const;

  const domi::AttrDef &GetDefaultAttr(const std::string &name) const;

  // verify op_def
  bool Verify(const ge::OpDescPtr op_def) const;

 private:
  friend class OpSchemaRegistry;

  std::string name_;

  std::vector<FormalParameter> inputs_;

  std::vector<FormalParameter> outputs_;

  std::unordered_map<std::string, Attribute> attributes_;
};

class OpSchemaFactory {
 public:
  // this is a singleton object
  static OpSchemaFactory &Instance();

  const OpSchema *Get(const std::string &op) const;

 private:
  OpSchemaFactory() = default;
  ~OpSchemaFactory() = default;

  friend class OpSchemaRegistry;
  // the op schema map
  std::unordered_map<std::string, OpSchema> op_schema_map_;
};

class FMK_FUNC_HOST_VISIBILITY OpSchemaRegistry {
 public:
  OpSchemaRegistry(OpSchema &op_schema);
  ~OpSchemaRegistry() = default;
};

#define DOMI_OP_SCHEMA(name) DOMI_OP_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define DOMI_OP_SCHEMA_UNIQ_HELPER(ctr, name) DOMI_OP_SCHEMA_UNIQ(ctr, name)
#define DOMI_OP_SCHEMA_UNIQ(ctr, name) \
  static OpSchemaRegistry op_schema_registry##ctr __attribute__((unused)) = OpSchema(#name)
}  // namespace ge
#endif  // DOMI_COMMON_OP_SCHEMA_H
