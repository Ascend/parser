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

#ifndef PARSER_COMMON_OP_PARSER_FACTORY_H_
#define PARSER_COMMON_OP_PARSER_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "parser/common/acl_graph_parser_util.h"
#include "framework/omg/parser/parser_types.h"
#include "framework/common/debug/ge_log.h"
#include "omg/omg_inner_types.h"
#include "external/register/register.h"

using domi::CAFFE;

namespace ge {
class OpParser;

/**
 * @ingroup domi_omg
 * @brief Used to create OpParser
 *
 */
class OpParserFactory {
 public:
  /**
   * @ingroup domi_omg
   * @brief Returns the OpParserFactory instance corresponding to the Framework
   * @return OpParserFactory object
   */
  static std::shared_ptr<OpParserFactory> Instance(const domi::FrameworkType framework);

  /**
   * @ingroup domi_omg
   * @brief Create OpParser based on input type
   * @param [in] op_type Op type
   * @return Created OpParser
   */
  std::shared_ptr<OpParser> CreateOpParser(const std::string &op_type);

  /**
   * @ingroup domi_omg
   * @brief Create fusion OpParser based on input type
   * @param [in] op_type Op type
   * @return Created OpParser
   */
  std::shared_ptr<OpParser> CreateFusionOpParser(const std::string &op_type);

  // The Factory instance is automatically released by shared_ptr.
  // The shared_ptr internally calls the destructor indirectly.
  // If the destructor is not public, it will generate a compilation error.
  // Another solution is to specify the deleter for shared_ptr, and set the deleter as a friend of the current class.
  // But this method is more complicated to implement.
  ~OpParserFactory() {}

  bool OpParserIsRegistered(const std::string &op_type, bool is_fusion_op = false);

 protected:
  /**
   * @ingroup domi_omg
   * @brief OpParser creation function
   * @return Created OpParser
   */
  // typedef shared_ptr<OpParser> (*CREATOR_FUN)(void);
  using CREATOR_FUN = std::function<std::shared_ptr<OpParser>(void)>;

  /**
   * @ingroup domi_omg
   * @brief Factory instances can only be created automatically, not new methods, so the constructor is not public.
   */
  OpParserFactory() {}

  /**
   * @ingroup domi_omg
   * @brief Register creation function
   * @param [in] type Op type
   * @param [in] fun OpParser creation function
   */
  void RegisterCreator(const std::string &type, CREATOR_FUN fun, bool is_fusion_op = false);

 private:
  /**
   * @ingroup domi_omg
   * @brief Each Op corresponds to a Creator function
   */
  std::map<std::string, CREATOR_FUN> op_parser_creator_map_;  // lint !e1073
  std::map<std::string, CREATOR_FUN> fusion_op_parser_creator_map_;

  friend class OpParserRegisterar;
  friend class domi::OpRegistrationData;
  friend class OpRegistrationTbe;
};

/**
 * @ingroup domi_omg
 * @brief For registering Creator functions for different types of Op
 *
 */
class OpParserRegisterar {
 public:
  /**
   * @ingroup domi_omg
   * @brief Constructor
   * @param [in] framework    Framework type
   * @param [in] op_type      Op type
   * @param [in] fun          Creator function corresponding to Op
   */
  OpParserRegisterar(const domi::FrameworkType framework, const std::string &op_type, OpParserFactory::CREATOR_FUN fun,
                     bool is_fusion_op = false) {
    OpParserFactory::Instance(framework)->RegisterCreator(op_type, fun, is_fusion_op);
  }
  ~OpParserRegisterar() {}
};

// Used to save the functions created by the xxxCustomParserAdapter class
class CustomParserAdapterRegistry {
 public:
  static CustomParserAdapterRegistry *Instance();
  using CREATOR_FUN = std::function<std::shared_ptr<OpParser>(void)>;
  void Register(const domi::FrameworkType framework, CREATOR_FUN fun);
  CREATOR_FUN GetCreateFunc(const domi::FrameworkType framework);

 private:
  map<domi::FrameworkType, CREATOR_FUN> funcs_;

  friend class CustomParserAdapterRegistrar;
};

// Register Creator function for the custom custom operator ParserAdapter
class CustomParserAdapterRegistrar {
 public:
  CustomParserAdapterRegistrar(const domi::FrameworkType framework, CustomParserAdapterRegistry::CREATOR_FUN fun) {
    CustomParserAdapterRegistry::Instance()->Register(framework, fun);
  }
  ~CustomParserAdapterRegistrar() {}
};

/**
 * @ingroup domi_omg
 * @brief OpParser Registration Macro
 * @param [in] framework    Framework type
 * @param [in] op_type      Op type
 * @param [in] clazz        OpParser implementation class
 */
#define REGISTER_OP_PARSER_CREATOR(framework, op_type, clazz)                              \
  std::shared_ptr<OpParser> Creator_##framework##_##op_type##_Op_Parser() {                \
    std::shared_ptr<clazz> ptr = ge::parser::MakeShared<clazz>();                                  \
    if (ptr == nullptr) {                                                                  \
      GELOGW("MakeShared failed, result is nullptr.");                                     \
    }                                                                                      \
    return std::shared_ptr<OpParser>(ptr);                                                 \
  }                                                                                        \
  ge::OpParserRegisterar g_##framework##_##op_type##_Op_Parser_Creator(framework, op_type, \
                                                                       Creator_##framework##_##op_type##_Op_Parser)

#define REGISTER_FUSION_OP_PARSER_CREATOR(framework, op_type, clazz)               \
  std::shared_ptr<OpParser> Creator_##framework##_##op_type##_Fusion_Op_Parser() { \
    std::shared_ptr<clazz> ptr = ge::parser::MakeShared<clazz>();                          \
    if (ptr == nullptr) {                                                          \
      GELOGW("MakeShared failed, result is nullptr.");                             \
    }                                                                              \
    return std::shared_ptr<OpParser>(ptr);                                         \
  }                                                                                \
  OpParserRegisterar g_##framework##_##op_type##_Fusion_Op_Parser_Creator(         \
    framework, op_type, Creator_##framework##_##op_type##_Fusion_Op_Parser, true)

/// @brief xxxCustomParserAdapter Registration Macro
/// @param [in] framework    Framework type
/// @param [in] clazz        CaffeCustomParserAdapter adaptation class
#define REGISTER_CUSTOM_PARSER_ADAPTER_CREATOR(framework, clazz)         \
  std::shared_ptr<OpParser> Creator_##framework##_Op_Parser_Adapter() { \
    std::shared_ptr<clazz> ptr = ge::parser::MakeShared<clazz>();               \
    if (ptr == nullptr) {                                               \
      GELOGW("MakeShared failed, result is nullptr.");                  \
    }                                                                   \
    return std::shared_ptr<OpParser>(ptr);                              \
  }                                                                     \
  CustomParserAdapterRegistrar g_##framework##_Op_Parser_Creator(framework, Creator_##framework##_Op_Parser_Adapter)
}  // namespace ge
#endif  // PARSER_COMMON_OP_PARSER_FACTORY_H_
