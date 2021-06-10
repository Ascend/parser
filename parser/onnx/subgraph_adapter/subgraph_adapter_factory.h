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

#ifndef GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_FACTORY_H_
#define GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_FACTORY_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <functional>
#include "subgraph_adapter.h"

namespace ge {
/**
 * @brief Used to create OpParser
 *
 */
class PARSER_FUNC_VISIBILITY SubgraphAdapterFactory {
public:
  /**
   * @brief Returns the SubgraphAdapterFactory instance
   * @return SubgraphAdapterFactory object
   */
  static SubgraphAdapterFactory* Instance();

  /**
   * @brief Create SubgraphAdapter based on input type
   * @param [in] op_type Op type
   * @return Created SubgraphAdapter
   */
  std::shared_ptr<SubgraphAdapter> CreateSubgraphAdapter(const std::string &op_type);

  ~SubgraphAdapterFactory() = default;
protected:
  /**
   * @brief SubgraphAdapter creation function
   * @return Created SubgraphAdapter
   */
  // typedef shared_ptr<SubgraphAdapter> (*CREATOR_FUN)(void);
  using CREATOR_FUN = std::function<std::shared_ptr<SubgraphAdapter>(void)>;

  /**
   * @brief Factory instances can only be created automatically, not new methods, so the constructor is not public.
   */
  SubgraphAdapterFactory() {}

  /**
   * @brief Register creation function
   * @param [in] type Op type
   * @param [in] fun OpParser creation function
   */
  void RegisterCreator(const std::string &type, CREATOR_FUN fun);

private:
  std::map<std::string, CREATOR_FUN> subgraph_adapter_creator_map_;  // lint !e1073

  friend class SubgraphAdapterRegisterar;
};

/**
 * @brief For registering Creator functions for different types of subgraph adapter
 *
 */
class PARSER_FUNC_VISIBILITY SubgraphAdapterRegisterar {
public:
  /**
   * @brief Constructor
   * @param [in] op_type      Op type
   * @param [in] fun          Creator function corresponding to Subgrap adapter
   */
  SubgraphAdapterRegisterar(const std::string &op_type, SubgraphAdapterFactory::CREATOR_FUN fun) {
    SubgraphAdapterFactory::Instance()->RegisterCreator(op_type, fun);
  }
  ~SubgraphAdapterRegisterar() {}
};

/**
 * @brief SubgraphAdapter Registration Macro
 * @param [in] op_type      Op type
 * @param [in] clazz        SubgraphAdapter implementation class
 */
#define REGISTER_SUBGRAPH_ADAPTER_CREATOR(op_type, clazz)                       \
  std::shared_ptr<SubgraphAdapter> Creator_##op_type##_Subgraph_Adapter() {     \
    std::shared_ptr<clazz> ptr(new (std::nothrow) clazz());                     \
    if (ptr == nullptr) {                                                       \
      GELOGW("MakeShared failed, result is nullptr.");                          \
    }                                                                           \
    return std::shared_ptr<SubgraphAdapter>(ptr);                               \
  }                                                                             \
  ge::SubgraphAdapterRegisterar g_##op_type##_Subgraph_Adapter_Creator(op_type, \
                                                                       Creator_##op_type##_Subgraph_Adapter)
}  // namespace ge

#endif  // GE_PARSER_ONNX_SUBGRAPH_ADAPTER_SUBGRAPH_ADAPTER_FACTORY_H_
