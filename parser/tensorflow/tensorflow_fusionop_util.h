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

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_FUSIONOP_UTIL_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_FUSIONOP_UTIL_H_
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "common/string_util.h"
#include "framework/omg/parser/parser_types.h"
#include "omg/omg_inner_types.h"
#include "proto/tensorflow/graph.pb.h"
#include "external/register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"

namespace ge {
using std::string;
using std::vector;
extern map<string, vector<std::pair<string, uint32_t>>> tensorflow_fusionop_input_const_weight_index_map;
extern vector<string> const_op_update_vec;

class TensorFlowFunsionOPUtil {
 public:
  /**
  * @ingroup domi_omg
  * @brief Check whether the operator can be a fusion operator
  * @param [in] node_name operation name
  * @return info fusion operator description
  * @return true maybe
  * @return false maybe not
  * @author
  */
  static bool MaybeFusionOp(const string &node_name, ScopeFusionOpInfo *info);

  /**
  * @ingroup domi_omg
  * @brief Confirm whether it is a fusion operator
  * @param [in] nodeDef
  * @return true
  * @return false
  * @author
  */
  static bool IsFusionOp(const domi::tensorflow::NodeDef *node_def);

  /**
  * @ingroup domi_omg
  * @brief Check the validity of fusion operator(All child nodes)
  * @param [in] fusion_node_name fusion operator name
  * @param [in] nodedef_list child nodes list
  * @param [in] funsion_op_type fusion operator type
  * @return legal/illegal
  * @author
  */
  static Status CheckFusionOpChildren(const string &fusion_node_name,
                                      const vector<const domi::tensorflow::NodeDef *> &nodedef_list,
                                      const string &funsion_op_type);

  /**
  * @ingroup domi_omg
  * @brief get inPut index of the fusion operator
  * @param [in] info  Child node description of fusion operator
  * @param [in] old_index  Child node original index
  * @return old_index As input index of the fusion operator
  * @return return code
  * @author
  */
  static Status GetInPutIndex(const ScopeFusionOpInfo &info, const int32_t old_index, int32_t &new_index);

  /**
  * @ingroup domi_omg
  * @brief get outPut index of the fusion operator
  * @param [in] info  Child node description of fusion operator
  * @param [in] old_index  Child node original index
  * @return old_index As output index of the fusion operator
  * @return 返回码
  * @author
  */
  static Status GetOutPutIndex(const ScopeFusionOpInfo &info, const int32_t old_index, int32_t &new_index);

  static bool FusionOpChildIgnore(const ScopeFusionOpInfo &info);
  /**
  * @ingroup domi_omg
  * @brief Get child node name of fusion operator eg: input: fastrcnn_predictions/map/TensorArray_2 output
  * :map/TensorArray_2
  * @param [in] node_name node name
  * @param [in] fusion_node_name fusion node name
  * @return Child node name of the fusion node
  * @author
  */
  static string GetChildName(const string &node_name, const string &fusion_node_name);

 private:
  /**
  * @ingroup domi_omg
  * @brief whether a string can be converted to an integer
  * @param [in] indexstr Operator suffix index
  * @return true can
  * @return false can not
  * @author
  */
  static bool IsIntegerStr(const string &index_str);

  /**
  * @ingroup domi_omg
  * @brief Get child node of fusion operator
  * @param [in] info Description of fusion operator
  * @param [in] old_index original index
  * @return new_index Fusion operator index
  * @author
  */
  static Status GetNodeindex(const ScopeFusionOpInfo &info, const int32_t old_index, int32_t &new_index,
                             const std::map<string, vector<std::pair<string, vector<int32_t>>>> &fusionop_context_map);
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_FUSIONOP_UTIL_H_
