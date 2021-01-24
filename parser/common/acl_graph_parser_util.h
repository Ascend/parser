/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef ACL_GRAPH_PARSE_UTIL_
#define ACL_GRAPH_PARSE_UTIL_

#include <google/protobuf/text_format.h>

#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/omg/parser/parser_types.h"
#include "graph/ascend_string.h"
#include "graph/utils/graph_utils.h"
#include "register/register_error_codes.h"

namespace ge {

using google::protobuf::Message;

class AclGrphParseUtil {
 public:
  AclGrphParseUtil() {}
  virtual ~AclGrphParseUtil() {}
  domi::Status LoadOpsProtoLib();
  void SaveCustomCaffeProtoPath();
  domi::Status AclParserInitialize(const std::map<std::string, std::string> &options);
  domi::Status SetOutputNodeInfo(ge::Graph &graph, const std::map<AscendString, AscendString> &parser_params);
  domi::Status ParseParamsBeforeGraph(const std::map<AscendString, AscendString> &parser_params,
                                      std::string &graph_name);
  domi::Status ParseParamsAfterGraph(ge::Graph &graph, const std::map<AscendString, AscendString> &parser_params);

 private:
  bool parser_initialized = false;
  domi::Status CheckOptions(const std::map<AscendString, AscendString> &parser_params);
  domi::Status GetOutputLeaf(NodePtr node, std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info);
  void GetOutputNodesNameAndIndex(std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info,
                                  std::vector<std::string> &output_nodes_name);
  void SetDefaultFormat();
  domi::Status ParseAclOutputNodes(const std::string &out_nodes);
  domi::Status ParseAclOutputFp16NodesFormat(const std::string &is_output_fp16);
  domi::Status ParseAclEnableScope(const std::string &enable_scope_fusion_passes);
  static void AddAttrsForInputNodes(const vector<string> &adjust_fp16_format_vec, const string &fp16_nodes_name,
                                    uint32_t index, OpDescPtr &op_desc);
  domi::Status ParseAclInputFp16Nodes(const ComputeGraphPtr &graph, const string &input_fp16_nodes,
                                      const string &is_input_adjust_hw_layout);
  domi::Status GetDefaultOutInfo(ge::ComputeGraphPtr &compute_graph,
                                 std::vector<std::pair<ge::NodePtr, int32_t>> &output_nodes_info);
};

namespace parser {
///
/// @ingroup: domi_common
/// @brief: get length of file
/// @param [in] input_file: path of file
/// @return long: File length. If the file length fails to be obtained, the value -1 is returned.
///
extern long GetFileLength(const std::string &input_file);

///
/// @ingroup domi_common
/// @brief Absolute path for obtaining files.
/// @param [in] path of input file
/// @param [out] Absolute path of a file. If the absolute path cannot be obtained, an empty string is returned
///
std::string RealPath(const char *path);

///
/// @ingroup domi_common
/// @brief Obtains the absolute time (timestamp) of the current system.
/// @return Timestamp, in microseconds (US)
///
///
uint64_t GetCurrentTimestamp();

///
/// @ingroup domi_common
/// @brief Reads all data from a binary file.
/// @param [in] file_name  path of file
/// @param [out] buffer  Output memory address, which needs to be released by the caller.
/// @param [out] length  Output memory size
/// @return false fail
/// @return true success
///
bool ReadBytesFromBinaryFile(const char *file_name, char **buffer, int &length);

///
/// @ingroup domi_common
/// @brief proto file in bianary format
/// @param [in] file path of proto file
/// @param [out] proto memory for storing the proto file
/// @return true success
/// @return false fail
///
bool ReadProtoFromBinaryFile(const char *file, Message *proto);

///
/// @ingroup domi_common
/// @brief Reads the proto structure from an array.
/// @param [in] data proto data to be read
/// @param [in] size proto data size
/// @param [out] proto Memory for storing the proto file
/// @return true success
/// @return false fail
///
bool ReadProtoFromArray(const void *data, int size, Message *proto);

///
/// @ingroup domi_proto
/// @brief Reads the proto file in the text format.
/// @param [in] file path of proto file
/// @param [out] message Memory for storing the proto file
/// @return true success
/// @return false fail
///
bool ReadProtoFromText(const char *file, google::protobuf::Message *message);

bool ReadProtoFromMem(const char *data, int size, google::protobuf::Message *message);

///
/// @brief get the Original Type of FrameworkOp
/// @param [in] node
/// @param [out] type
/// @return Status
///
domi::Status GetOriginalType(const ge::NodePtr &node, string &type);

///
/// @ingroup domi_common
/// @brief Check whether the file path meets the whitelist verification requirements.
/// @param [in] filePath file path
/// @param [out] result
///
bool ValidateStr(const std::string &filePath, const std::string &mode);

///
/// @ingroup domi_common
/// @brief Obtains the current time string.
/// @return Time character string in the format: %Y%m%d%H%M%S, eg: 20171011083555
///
std::string CurrentTimeInStr();

template <typename T, typename... Args>
static inline std::shared_ptr<T> MakeShared(Args &&... args) {
  typedef typename std::remove_const<T>::type T_nc;
  std::shared_ptr<T> ret(new (std::nothrow) T_nc(std::forward<Args>(args)...));
  return ret;
}

/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline domi::Status Int64MulCheckOverflow(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return domi::FAILED;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return domi::FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return domi::FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return domi::FAILED;
      }
    }
  }
  return domi::SUCCESS;
}

/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline domi::Status CheckInt64Uint32MulOverflow(int64_t a, uint32_t b) {
  if (a == 0 || b == 0) {
    return domi::SUCCESS;
  }
  if (a > 0) {
    if (a > (INT64_MAX / b)) {
      return domi::FAILED;
    }
  } else {
    if (a < (INT64_MIN / b)) {
      return domi::FAILED;
    }
  }
  return domi::SUCCESS;
}

#define PARSER_INT64_MULCHECK(a, b)                                                                             \
  if (ge::parser::Int64MulCheckOverflow((a), (b)) != SUCCESS) {                                                 \
    GELOGW("Int64 %ld and %ld multiplication can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                                            \
    return INTERNAL_ERROR;                                                                                      \
  }

#define PARSER_INT64_UINT32_MULCHECK(a, b)                                                                         \
  if (ge::parser::CheckInt64Uint32MulOverflow((a), (b)) != SUCCESS) {                                              \
    GELOGW("Int64 %ld and Uint32 %u multiplication can result in overflow!", static_cast<uint64_t>(a),             \
           static_cast<uint32_t>(b));                                                                              \
    return INTERNAL_ERROR;                                                                                         \
  }
}  // namespace parser
}  // namespace ge

/*lint --emacro((773),GE_TIMESTAMP_START)*/
/*lint -esym(773,GE_TIMESTAMP_START)*/
#define PARSER_TIMESTAMP_START(stage) uint64_t startUsec_##stage = ge::parser::GetCurrentTimestamp()

#define PARSER_TIMESTAMP_END(stage, stage_name)                                       \
  do {                                                                                \
    uint64_t endUsec_##stage = ge::parser::GetCurrentTimestamp();                     \
    GELOGI("[GEPERFTRACE] The time cost of %s is [%lu] micro second.", (stage_name),  \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (0);

#define PARSER_TIMESTAMP_EVENT_END(stage, stage_name)                                 \
  do {                                                                                \
    uint64_t endUsec_##stage = ge::parser::GetCurrentTimestamp();                     \
    GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro second.", (stage_name), \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (0);

#endif  // ACL_GRAPH_PARSE_UTIL_
