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

#ifndef PARSER_COMMON_FILE_SAVER_H_
#define PARSER_COMMON_FILE_SAVER_H_

#include <string>

#include "ge/ge_api_error_codes.h"
#include "register/register_types.h"
#include "nlohmann/json.hpp"

namespace ge {
namespace parser {
using Json = nlohmann::json;
using std::string;

class ModelSaver {
public:
  /**
   * @ingroup domi_common
   * @brief Save JSON object to file
   * @param [in] file_path File output path
   * @param [in] model json object
   * @return Status result
   */
  static Status SaveJsonToFile(const char *file_path, const Json &model);

private:
  ///
  /// @ingroup domi_common
  /// @brief Check validity of the file path
  /// @return Status  result
  ///
  static Status CheckPath(const string &file_path);

  static int CreateDirectory(const std::string &directory_path);
};
}  // namespace parser
}  // namespace ge

#endif //PARSER_COMMON_FILE_SAVER_H_
