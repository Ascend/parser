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

#include <sys/stat.h>
#include <fcntl.h>

#include "parser/common/model_saver.h"
#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "mmpa/mmpa_api.h"

namespace {
const int kFileOpSuccess = 0;
}  //  namespace

namespace ge {
namespace parser {
const uint32_t kInteval = 2;

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelSaver::SaveJsonToFile(const char *file_path,
                                                                                   const Json &model) {
  Status ret = SUCCESS;
  if (file_path == nullptr || SUCCESS != CheckPath(file_path)) {
    REPORT_INNER_ERROR("E19999", "param file_path is nullptr or checkpath not return success");
    GELOGE(FAILED, "[Check][Param]Check output file failed.");
    return FAILED;
  }
  std::string model_str;
  try {
    model_str = model.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERROR("E19999", "Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    GELOGE(FAILED, "[Invoke][Dump] Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    return FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Failed to convert JSON to string, savefile:%s.", file_path);
    GELOGE(FAILED, "[Invoke][Dump] Failed to convert JSON to string, savefile:%s.", file_path);
    return FAILED;
  }

  char real_path[PATH_MAX] = {0};
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(file_path) >= PATH_MAX,
                                 REPORT_INNER_ERROR("E19999", "file path %s is too long!", file_path);
                                 return FAILED, "[Check][Param] file path %s is too long!", file_path);
  if (realpath(file_path, real_path) == nullptr) {
    GELOGI("File %s does not exit, it will be created.", file_path);
  }

  // Open file
  mode_t mode = S_IRUSR | S_IWUSR;
  int32_t fd = mmOpen2(real_path, O_RDWR | O_CREAT | O_TRUNC, mode);
  if (fd == EN_ERROR || fd == EN_INVALID_PARAM) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file_path, strerror(errno)});
    GELOGE(FAILED, "[Open][File] [%s] failed. %s", file_path, strerror(errno));
    return FAILED;
  }
  const char *model_char = model_str.c_str();
  uint32_t len = static_cast<uint32_t>(model_str.length());
  // Write data to file
  mmSsize_t mmpa_ret = mmWrite(fd, const_cast<void *>((const void *)model_char), len);
  if (mmpa_ret == EN_ERROR || mmpa_ret == EN_INVALID_PARAM) {
    ErrorManager::GetInstance().ATCReportErrMessage(
            "E19004", {"file", "errmsg"}, {file_path, strerror(errno)});
    // Need to both print the error info of mmWrite and mmClose, so return ret after mmClose
    GELOGE(FAILED, "[WriteTo][File] %s failed. errno = %ld, %s", file_path, mmpa_ret, strerror(errno));
    ret = FAILED;
  }
  // Close file
  if (mmClose(fd) != EN_OK) {
    REPORT_INNER_ERROR("E19999", "close file:%s failed. errmsg:%s", file_path, strerror(errno));
    GELOGE(FAILED, "[Close][File] %s failed. errmsg:%s", file_path, strerror(errno));
    ret = FAILED;
  }
  return ret;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status ModelSaver::CheckPath(const std::string &file_path) {
  // Determine file path length
  if (file_path.size() >= PATH_MAX) {
    REPORT_INNER_ERROR("E19999", "Path is too long:%zu", file_path.size());
    GELOGE(FAILED, "[Check][Param] Path is too long:%zu", file_path.size());
    return FAILED;
  }

  // Find the last separator
  int path_split_pos = static_cast<int>(file_path.size() - 1);
  for (; path_split_pos >= 0; path_split_pos--) {
    if (file_path[path_split_pos] == '\\' || file_path[path_split_pos] == '/') {
      break;
    }
  }

  if (path_split_pos == 0) {
    return SUCCESS;
  }

  // If there is a path before the file name, create the path
  if (path_split_pos != -1) {
    if (CreateDirectory(std::string(file_path).substr(0, static_cast<size_t>(path_split_pos))) != kFileOpSuccess) {
      GELOGE(FAILED, "[Create][Directory] failed, file path:%s.", file_path.c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY int ModelSaver::CreateDirectory(const std::string &directory_path) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(), return -1, "directory path is empty.");
  auto dir_path_len = directory_path.length();
  if (dir_path_len >= PATH_MAX) {
    ErrorManager::GetInstance().ATCReportErrMessage(
            "E19002", {"filepath", "size"}, {directory_path, std::to_string(PATH_MAX)});
    GELOGW("Path[%s] len is too long, it must be less than %d", directory_path.c_str(), PATH_MAX);
    return -1;
  }
  char tmp_dir_path[PATH_MAX] = {0};
  for (size_t i = 0; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      if (access(tmp_dir_path, F_OK) != 0) {
        int32_t ret = mmMkdir(tmp_dir_path, S_IRUSR | S_IWUSR | S_IXUSR);  // 700
        if (ret != 0) {
          if (errno != EEXIST) {
            REPORT_CALL_ERROR("E19999",
                              "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                              directory_path.c_str(), strerror(errno));
            GELOGW("Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                   directory_path.c_str(), strerror(errno));
            return ret;
          }
        }
      }
    }
  }
  int32_t ret = mmMkdir(const_cast<char *>(directory_path.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);  // 700
  if (ret != 0) {
    if (errno != EEXIST) {
      REPORT_CALL_ERROR("E19999",
                        "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                        directory_path.c_str(), strerror(errno));
      GELOGW("Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
             directory_path.c_str(), strerror(errno));
      return ret;
    }
  }
  return 0;
}

}  // namespace parser
}  // namespace ge