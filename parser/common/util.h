/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef PARSER_COMMON_UTIL_H_
#define PARSER_COMMON_UTIL_H_

#include "framework/common/debug/ge_log.h"
#include "mmpa/mmpa_api.h"

#define CHECK_FALSE_EXEC(expr, exec_expr, ...) \
  {                                            \
    bool b = (expr);                           \
    if (!b) {                                  \
      exec_expr;                               \
    }                                          \
  }

// For propagating errors when calling a function.
#define GE_RETURN_IF_ERROR(expr)         \
  do {                                   \
    const ::ge::Status _status = (expr); \
    if (_status) return _status;         \
  } while (0)

#define GE_RETURN_WITH_LOG_IF_ERROR(expr, ...) \
  do {                                         \
    const ::ge::Status _status = (expr);       \
    if (_status) {                             \
      GELOGE(ge::FAILED, __VA_ARGS__);         \
      return _status;                          \
    }                                          \
  } while (0)

// check whether the parameter is true. If it is, return FAILED and record the error log
#define GE_RETURN_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                             \
    if (condition) {                               \
      GELOGE(ge::FAILED, __VA_ARGS__);             \
      return ge::FAILED;                           \
    }                                              \
  } while (0)

// Check if the parameter is false. If yes, return FAILED and record the error log
#define GE_RETURN_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                              \
    bool _condition = (condition);                  \
    if (!_condition) {                              \
      GELOGE(ge::FAILED, __VA_ARGS__);              \
      return ge::FAILED;                            \
    }                                               \
  } while (0)

// Check if the parameter is false. If yes, return PARAM_INVALID and record the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                                        \
    bool _condition = (condition);                            \
    if (!_condition) {                                        \
      GELOGE(ge::FAILED, __VA_ARGS__);                        \
      return ge::PARAM_INVALID;                               \
    }                                                         \
  } while (0)

// Check if the parameter is null. If yes, return PARAM_INVALID and record the error
#define GE_CHECK_NOTNULL(val)                                    \
  do {                                                           \
    if (val == nullptr) {                                        \
      GELOGE(ge::FAILED, "param[%s] must not be null.", #val);   \
      return ge::PARAM_INVALID;                                  \
    }                                                            \
  } while (0)

// Check whether the parameter is null. If so, execute the exec_expr expression and record the error log
#define GE_CHECK_NOTNULL_EXEC(val, exec_expr)                    \
  do {                                                           \
    if (val == nullptr) {                                        \
      GELOGE(ge::FAILED, "param[%s] must not be null.", #val);   \
      exec_expr;                                                 \
    }                                                            \
  } while (0)

// Check if the value on the left is greater than or equal to the value on the right
#define GE_CHECK_GE(lhs, rhs)                                        \
  do {                                                               \
    if (lhs < rhs) {                                                 \
      GELOGE(ge::FAILED, "param[%s] is less than[%s]", #lhs, #rhs);  \
      return ge::PARAM_INVALID;                                      \
    }                                                                \
  } while (0)

#define GE_DELETE_NEW_SINGLE(var) \
  do {                            \
    if (var != nullptr) {         \
      delete var;                 \
      var = nullptr;              \
    }                             \
  } while (0)

#define GE_DELETE_NEW_ARRAY(var) \
  do {                           \
    if (var != nullptr) {        \
      delete[] var;              \
      var = nullptr;             \
    }                            \
  } while (0)

// If expr is true, execute exec_expr without printing logs
#define GE_IF_BOOL_EXEC(expr, exec_expr) \
  {                                      \
    if (expr) {                          \
      exec_expr;                         \
    }                                    \
  }

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_TRUE_EXEC_INFO(expr, exec_expr, ...) \
  {                                                      \
    bool b = (expr);                                     \
    if (b) {                                             \
      GELOGI(__VA_ARGS__);                               \
      exec_expr;                                         \
    }                                                    \
  }

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...) \
  do {                                             \
    bool b = (expr);                               \
    if (!b) {                                      \
      GELOGE(_status, __VA_ARGS__);                \
      return _status;                              \
    }                                              \
  } while (0);

// If expr is not SUCCESS, print the log and execute the expression + return _status
#define GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(expr, _status, exec_expr, ...) \
  {                                                                     \
    bool b = (expr);                                                    \
    if (b) {                                                            \
      GELOGE(ge::FAILED, __VA_ARGS__);                                  \
      exec_expr;                                                        \
      return _status;                                                   \
    }                                                                   \
  }

// If expr is not SUCCESS, print the log and return the same value
#define GE_CHK_STATUS_RET(expr, ...)   \
  do {                                 \
    const ge::Status _status = (expr); \
    if (_status != ge::SUCCESS) {      \
      GELOGE(ge::FAILED, __VA_ARGS__); \
      return _status;                  \
    }                                  \
  } while (0);

// If expr is true, print logs and execute custom statements
#define GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(expr, exec_expr, ...) \
  {                                                          \
    bool b = (expr);                                         \
    if (b) {                                                 \
      GELOGE(ge::FAILED, __VA_ARGS__);                       \
      exec_expr;                                             \
    }                                                        \
  }

// If expr is not SUCCESS, print the log and do not execute return
#define GE_CHK_STATUS(expr, ...)       \
  do {                                 \
    const ge::Status _status = (expr); \
    if (_status != ge::SUCCESS) {      \
      GELOGE(ge::FAILED, __VA_ARGS__); \
    }                                  \
  } while (0);

#define GE_LOGE_IF(condition, ...)       \
  if ((condition)) {                     \
    GELOGE(ge::FAILED, __VA_ARGS__);     \
  }

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC(expr, exec_expr, ...) \
  {                                            \
    bool b = (expr);                           \
    if (!b) {                                  \
      GELOGE(ge::FAILED, __VA_ARGS__);         \
      exec_expr;                               \
    }                                          \
  }

// ge marco
#define GE_LOGI_IF(condition, ...) \
  if ((condition)) {               \
    GELOGI(__VA_ARGS__);           \
  }

#define GE_LOGW_IF(condition, ...) \
  if ((condition)) {               \
    GELOGW(__VA_ARGS__);           \
  }

// If expr is not true, execute a custom statement
#define GE_CHK_BOOL_EXEC_NOLOG(expr, exec_expr) \
  {                                             \
    bool b = (expr);                            \
    if (!b) {                                   \
      exec_expr;                                \
    }                                           \
  }

// If expr is not SUCCESS, print the log and execute a custom statement
#define GE_CHK_STATUS_EXEC(expr, exec_expr, ...)                  \
  do {                                                            \
    const ge::Status _status = (expr);                            \
    GE_CHK_BOOL_EXEC(_status == SUCCESS, exec_expr, __VA_ARGS__); \
  } while (0);

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC_INFO(expr, exec_expr, ...) \
  {                                                 \
    bool b = (expr);                                \
    if (!b) {                                       \
      GELOGI(__VA_ARGS__);                          \
      exec_expr;                                    \
    }                                               \
  }

// If make_shared is abnormal, print the log and execute the statement
#define GE_MAKE_SHARED(exec_expr0, exec_expr1) \
  try {                                        \
    exec_expr0;                                \
  } catch (const std::bad_alloc &) {           \
    GELOGE(ge::FAILED, "Make shared failed");  \
    exec_expr1;                                \
  }

#endif  // PARSER_COMMON_UTIL_H_
