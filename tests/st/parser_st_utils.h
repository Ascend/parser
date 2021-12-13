/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef GE_PARSER_TESTS_UT_PARSER_H_
#define GE_PARSER_TESTS_UT_PARSER_H_

#include "framework/omg/parser/parser_inner_ctx.h"

namespace ge {
struct MemBuffer {
  void *data;
  uint32_t size;
};

class ParerSTestsUtils {
 public:
  static void ClearParserInnerCtx();
  static MemBuffer* MemBufferFromFile(const char *path);
  static bool ReadProtoFromText(const char *file, google::protobuf::Message *message);
  static void WriteProtoToBinaryFile(const google::protobuf::Message &proto, const char *filename);
};
}  // namespace ge

#endif  // GE_PARSER_TESTS_UT_PARSER_H_
