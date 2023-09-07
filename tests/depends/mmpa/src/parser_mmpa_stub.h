/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef PARSER_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_
#define PARSER_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_

#include "mmpa/mmpa_api.h"
#include <memory>

#include <iostream>

namespace ge {
class MmpaStubApi {
 public:
  MmpaStubApi() = default;
  virtual ~MmpaStubApi() = default;

  virtual INT32 mmDladdr(VOID *addr, mmDlInfo *info)
  {
    return 0;
  }

  virtual VOID *mmDlopen(const CHAR *fileName, INT32 mode)
  {
    return NULL;
  }

  virtual INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen)
  {
    return 0;
  }
};

class MmpaStub {
 public:
  static MmpaStub& GetInstance() {
    static MmpaStub mmpa_stub;
    return mmpa_stub;
  }

  void SetImpl(const std::shared_ptr<MmpaStubApi> &impl) {
    impl_ = impl;
  }

  MmpaStubApi* GetImpl() {
    return impl_.get();
  }

  void Reset() {
    impl_ = std::make_shared<MmpaStubApi>();
  }

 private:
  MmpaStub(): impl_(std::make_shared<MmpaStubApi>()){}
  std::shared_ptr<MmpaStubApi> impl_;
};

}  // namespace parser
#endif // PARSER_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_