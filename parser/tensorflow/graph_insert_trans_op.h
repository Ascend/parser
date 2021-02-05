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

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_INSERT_TRANS_OP_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_INSERT_TRANS_OP_H_
#include <map>
#include <string>
#include <vector>
#include "common/fmk_types.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/op_registry.h"

namespace ge {
enum InFmtSupportEnum {
  InFmtSupportUndefined,
  InFmtSupportElewise,
  InFmtSupport4D,
  InFmtSupport5D,
  InFmtSupport4D_5D,
  InFmtSupportNCHW_NC1HWC0
};

enum InDtSupportEnum {
  InDtSupportUndefined = 0,
  InDtSupportAll = 1,
};

enum OutFmtSupportEnum {
  OutFmtSupportUndefined = 0,
  OutFmtSupportAsInput = 1,
};

enum OutDtSupportEnum {
  OutDtSupportUndefined = 0,
  OutDtSupportAsInput = 1,
};

struct OpSupportTranInfo {
  InFmtSupportEnum inputFormatSupportEnum = InFmtSupportUndefined;
  InDtSupportEnum inputDataTypeSupportEnum = InDtSupportUndefined;
  OutFmtSupportEnum outputFormatSupportEnum = OutFmtSupportUndefined;
  OutDtSupportEnum outputDataTypeSupportEnum = OutDtSupportUndefined;

  std::vector<ge::Format> inputFormats;
  std::vector<ge::DataType> inputDataTypes;
  ge::Format limitOutputFormat = ge::FORMAT_RESERVED;
  ge::DataType limitOutputDataType = ge::DT_UNDEFINED;
};

extern std::map<std::string, OpSupportTranInfo> g_OpSupportTranInfo;

class OpTransAddSupportReg {
 public:
  template <class InFmts, class InDts, class OutFmts, class OutDts>
  OpTransAddSupportReg(const std::string &cceTbeTg, const std::string &opType,
                       InFmts inputFormats, InDts inputDataTypes,
                       OutFmts outputormat, OutDts outputDataType) {
    auto cceTbeOpType = cceTbeTg + ":" + opType;
    g_OpSupportTranInfo.erase(cceTbeOpType);
    SetInputFormat(cceTbeOpType, inputFormats);
    SetInputDataType(cceTbeOpType, inputDataTypes);
    SetOutputFormat(cceTbeOpType, outputormat);
    SetOutputDataType(cceTbeOpType, outputDataType);
  }
  ~OpTransAddSupportReg() = default;

 private:
  void SetInputFormat(std::string opType,
                      const std::vector<ge::Format>& supportFormat) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    for (auto& format : supportFormat) {
      opInfo.inputFormats.push_back(format);
    }
  }

  void SetInputFormat(std::string opType, ge::Format supportFormat) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.inputFormats.push_back(supportFormat);
  }

  void SetInputFormat(std::string opType, InFmtSupportEnum enumFormat) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.inputFormatSupportEnum = enumFormat;
    switch (enumFormat) {
      case InFmtSupportElewise:
        opInfo.inputFormats = {ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN,
                               ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC,
                               ge::FORMAT_NCHW};
        break;
      case InFmtSupport4D:
        opInfo.inputFormats = {ge::FORMAT_HWCN, ge::FORMAT_NHWC,
                               ge::FORMAT_NCHW};
        break;
      case InFmtSupport5D:
        opInfo.inputFormats = {ge::FORMAT_NC1HWC0};
        break;
      case InFmtSupport4D_5D:
        opInfo.inputFormats = {ge::FORMAT_HWCN, ge::FORMAT_NHWC,
                               ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0};
        break;
      case InFmtSupportNCHW_NC1HWC0:
        opInfo.inputFormats = {ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW};
        break;
      default:
        break;
    }
  }

  void SetInputDataType(std::string opType,
                        const std::vector<ge::DataType>& supportDataType) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    for (auto& dataType : supportDataType) {
      opInfo.inputDataTypes.push_back(dataType);
    }
  }

  void SetInputDataType(std::string opType, ge::DataType supportDataType) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.inputDataTypes.push_back(supportDataType);
  }

  void SetInputDataType(std::string opType, InDtSupportEnum enumDataType) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.inputDataTypeSupportEnum = enumDataType;
  }

  void SetOutputFormat(std::string opType, ge::Format limitOutputormat) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.limitOutputFormat = limitOutputormat;
  }

  void SetOutputFormat(std::string opType, OutFmtSupportEnum enumFormat) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.outputFormatSupportEnum = enumFormat;
  }

  void SetOutputDataType(std::string opType, ge::DataType limitOutputDataType) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.limitOutputDataType = limitOutputDataType;
  }

  void SetOutputDataType(std::string opType, OutDtSupportEnum enumDataType) {
    auto& opInfo = g_OpSupportTranInfo[opType];
    opInfo.outputDataTypeSupportEnum = enumDataType;
  }
};

#define TBE_SET_FORMAT_DATAYPE_INFO(cce_tbe, op, inputFormats, inputDataType, \
                                    outFormats, outputDataTypes)              \
  TBE_SET_FORMAT_DATAYPE_INFO_UNIQ_HELPER(__COUNTER__, #cce_tbe, op,          \
                                          inputFormats, inputDataType,        \
                                          outFormats, outputDataTypes)
#define TBE_SET_FORMAT_DATAYPE_INFO_UNIQ_HELPER(ctr, cce_tbe, op,            \
                                                inputFormats, inputDataType, \
                                                outFormats, outputDataTypes) \
  TBE_SET_FORMAT_DATAYPE_INFO_UNIQ(ctr, cce_tbe, op, inputFormats,           \
                                   inputDataType, outFormats, outputDataTypes)
#define TBE_SET_FORMAT_DATAYPE_INFO_UNIQ(ctr, cce_tbe, op, inputFormats, \
                                         inputDataType, outFormats,      \
                                         outputDataTypes)                \
  OpTransAddSupportReg __gOpTransAddSupportReg##ctr(                     \
      cce_tbe, op, inputFormats, inputDataType, outFormats, outputDataTypes);
}  // namespace domi
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_INSERT_TRANS_OP_H_
