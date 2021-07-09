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

#include "graph_optimizer.h"
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include "./graph_insert_trans_op.h"
#include "cce/cce.h"
#include "cce/dnn.h"
#include "parser/common/acl_graph_parser_util.h"
#include "common/op_map.h"
#include "common/op_types.h"
#include "common/types_map.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_functiondef.h"
#include "parser/common/acl_graph_parser_util.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "register/op_registry.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using domi::tensorflow::TensorShapeProto;
using domi::tensorflow::TensorShapeProto_Dim;

using ge::FORMAT_NC1HWC0;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;

using ge::AttrUtils;
using ge::Buffer;
using ge::ComputeGraph;
using ge::ComputeGraphPtr;
using ge::GE_TENSORFLOW_DATA_TYPE_MAP;
using ge::GeShape;
using ge::GeTensorDesc;
using ge::GeTensorPtr;
using ge::GRAPH_SUCCESS;
using ge::GraphToFunctionDef;
using ge::GraphUtils;
using ge::InControlAnchorPtr;
using ge::InDataAnchorPtr;
using ge::is_dataset_op_vec;
using ge::local_framework_op_vec;
using ge::NodePtr;
using ge::OpDesc;
using ge::OpDescPtr;
using ge::OutControlAnchorPtr;
using ge::OutDataAnchorPtr;
using ge::TensorUtils;

using ge::ATTR_NAME_INPUT_DATATYPE;
using ge::ATTR_NAME_OUTPUT_DATATYPE;

namespace ge {
REGISTER_OPTYPE_DEFINE(TF_MAXIMUM_GRAD, "MaximumGrad");
REGISTER_OPTYPE_DEFINE(TF_MATMUL, "Matmul");
REGISTER_OPTYPE_DEFINE(TFRELU6, "Relu6");
REGISTER_OPTYPE_DEFINE(TF_BATCH_MATMUL, "BatchMatmul");
}  // namespace ge

namespace ge {
namespace {
const char RRTVAL_NODE_NAME_SUFFIX[] = "_RetVal";
const char *const kShapeNodeName = "Shape";
}  // namespace

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::map<string, OpSupportTranInfo> g_OpSupportTranInfo = {};

TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::CAST, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportUndefined)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::CAST, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportUndefined)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::ADDN, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::ADDN, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::ADD, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::ADD, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::MUL,
                            std::vector<ge::Format>({ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                                     ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0}),
                            InDtSupportAll, OutFmtSupportAsInput, OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::L2LOSS,
                            std::vector<ge::Format>({ge::FORMAT_FRACTAL_Z, ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC,
                                                     ge::FORMAT_HWCN}),  // inputformats
                            ge::DT_FLOAT, ge::FORMAT_NC1HWC0, ge::DT_FLOAT)

TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::CONVGRADFILTER, InFmtSupportUndefined, InDtSupportUndefined,
                            ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::CONV2DBACKPROPINPUT, InFmtSupportUndefined, InDtSupportUndefined,
                            ge::FORMAT_NC1HWC0, ge::DT_FLOAT16)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::BIASADDGRAD, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,
                            ge::DT_FLOAT)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::BIASADD, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::DT_FLOAT)

TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::ACTIVATION, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,
                            ge::DT_FLOAT16)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::ACTIVATIONGRAD, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,
                            ge::DT_FLOAT16)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::SOFTMAX, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,
                            ge::DT_FLOAT16)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SOFTMAX, InFmtSupport4D, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)

TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::DEPTHWISECONV2DBACKPROPFILTER, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16,
                            ge::FORMAT_C1HWNCoC0, ge::DT_FLOAT)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::DEPTHWISECONV2DBACKPORPINPUT, InFmtSupportUndefined, InDtSupportUndefined,
                            OutFmtSupportAsInput, OutDtSupportUndefined)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::DEPTHWISECONV2DFORWARDNATIVE, InFmtSupportUndefined, InDtSupportUndefined,
                            OutFmtSupportAsInput, OutDtSupportUndefined)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::FUSEDBATCHNORM, InFmtSupportUndefined, InDtSupportUndefined,
                            OutFmtSupportAsInput, OutDtSupportUndefined)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::FUSEDBATCHNORMGRAD, InFmtSupportUndefined, InDtSupportUndefined,
                            OutFmtSupportAsInput, OutDtSupportUndefined)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::CONV2D, InFmtSupportUndefined, InDtSupportUndefined, OutFmtSupportAsInput,
                            OutDtSupportUndefined)

TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::RESHAPE, ge::FORMAT_NHWC, InDtSupportAll, ge::FORMAT_NHWC,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::SPARSESOFTMAXCROSSENTROPYWITHLOGITS, InFmtSupport5D, ge::DT_FLOAT16,
                            OutFmtSupportAsInput, OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::TF_MAXIMUM_GRAD, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::APPLYRMSPROP,
                            std::vector<ge::Format>({ge::FORMAT_FRACTAL_Z, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                                     ge::FORMAT_HWCN, ge::FORMAT_NC1HWC0}),
                            ge::DT_FLOAT, OutFmtSupportAsInput, OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::DROPOUTDOMASK, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::LOG, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SQRTGRAD, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SIGMOIDGRAD, InFmtSupport4D, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SIGMOID, InFmtSupport4D, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::ARGMAX, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::AVGPOOLGRAD, InFmtSupport5D, ge::DT_FLOAT16, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::NEG, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::RECIPROCAL, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SQUARE, InFmtSupport4D, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SUB, InFmtSupport4D, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SUM, InFmtSupport4D, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::TF_MATMUL, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput, OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::GATHERV2, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::GREATEREQUAL, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::REALDIV, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SQRT, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::STRIDEDSLICE, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::TILE, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::TFRELU6, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::RELU6GRAD, InFmtSupportElewise, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::EQUAL, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::GREATER, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::SELECT, InFmtSupport4D, ge::DT_FLOAT, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::TF_BATCH_MATMUL, ge::FORMAT_NHWC, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(TBE, ge::parser::TRANSPOSE, ge::FORMAT_NHWC, InDtSupportAll, OutFmtSupportAsInput,
                            OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::STREAMMERGE,
                            std::vector<ge::Format>({ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0}),
                            InDtSupportAll, OutFmtSupportAsInput, OutDtSupportAsInput)
TBE_SET_FORMAT_DATAYPE_INFO(CCE, ge::parser::MEMCPYASYNC,
                            std::vector<ge::Format>({ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0}),
                            InDtSupportAll, OutFmtSupportAsInput, OutDtSupportAsInput)

bool GetCceTbeTransInfo(string opType, OpSupportTranInfo &opSupportInfo) {
  static bool fmtInited = false;
  GE_IF_BOOL_EXEC(
      !fmtInited, fmtInited = true;
      if (domi::OpRegistry().Instance()->GetImplyType(ge::parser::DEPTHWISEWEIGHT4D26D) == domi::ImplyType::TVM) {
        auto it = g_OpSupportTranInfo.find(string("TBE:") + ge::parser::MUL);
        if (it != g_OpSupportTranInfo.end()) {
          auto &fmts = it->second.inputFormats;
          auto itFmt = std::find(fmts.begin(), fmts.end(), ge::FORMAT_NC1HWC0);
          fmts.erase(itFmt);
        }
      })
  string cceTbeOpType = "TBE";
  GE_IF_BOOL_EXEC(domi::OpRegistry().Instance()->GetImplyType(opType) == domi::ImplyType::BUILDIN,
                  cceTbeOpType = "CCE";)
  cceTbeOpType = cceTbeOpType + ":" + opType;
  GE_IF_BOOL_EXEC(g_OpSupportTranInfo.find(cceTbeOpType) != g_OpSupportTranInfo.end(),
                  opSupportInfo = g_OpSupportTranInfo[cceTbeOpType];
                  return true;)
  return false;
}

Status ParserGraphOptimizer::Optimize() { return SUCCESS; }

Status ParserGraphOptimizer::OptimizeAfterCal() { return SUCCESS; }

void SetStringAttr(const string &originalType, OpDescPtr &opDesc,
                   google::protobuf::Map<string, domi::tensorflow::AttrValue> *tfAttr,
                   const pair<const string, ge::GeAttrValue> &attr) {
  string s;
  (void)AttrUtils::GetStr(opDesc, attr.first, s);

  if (originalType == "ParallelMapDataset" || originalType == "FilterDataset" ||
      originalType == "MapAndBatchDatasetV2") {
    ::domi::tensorflow::NameAttrList *nameAttrList = (*tfAttr)[attr.first].mutable_func();
    nameAttrList->set_name(s);
  } else {
    (*tfAttr)[attr.first].set_s(s);
  }
}

void SetIntAttr(const string &originalType, OpDescPtr &opDesc,
                google::protobuf::Map<string, domi::tensorflow::AttrValue> *tfAttr,
                const pair<const string, ge::GeAttrValue> &attr) {
  int32_t i = 0;
  (void)AttrUtils::GetInt(opDesc, attr.first, i);

  if (originalType == "Pack" && (attr.first == "axis" || attr.first == "N")) {
    (*tfAttr)[attr.first].set_i(i);
  } else if (originalType == "TruncatedNormal" && (attr.first == "seed" || attr.first == "seed2")) {
    (*tfAttr)[attr.first].set_i(i);
  } else {
    (*tfAttr)[attr.first].set_type((domi::tensorflow::DataType)i);
  }
}

void SetSqueezeDims(const string &originalType, google::protobuf::Map<string, domi::tensorflow::AttrValue> *tfAttr,
                    const pair<const string, ge::GeAttrValue> &attr, const vector<int> &intList,
                    const domi::tensorflow::AttrValue &attrValue, domi::tensorflow::AttrValue_ListValue *list) {
  if (originalType == "Squeeze" && (attr.first == "squeeze_dims")) {
    for (auto i : intList) {
      list->add_i(i);
    }
    (*tfAttr)[attr.first] = attrValue;
  }
}

void SetListIntAttr(const string &originalType, OpDescPtr &opDesc,
                    google::protobuf::Map<string, domi::tensorflow::AttrValue> *tfAttr,
                    const pair<const string, ge::GeAttrValue> &attr) {
  vector<int> intList;
  (void)AttrUtils::GetListInt(opDesc, attr.first, intList);

  domi::tensorflow::AttrValue attrValue;
  domi::tensorflow::AttrValue_ListValue *list = attrValue.mutable_list();

  vector<string>::iterator iter = std::find(is_dataset_op_vec.begin(), is_dataset_op_vec.end(), originalType);
  if (iter != is_dataset_op_vec.end()) {
    if (attr.first == "Toutput_types" || attr.first == "output_types") {
      for (auto i : intList) {
        list->add_type((domi::tensorflow::DataType)i);
      }
      (*tfAttr)[attr.first] = attrValue;
    } else if ((originalType == "ParallelMapDataset" || originalType == "FilterDataset" ||
                originalType == "MapAndBatchDatasetV2") &&
               attr.first == "Targuments") {
      (*tfAttr)[attr.first] = attrValue;
    }
  } else {
    SetSqueezeDims(originalType, tfAttr, attr, intList, attrValue, list);
  }
}

void SetListListIntAttr(const string &originalType, OpDescPtr &opDesc,
                        google::protobuf::Map<string, domi::tensorflow::AttrValue> *tfAttr,
                        const pair<const string, ge::GeAttrValue> &attr) {
  vector<vector<int64_t>> intListList;
  (void)AttrUtils::GetListListInt(opDesc, attr.first, intListList);

  domi::tensorflow::AttrValue attrValue;
  domi::tensorflow::AttrValue_ListValue *list = attrValue.mutable_list();

  if ((originalType == "IteratorV2" || originalType == "BatchDatasetV2" || originalType == "IteratorGetNext" ||
       originalType == "ParallelMapDataset" || originalType == "DeviceQueueDataset" || originalType == "QueueDataset" ||
       originalType == "FilterDataset" || originalType == "MapAndBatchDatasetV2") &&
      attr.first == "output_shapes") {
    for (size_t ill = 0; ill < intListList.size(); ill++) {
      TensorShapeProto *tensorShape = list->add_shape();
      auto intList_ = intListList[ill];
      for (auto i : intList_) {
        TensorShapeProto_Dim *dim = tensorShape->add_dim();
        dim->set_size(i);
      }
    }
    (*tfAttr)[attr.first] = attrValue;
  } else if (originalType == "TensorDataset" && attr.first == "output_shapes") {
    domi::tensorflow::TensorShapeProto *tensorShape = list->add_shape();
    domi::tensorflow::TensorShapeProto_Dim *dim = tensorShape->add_dim();
    dim->set_size(0);
    (*tfAttr)[attr.first] = attrValue;
  }
}

void SetTensorValue(const ge::ConstGeTensorPtr &geTensor, domi::tensorflow::TensorProto *tfTensor, int32_t dataCount) {
  if (dataCount > 1) {
    tfTensor->set_tensor_content(geTensor->GetData().data(), geTensor->GetData().size());
  } else {
    switch (geTensor->GetTensorDesc().GetDataType()) {
      case ge::DT_FLOAT: {
        float f = *(reinterpret_cast<const float *>(geTensor->GetData().data()));
        tfTensor->add_float_val(f);
        break;
      }

      case ge::DT_INT32: {
        int32_t i = *(reinterpret_cast<const int32_t *>(geTensor->GetData().data()));
        tfTensor->add_int_val(i);
        break;
      }

      case ge::DT_BOOL: {
        bool b = *(reinterpret_cast<const bool *>(geTensor->GetData().data()));
        tfTensor->add_bool_val(b);
        break;
      }

      case ge::DT_INT64: {
        int64_t i = *(reinterpret_cast<const int64_t *>(geTensor->GetData().data()));
        tfTensor->add_int64_val(i);
        break;
      }

      case ge::DT_FLOAT16: {
        int32_t f = *(reinterpret_cast<const int32_t *>(geTensor->GetData().data()));
        tfTensor->add_half_val(f);
        break;
      }

      default: {
        GELOGW("SetTensorValue not support the data type %s.",
               ge::TypeUtils::DataTypeToSerialString(geTensor->GetTensorDesc().GetDataType()).c_str());
      }
    }
  }
}

Status SetTensorAttr(ge::OpDescPtr &opDesc, google::protobuf::Map<string, domi::tensorflow::AttrValue> *tfAttr,
                     const pair<const string, ge::GeAttrValue> &attr) {
  ge::ConstGeTensorPtr ge_tensor;
  (void)ge::AttrUtils::GetTensor(opDesc, attr.first, ge_tensor);

  domi::tensorflow::TensorProto *tf_tensor = (*tfAttr)[attr.first].mutable_tensor();

  // Set datatype
  domi::tensorflow::DataType datatype;
  auto ge_datatype = ge_tensor->GetTensorDesc().GetDataType();
  int32_t data_count = 1;
  switch (ge_datatype) {
    case ge::DataType::DT_FLOAT:
      datatype = domi::tensorflow::DataType::DT_FLOAT;
      data_count = ge_tensor->GetData().size() / sizeof(float);
      break;
    case ge::DataType::DT_FLOAT16:
      datatype = domi::tensorflow::DataType::DT_HALF;
      data_count = ge_tensor->GetData().size() / sizeof(int16_t);
      break;
    case ge::DataType::DT_INT32:
      datatype = domi::tensorflow::DataType::DT_INT32;
      data_count = ge_tensor->GetData().size() / sizeof(int32_t);
      break;
    case ge::DataType::DT_INT64:
      datatype = domi::tensorflow::DataType::DT_INT64;
      data_count = ge_tensor->GetData().size() / sizeof(int64_t);
      break;
    case ge::DataType::DT_UINT8:
      datatype = domi::tensorflow::DataType::DT_UINT8;
      data_count = ge_tensor->GetData().size() / sizeof(uint8_t);
      break;
    case ge::DataType::DT_BOOL:
      datatype = domi::tensorflow::DataType::DT_BOOL;
      data_count = ge_tensor->GetData().size() / sizeof(bool);
      break;
    default:
      REPORT_INNER_ERROR("E19999", "datatype:%d of Attr:%s in node:%s:%s is not supported",
                         ge_datatype, attr.first.c_str(), opDesc->GetName().c_str(), opDesc->GetType().c_str());
      GELOGE(PARAM_INVALID, "NO SUPPORT datatype = %s", ge::TypeUtils::DataTypeToSerialString(ge_datatype).c_str());
      return PARAM_INVALID;
  }
  tf_tensor->set_dtype(datatype);

  domi::tensorflow::TensorShapeProto *tf_shape = tf_tensor->mutable_tensor_shape();
  for (auto dim : ge_tensor->GetTensorDesc().GetShape().GetDims()) {
    domi::tensorflow::TensorShapeProto_Dim *tf_dims = tf_shape->add_dim();
    tf_dims->set_size(dim);
  }

  SetTensorValue(ge_tensor, tf_tensor, data_count);
  return SUCCESS;
}

Status SetNodedefProto(domi::tensorflow::NodeDef &proto, ge::NodePtr n, string original_type) {
  GELOGI("graph_optimizer.cpp && SetNodedefProto");
  // Set proto head
  Status ret;
  auto op_desc = n->GetOpDesc();
  GELOGI("n->GetName =%s, original_type =%s", n->GetName().c_str(), original_type.c_str());
  proto.set_name(n->GetName());
  proto.set_op(original_type);

  // Set input
  auto input_names = op_desc->GetInputName();

  for (auto anchor : n->GetAllInDataAnchors()) {
    GE_IF_BOOL_EXEC(anchor == nullptr || anchor->GetPeerOutAnchor() == nullptr ||
                        anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr ||
                        anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr,
                    continue);
    OutDataAnchorPtr src_anchor = anchor->GetPeerOutAnchor();
    NodePtr src_node = anchor->GetPeerOutAnchor()->GetOwnerNode();
    OpDescPtr src_opdesc = src_node->GetOpDesc();
    GELOGI("inedge src:%s, src_out_index:%d, dst:%s, dst_in_index:%d", src_opdesc->GetName().c_str(),
           src_anchor->GetIdx(), op_desc->GetName().c_str(), anchor->GetIdx());
    string inputName;
    inputName = src_opdesc->GetName() + ":" + "output:" + std::to_string(src_anchor->GetIdx());
    GELOGI("inputName =%s\n", inputName.c_str());
    proto.add_input(inputName);
  }

  // Set device
  proto.set_device("CPU");

  // Set proto attr
  google::protobuf::Map<std::string, domi::tensorflow::AttrValue> *tf_attr = proto.mutable_attr();
  map<string, ge::GeAttrValue> allattrs = op_desc->GetAllAttrs();
  allattrs.erase(ge::ATTR_NAME_FRAMEWORK_FWK_TYPE);
  allattrs.erase(ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE);
  if (original_type == "Add") {
    allattrs.erase(ge::ATTR_NAME_MODE);
  } else if (original_type == "IteratorGetNext") {
    allattrs.erase("output_num");
  }

  for (const auto &attr : allattrs) {
    ge::GeAttrValue::ValueType v_t = attr.second.GetValueType();
    switch (v_t) {
      case ge::GeAttrValue::ValueType::VT_STRING: {
        SetStringAttr(original_type, op_desc, tf_attr, attr);

        break;
      }

      case ge::GeAttrValue::ValueType::VT_INT: {
        SetIntAttr(original_type, op_desc, tf_attr, attr);

        break;
      }
      case ge::GeAttrValue::ValueType::VT_BOOL: {
        bool i = false;
        (void)ge::AttrUtils::GetBool(op_desc, attr.first, i);
        (*tf_attr)[attr.first].set_b(i);
        break;
      }
      case ge::GeAttrValue::ValueType::VT_LIST_INT: {
        SetListIntAttr(original_type, op_desc, tf_attr, attr);

        break;
      }
      case ge::GeAttrValue::ValueType::VT_LIST_LIST_INT: {
        SetListListIntAttr(original_type, op_desc, tf_attr, attr);

        break;
      }
      case ge::GeAttrValue::ValueType::VT_TENSOR: {
        ret = SetTensorAttr(op_desc, tf_attr, attr);
        GE_IF_BOOL_EXEC(ret != SUCCESS, return ret);
        break;
      }
      default:
        break;
    }
  }

  return SUCCESS;
}

typedef Status (*PIOListHandle)(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                                ge::OpDescPtr &opDesc);

Status GatherV2IOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                      ge::OpDescPtr &opDesc) {
  int tparams;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "Tparams", tparams)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:Tparams from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get Tparams error.");
  int tindices;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "Tindices", tindices)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:Tindices from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get Tindices error.");
  int taxis;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "Taxis", taxis)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:Taxis from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get Taxis error.");

  // input_list - eg:{1, 3, 3}
  input_list.push_back(tparams);
  input_list.push_back(tindices);
  input_list.push_back(taxis);
  // output_list - eg:{3}
  output_list.push_back(tparams);

  return SUCCESS;
}

Status ConstIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                   ge::OpDescPtr &opDesc) {
  int dtype;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "dtype", dtype)), return PARAM_INVALID, "Get dtype error.");
  // output_list - {3}
  output_list.push_back(dtype);

  return SUCCESS;
}

Status MaxMinIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                    ge::OpDescPtr &opDesc) {
  int attrT;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", attrT)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get Tparams error.");

  // input_list
  input_list.push_back(attrT);
  input_list.push_back(attrT);

  // output_list
  output_list.push_back(attrT);

  return SUCCESS;
}

Status CastIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                  ge::OpDescPtr &opDesc) {
  int srcT;
  int dstT;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "SrcT", srcT)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:SrcT from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get srcT error.");
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "DstT", dstT)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:DstT from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get dstT error.");
  input_list.push_back(srcT);
  output_list.push_back(dstT);

  return SUCCESS;
}

Status AddIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list, ge::OpDescPtr &opDesc) {
  int type;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", type)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get T error.");

  input_list.push_back(type);
  input_list.push_back(type);

  output_list.push_back(type);

  return SUCCESS;
}

Status LessIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                  ge::OpDescPtr &opDesc) {
  int dtype;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", dtype)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get dtype error.");

  input_list.push_back(dtype);
  input_list.push_back(dtype);
  output_list.push_back(domi::tensorflow::DataType::DT_BOOL);

  return SUCCESS;
}

Status MulIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list, ge::OpDescPtr &opDesc) {
  int dataType;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, ge::ATTR_NAME_T, dataType)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ge::ATTR_NAME_T.c_str(),
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID,
                   "Get Tparams error.");

  input_list.push_back(dataType);
  input_list.push_back(dataType);

  output_list.push_back(dataType);

  return SUCCESS;
}

Status RealDivIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                     ge::OpDescPtr &opDesc) {
  int t;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", t)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get beta error.");

  input_list.push_back(t);
  input_list.push_back(t);

  output_list.push_back(t);

  return SUCCESS;
}

Status SelectIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                    ge::OpDescPtr &opDesc) {
  int t;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", t)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get e error.");

  input_list.push_back(domi::tensorflow::DataType::DT_BOOL);
  input_list.push_back(t);
  input_list.push_back(t);

  output_list.push_back(t);

  return SUCCESS;
}

Status SqrtIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                  ge::OpDescPtr &opDesc) {
  int dataType;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, ge::ATTR_NAME_T, dataType)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:%s from op:%s(%s) failed", ge::ATTR_NAME_T.c_str(),
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID,
                   "Get Tparam error.");

  input_list.push_back(dataType);

  output_list.push_back(dataType);

  return SUCCESS;
}

Status TruncatedNormalIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                             ge::OpDescPtr &opDesc) {
  int t;
  int dtype;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", t)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get T error.");
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "dtype", dtype)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:dtype from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get e error.");

  input_list.push_back(t);

  output_list.push_back(dtype);

  return SUCCESS;
}

Status PackIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                  ge::OpDescPtr &opDesc) {
  int t;
  int n;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", t)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get T error.");
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "N", n)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:N from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get N error.");

  for (int i = 0; i < n; i++) {
    input_list.push_back(t);
  }

  output_list.push_back(t);

  return SUCCESS;
}

Status DropOutGenMaskIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                            ge::OpDescPtr &opDesc) {
  input_list.push_back(domi::tensorflow::DT_INT64);
  input_list.push_back(domi::tensorflow::DT_FLOAT);
  output_list.push_back(domi::tensorflow::DT_UINT8);

  return SUCCESS;
}

Status ExpandDimsIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                        ge::OpDescPtr &opDesc) {
  int dataType;
  int dimType;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", dataType)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get T error.");
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "Tdim", dimType)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:Tdim from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get Tdim error.");
  // input_list - x y data type
  input_list.push_back(dataType);
  input_list.push_back(dimType);
  // output_list - z data type
  output_list.push_back(dataType);

  return SUCCESS;
}

Status SqueezeIOList(ge::GeAttrValue::LIST_INT &input_list, ge::GeAttrValue::LIST_INT &output_list,
                     ge::OpDescPtr &opDesc) {
  // Set - TENSORFLOW_IN_DATATYPE/TENSORFLOW_OUT_DATATYPE
  int dataType;
  vector<int> dimTypeList;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", dataType)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get T error.");
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetListInt(opDesc, "squeeze_dims", dimTypeList)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:squeeze_dims from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID,
                   "Get squeeze_dims error.");
  for (auto i : dimTypeList) {
    GELOGI("squeeze_dims = %d.\n", i);
  }

  // input_list - x y data type
  input_list.push_back(dataType);
  // output_list - z data type
  output_list.push_back(dataType);

  return SUCCESS;
}

Status TopKV2IOList(ge::GeAttrValue::LIST_INT &inputList, ge::GeAttrValue::LIST_INT &outputList,
                    ge::OpDescPtr &opDesc) {
  int t;
  GE_CHK_BOOL_EXEC((ge::AttrUtils::GetInt(opDesc, "T", t)),
                   REPORT_CALL_ERROR("E19999", "Get Attr:T from op:%s(%s) failed",
                                     opDesc->GetName().c_str(), opDesc->GetType().c_str());
                   return PARAM_INVALID, "Get T error.");

  // input_list - eg:{1, 3}
  inputList.push_back(t);
  inputList.push_back(domi::tensorflow::DataType::DT_INT32);

  // output_list - eg:{1, 3}
  outputList.push_back(t);
  outputList.push_back(domi::tensorflow::DataType::DT_INT32);

  return SUCCESS;
}

void CreateIOListFuncMap(map<string, PIOListHandle> &mOpIOListFuncMap) {
  mOpIOListFuncMap.insert({"GatherV2", GatherV2IOList});
  mOpIOListFuncMap.insert({"Const", ConstIOList});
  mOpIOListFuncMap.insert({"Maximum", MaxMinIOList});
  mOpIOListFuncMap.insert({"Minimum", MaxMinIOList});
  mOpIOListFuncMap.insert({"Cast", CastIOList});
  mOpIOListFuncMap.insert({"Add", AddIOList});
  mOpIOListFuncMap.insert({"Less", LessIOList});
  mOpIOListFuncMap.insert({"Mul", MulIOList});
  mOpIOListFuncMap.insert({"RealDiv", RealDivIOList});
  mOpIOListFuncMap.insert({"Select", SelectIOList});
  mOpIOListFuncMap.insert({"TruncatedNormal", TruncatedNormalIOList});
  mOpIOListFuncMap.insert({"Pack", PackIOList});
  mOpIOListFuncMap.insert({"DropOutGenMask", DropOutGenMaskIOList});
  mOpIOListFuncMap.insert({"ExpandDims", ExpandDimsIOList});
  mOpIOListFuncMap.insert({"Squeeze", SqueezeIOList});
  mOpIOListFuncMap.insert({"TopKV2", TopKV2IOList});
}

Status CreateNodeDefBytes(ge::NodePtr n, string originalType, map<string, PIOListHandle> &mOpIOListFuncMap) {
  Status ret;
  auto opDesc = n->GetOpDesc();
  GELOGI("n->GetName() = %s.\n", n->GetName().c_str());
  // Set - NodeDef PROTO
  domi::tensorflow::NodeDef proto;
  ge::GeAttrValue::LIST_INT inputList;
  ge::GeAttrValue::LIST_INT outputList;
  ret = SetNodedefProto(proto, n, originalType);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "SetNodedefProto failed.");

  // Set inputList & outputList
  // Set - TENSORFLOW_IN_DATATYPE/TENSORFLOW_OUT_DATATYPE
  PIOListHandle funcPtr = nullptr;
  map<string, PIOListHandle>::iterator it = mOpIOListFuncMap.find(originalType);
  if (it != mOpIOListFuncMap.end()) {
    funcPtr = it->second;
  }

  if (funcPtr != nullptr) {
    ret = ((PIOListHandle)funcPtr)(inputList, outputList, opDesc);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  vector<string>::iterator iter = std::find(is_dataset_op_vec.begin(), is_dataset_op_vec.end(), originalType);
  if (iter == is_dataset_op_vec.end()) {
    (void)ge::AttrUtils::SetListInt(opDesc, ge::T_IN_DATATYPE, inputList);
    (void)ge::AttrUtils::SetListInt(opDesc, ge::T_OUT_DATATYPE, outputList);
  }

  // Set size
  for (auto ge_desc : opDesc->GetAllOutputsDescPtr()) {
    int64_t real_size = 1;
    int64_t tmp_dim = 0;
    auto data_type = ge_desc->GetDataType();

    uint32_t size_type = 1;
    bool type_ret = ge::TypeUtils::GetDataTypeLength(data_type, size_type);
    GE_IF_BOOL_EXEC(!type_ret,
                    REPORT_CALL_ERROR("E19999", "Can't get DataType:%s length of op:%s(%s)",
                                      ge::TypeUtils::DataTypeToSerialString(data_type).c_str(),
                                      n->GetName().c_str(), n->GetType().c_str());
                    GELOGE(PARAM_INVALID, "Can't GetDataTypeLength of data_type: %s",
                           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
                    return PARAM_INVALID);

    // calculate size
    for (uint32_t j = 0; j < ge_desc->GetShape().GetDimNum(); ++j) {
      tmp_dim = ge_desc->GetShape().GetDim(j);
      GE_CHECK_GE(tmp_dim, 0);
      PARSER_INT64_MULCHECK(real_size, tmp_dim);
      real_size *= tmp_dim;
    }
    ge::TensorUtils::SetSize(*ge_desc, real_size * size_type);
    ge::TensorUtils::SetRealDimCnt(*ge_desc, ge_desc->GetShape().GetDimNum());
  }

  // Serial - nodedef proto
  string nodefStr;
  GE_IF_BOOL_EXEC(!proto.SerializeToString(&nodefStr),
                  REPORT_CALL_ERROR("E19999", "Serialize nodedef to string failed, op:%s(%s)",
                                    n->GetName().c_str(), n->GetType().c_str());
                  GELOGE(PARAM_INVALID, "Serialize nodedef to string failed.");
                  return PARAM_INVALID);

  // Set - ATTR_NAME_FRAMEWORK_NODE_DEF
  ge::GeAttrValue::BYTES nodeDefBytes;
  (void)ge::AttrUtils::SetZeroCopyBytes(
      opDesc, ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
      nodeDefBytes.CopyFrom(reinterpret_cast<const uint8_t *>(nodefStr.data()), nodefStr.length()));

  // print proto
  string nodefstr;
  google::protobuf::TextFormat::PrintToString(proto, &nodefstr);
  GELOGI("---> ! CreateNodeDefBytes() nodefstr : %s", nodefstr.c_str());
  return SUCCESS;
}

Status CreateOpDefBytes(ge::NodePtr n, string original_type) {
  auto opDesc = n->GetOpDesc();
  GELOGI("n->GetName() =%s, original_type =%s", n->GetName().c_str(), original_type.c_str());

  // Set - OpDef PROTO
  domi::tensorflow::OpDef proto;
  proto.set_name(original_type);

  if (original_type == "Const") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("output");
    outputArgdef->set_type_attr("dtype");

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("value");
    attr1->set_type("tensor");

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("dtype");
    attr2->set_type("type");
  } else if (original_type == "TensorDataset") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef = proto.add_input_arg();
    inputArgdef->set_name("components");
    inputArgdef->set_type_list_attr("Toutput_types");

    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("Toutput_types");
    attr1->set_type("list(type)");
    attr1->set_has_minimum(true);
    attr1->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_shapes");
    attr2->set_type("list(shape)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    // Set stateful
    proto.set_is_stateful(true);
  } else if (original_type == "QueueDataset") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef = proto.add_input_arg();
    inputArgdef->set_name("input_dataset");
    inputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("sourcedata");
    attr1->set_type("string");

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_types");
    attr2->set_type("list(type)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("output_shapes");
    attr3->set_type("list(shape)");
    attr3->set_has_minimum(true);
    attr3->set_minimum(1);

    // Set stateful
    proto.set_is_stateful(true);
  } else if (original_type == "DeviceQueueDataset") {
    // Set output_arg
    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("channel_name");
    attr1->set_type("string");

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_types");
    attr2->set_type("list(type)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("output_shapes");
    attr3->set_type("list(shape)");
    attr3->set_has_minimum(true);
    attr3->set_minimum(1);

    // Set stateful
    proto.set_is_stateful(true);
  } else if (original_type == "ParallelMapDataset") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef1 = proto.add_input_arg();
    inputArgdef1->set_name("input_dataset");
    inputArgdef1->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    domi::tensorflow::OpDef::ArgDef *inputArgdef2 = proto.add_input_arg();
    inputArgdef2->set_name("other_arguments");
    inputArgdef2->set_type_list_attr("Targuments");

    domi::tensorflow::OpDef::ArgDef *inputArgdef3 = proto.add_input_arg();
    inputArgdef3->set_name("num_parallel_calls");
    inputArgdef3->set_type(::domi::tensorflow::DataType::DT_INT32);

    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr0 = proto.add_attr();
    attr0->set_name("f");
    attr0->set_type("func");

    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("Targuments");
    attr1->set_type("list(type)");
    attr1->set_has_minimum(true);

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_types");
    attr2->set_type("list(type)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("output_shapes");
    attr3->set_type("list(shape)");
    attr3->set_has_minimum(true);
    attr3->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr4 = proto.add_attr();
    attr4->set_name("use_iter_op_parallelism");
    attr4->set_type("bool");
    ::domi::tensorflow::AttrValue *default_value = attr4->mutable_default_value();
    default_value->set_b(true);
  } else if (original_type == "BatchDatasetV2") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef1 = proto.add_input_arg();
    inputArgdef1->set_name("input_dataset");
    inputArgdef1->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    domi::tensorflow::OpDef::ArgDef *inputArgdef2 = proto.add_input_arg();
    inputArgdef2->set_name("batch_size");
    inputArgdef2->set_type(::domi::tensorflow::DataType::DT_INT64);

    domi::tensorflow::OpDef::ArgDef *inputArgdef3 = proto.add_input_arg();
    inputArgdef3->set_name("drop_remainder");
    inputArgdef3->set_type(::domi::tensorflow::DataType::DT_BOOL);

    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("output_types");
    attr1->set_type("list(type)");
    attr1->set_has_minimum(true);
    attr1->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_shapes");
    attr2->set_type("list(shape)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);
  } else if (original_type == "IteratorV2") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_RESOURCE);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("shared_name");
    attr1->set_type("string");

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("container");
    attr2->set_type("string");

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("output_types");
    attr3->set_type("list(type)");
    attr3->set_has_minimum(true);
    attr3->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr4 = proto.add_attr();
    attr4->set_name("output_shapes");
    attr4->set_type("list(shape)");
    attr4->set_has_minimum(true);
    attr4->set_minimum(1);

    // Set stateful
    proto.set_is_stateful(true);
  } else if (original_type == "MakeIterator") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef1 = proto.add_input_arg();
    inputArgdef1->set_name("dataset");
    inputArgdef1->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    domi::tensorflow::OpDef::ArgDef *inputArgdef2 = proto.add_input_arg();
    inputArgdef2->set_name("iterator");
    inputArgdef2->set_type(::domi::tensorflow::DataType::DT_RESOURCE);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("_kernel");
    attr1->set_type("dp");

    // Set stateful
    proto.set_is_stateful(true);
  } else if (original_type == "IteratorGetNext") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *input_argdef_1 = proto.add_input_arg();
    input_argdef_1->set_name("iterator");
    input_argdef_1->set_type(::domi::tensorflow::DataType::DT_RESOURCE);

    domi::tensorflow::OpDef::ArgDef *output_argdef = proto.add_output_arg();
    output_argdef->set_name("components");
    output_argdef->set_type_list_attr("output_types");

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("output_types");
    attr1->set_type("list(type)");
    attr1->set_has_minimum(true);
    attr1->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_shapes");
    attr2->set_type("list(shape)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("_kernel");
    attr3->set_type("dp");

    // Set stateful
    proto.set_is_stateful(true);
  } else if (original_type == "FilterDataset") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef1 = proto.add_input_arg();
    inputArgdef1->set_name("input_dataset");
    inputArgdef1->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    domi::tensorflow::OpDef::ArgDef *inputArgdef2 = proto.add_input_arg();
    inputArgdef2->set_name("other_arguments");
    inputArgdef2->set_type_list_attr("Targuments");

    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr0 = proto.add_attr();
    attr0->set_name("predicate");
    attr0->set_type("func");

    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("Targuments");
    attr1->set_type("list(type)");
    attr1->set_has_minimum(true);

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_types");
    attr2->set_type("list(type)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("output_shapes");
    attr3->set_type("list(shape)");
    attr3->set_has_minimum(true);
    attr3->set_minimum(1);
  } else if (original_type == "MapAndBatchDatasetV2") {
    // Set input_arg & output_arg
    domi::tensorflow::OpDef::ArgDef *inputArgdef1 = proto.add_input_arg();
    inputArgdef1->set_name("input_dataset");
    inputArgdef1->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    domi::tensorflow::OpDef::ArgDef *inputArgdef2 = proto.add_input_arg();
    inputArgdef2->set_name("other_arguments");
    inputArgdef2->set_type_list_attr("Targuments");

    domi::tensorflow::OpDef::ArgDef *inputArgdef3 = proto.add_input_arg();
    inputArgdef3->set_name("batch_size");
    inputArgdef3->set_type(::domi::tensorflow::DataType::DT_INT64);

    domi::tensorflow::OpDef::ArgDef *inputArgdef4 = proto.add_input_arg();
    inputArgdef4->set_name("num_parallel_calls");
    inputArgdef4->set_type(::domi::tensorflow::DataType::DT_INT64);

    domi::tensorflow::OpDef::ArgDef *inputArgdef5 = proto.add_input_arg();
    inputArgdef5->set_name("drop_remainder");
    inputArgdef5->set_type(::domi::tensorflow::DataType::DT_BOOL);

    domi::tensorflow::OpDef::ArgDef *outputArgdef = proto.add_output_arg();
    outputArgdef->set_name("handle");
    outputArgdef->set_type(::domi::tensorflow::DataType::DT_VARIANT);

    // Set domi::AttrDef
    domi::tensorflow::OpDef_AttrDef *attr0 = proto.add_attr();
    attr0->set_name("f");
    attr0->set_type("func");

    domi::tensorflow::OpDef_AttrDef *attr1 = proto.add_attr();
    attr1->set_name("Targuments");
    attr1->set_type("list(type)");
    attr1->set_has_minimum(true);

    domi::tensorflow::OpDef_AttrDef *attr2 = proto.add_attr();
    attr2->set_name("output_types");
    attr2->set_type("list(type)");
    attr2->set_has_minimum(true);
    attr2->set_minimum(1);

    domi::tensorflow::OpDef_AttrDef *attr3 = proto.add_attr();
    attr3->set_name("output_shapes");
    attr3->set_type("list(shape)");
    attr3->set_has_minimum(true);
    attr3->set_minimum(1);
  }
  // set - opdef
  string opdefString;
  GE_IF_BOOL_EXEC(!proto.SerializeToString(&opdefString),
                  REPORT_CALL_ERROR("E19999", "Serialize opdef to string failed, op:%s(%s)",
                                    n->GetName().c_str(), n->GetType().c_str());
                  GELOGE(PARAM_INVALID, "Serialize opdef to string failed.");
                  return PARAM_INVALID);

  (void)ge::AttrUtils::SetStr(opDesc, ge::ATTR_NAME_FRAMEWORK_OP_DEF, opdefString);

  // print proto
  string opdefstr;
  google::protobuf::TextFormat::PrintToString(proto, &opdefstr);
  GELOGI("---> ! CreateOpDefBytes() opdefstr  :\n");
  GELOGI("%s", opdefstr.c_str());
  return SUCCESS;
}

Status CreateFuncDefBytes(ge::NodePtr n, string original_type, string func_bin_path) {
  GELOGI("func_bin_path = %s", func_bin_path.c_str());
  auto opDesc = n->GetOpDesc();

  std::string func_string;
  if (original_type == "ParallelMapDataset" || original_type == "MapAndBatchDatasetV2") {
    GE_LOGI_IF(ge::AttrUtils::GetStr(n->GetOpDesc(), "f", func_string) != true, "get func string failed.");
  } else if (original_type == "FilterDataset") {
    GE_LOGI_IF(ge::AttrUtils::GetStr(n->GetOpDesc(), "predicate", func_string) != true, "get func string failed.");
  }
  GELOGI("func_string = %s", func_string.c_str());

  std::string file = func_bin_path + "/" + func_string + ".bin";
  GELOGI("file = %s", file.c_str());

  char *buf = nullptr;
  int32_t len = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(!ge::parser::ReadBytesFromBinaryFile(file.c_str(), &buf, len),
                                 REPORT_CALL_ERROR("E19999", "Read bytes from file:%s failed", file.c_str());
                                 return false,
                                 "read bytes file error!");

  GELOGI("len =%d\n", len);

  ge::GeAttrValue::BYTES funcDefBytes;
  funcDefBytes = ge::Buffer::CopyFrom((std::uint8_t *)buf, len);
  (void)ge::AttrUtils::SetBytes(opDesc, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF, funcDefBytes);
  GELOGI("funcDefBytes.GetSize() =%zu", funcDefBytes.GetSize());

  // print proto
  if (funcDefBytes.GetSize() > 0 && funcDefBytes.GetData() != nullptr) {
    domi::tensorflow::FunctionDefLibrary funcdeflib;
    (void)funcdeflib.ParseFromArray(funcDefBytes.GetData(), funcDefBytes.GetSize());

    string funcdeflibrarystr;
    google::protobuf::TextFormat::PrintToString(funcdeflib, &funcdeflibrarystr);
    GELOGI("---> !CreateFuncDefBytes() funcdeflibrarystr :");
  }

  delete[] buf;
  return SUCCESS;
}

Status ParserGraphOptimizer::MakeTfProtoDef() {
  GE_CHK_STATUS_RET(graph_->TopologicalSorting(), "graph sort failed.");

  map<string, PIOListHandle> mOpIOListFuncMap;
  CreateIOListFuncMap(mOpIOListFuncMap);

  for (ge::NodePtr n : graph_->GetDirectNode()) {
    if (n->GetType() != ge::parser::FRAMEWORKOP) continue;
    std::string original_type;
    GE_LOGI_IF(ge::AttrUtils::GetStr(n->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type) != true,
               "get original type failed.");

    // create frameworkop nodedefbytes & TFindatatype & TFoutdatatype
    vector<string>::iterator iter =
        std::find(local_framework_op_vec.begin(), local_framework_op_vec.end(), original_type);
    if (iter != local_framework_op_vec.end()) {
      Status ret = CreateNodeDefBytes(n, original_type, mOpIOListFuncMap);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "Create nodedefBytes failed!");

      vector<string>::iterator iter_dataset =
          std::find(is_dataset_op_vec.begin(), is_dataset_op_vec.end(), original_type);
      if (original_type == "Const" || iter_dataset != is_dataset_op_vec.end()) {
        ret = CreateOpDefBytes(n, original_type);
        GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "Create opdefBytes failed!");
        if (original_type == "ParallelMapDataset" || original_type == "FilterDataset" ||
            original_type == "MapAndBatchDatasetV2") {
          ret = CreateFuncDefBytes(n, original_type, GetFuncBinPath());
          GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, ret, "Create funcdefBytes failed!");
        }
      }
    }
  }

  return SUCCESS;
}

Status ParserGraphOptimizer::FusionFmkop() {
  GELOGI("graph_optimizer.cpp && FustionFmkop()");
  GELOGI("GetLocalFmkopFlag() =%d", GetLocalFmkopFlag());
  GE_IF_BOOL_EXEC(GetLocalFmkopFlag() == 1, MakeTfProtoDef());

  GE_CHECK_NOTNULL(graph_);
  std::unordered_map<string, std::vector<NodePtr>> node_cluser_Map;
  GE_CHK_STATUS_RET(MarkForFusion(node_cluser_Map), "find framework node to be fused fail.");
  GE_IF_BOOL_EXEC(node_cluser_Map.size() == 0, return SUCCESS);

  for (auto it = node_cluser_Map.begin(); it != node_cluser_Map.end(); ++it) {
    GE_CHK_STATUS_RET(UpdateGraph(it->second), "fusion framework nodes failed. node：%s", (it->first).c_str());
  }
  // fuse all fmkop and then delete node
  for (auto it = node_cluser_Map.begin(); it != node_cluser_Map.end(); ++it) {
    for (auto node : it->second) {
      GE_CHK_STATUS_RET(GraphUtils::IsolateNode(node, {}), "Isolate removed node: %s, type: %s failed",
                        node->GetName().c_str(), node->GetType().c_str());
      GE_CHK_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph_, node),
                        "Remove node: %s, type: %s without relink failed", node->GetName().c_str(),
                        node->GetType().c_str());
    }
  }

  return SUCCESS;
}

Status ParserGraphOptimizer::MarkForFusion(unordered_map<string, vector<NodePtr>> &node_cluser_Map) {
  GE_CHECK_NOTNULL(graph_);
  bool hasGetNext = false;
  for (auto node : graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != ge::parser::FRAMEWORK_OP_TYPE, continue);
    string type = "";
    GE_CHK_STATUS_RET(ge::parser::GetOriginalType(node, type));
    if (type == "IteratorGetNext") {
      hasGetNext = true;
      break;
    }
  }
  for (auto node : graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_IF_BOOL_EXEC(node->GetOpDesc()->GetType() != ge::parser::FRAMEWORK_OP_TYPE, continue)
    string type = "";
    GE_CHK_STATUS_RET(ge::parser::GetOriginalType(node, type));
    if (type == "IteratorGetNext") {
      vector<NodePtr> temp_node_cluser;
      for (auto in_anchor : node->GetAllInDataAnchors()) {
        OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
        GE_CHECK_NOTNULL(peer_out_anchor);
        NodePtr src_node = peer_out_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(src_node);
        temp_node_cluser.push_back(src_node);
      }
      temp_node_cluser.push_back(node);
      for (auto out_anchor : node->GetAllOutDataAnchors()) {
        GE_CHECK_NOTNULL(out_anchor);
        for (auto in_anchor : out_anchor->GetPeerInDataAnchors()) {
          GE_CHECK_NOTNULL(in_anchor);
          NodePtr dst_node = in_anchor->GetOwnerNode();
          GE_CHECK_NOTNULL(dst_node);
          GE_CHECK_NOTNULL(dst_node->GetOpDesc());
          if (dst_node->GetOpDesc()->GetType() == kShapeNodeName) {
            temp_node_cluser.emplace_back(dst_node);
          }
        }
      }
      if (temp_node_cluser.size() > 1) {
        vector<NodePtr> node_cluser;
        node_cluser.assign(temp_node_cluser.begin(), temp_node_cluser.end());
        node_cluser_Map[temp_node_cluser[0]->GetName()] = node_cluser;
      }
      temp_node_cluser.clear();
      GELOGI("MarkForFusion, IteratorGetNext graph mark success.");
    }

    if (!hasGetNext && (type == "Iterator" || type == "IteratorV2")) {
      GE_CHK_STATUS_RET(FindFmkNodeCluser(node_cluser_Map), "find framework node to be fused fail.");
      GELOGI("MarkForFusion, Iterator init graph mark success.");
    }
  }
  return SUCCESS;
}

// find frameworkOP
Status ParserGraphOptimizer::FindFmkNodeCluser(unordered_map<string, vector<NodePtr>> &node_cluser_Map) {
  vector<NodePtr> temp_node_cluser;

  for (auto node : graph_->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr temp_node_desc_ptr = node->GetOpDesc();
    GE_CHECK_NOTNULL(temp_node_desc_ptr);
    GE_IF_BOOL_EXEC(temp_node_desc_ptr->GetType() == ge::parser::DATA_TYPE, continue);

    if (temp_node_desc_ptr->GetType() == ge::parser::FRAMEWORK_OP_TYPE &&
        (temp_node_desc_ptr->GetName().find(RRTVAL_NODE_NAME_SUFFIX) == string::npos)) {
      temp_node_cluser.push_back(node);
    } else {
      if (temp_node_cluser.size() > 1) {
        vector<NodePtr> node_cluser;
        node_cluser.assign(temp_node_cluser.begin(), temp_node_cluser.end());
        node_cluser_Map[temp_node_cluser[0]->GetName()] = node_cluser;
      }
      temp_node_cluser.clear();
    }
  }
  if (temp_node_cluser.size() > 1) {
    vector<NodePtr> node_cluser;
    node_cluser.assign(temp_node_cluser.begin(), temp_node_cluser.end());
    node_cluser_Map[temp_node_cluser[0]->GetName()] = node_cluser;
  }
  return SUCCESS;
}

Status CollectNodeFuncs(vector<ge::NodePtr> &nodes, FunctionDefLibrary *library) {
  for (auto node : nodes) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr opDef = node->GetOpDesc();
    string funcdefStr;
    ge::GeAttrValue::BYTES funcDefBytes;

    GE_IF_BOOL_EXEC(
        AttrUtils::GetBytes(opDef, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF, funcDefBytes), FunctionDefLibrary funcLib;
        GE_CHECK_NOTNULL(funcDefBytes.GetData());
        string str(reinterpret_cast<char *>(funcDefBytes.GetData()), funcDefBytes.GetSize());
        GELOGI("FUNCDEF: Get function -> %s.", str.c_str()); GE_IF_BOOL_EXEC(
            funcLib.ParseFromArray(funcDefBytes.GetData(), funcDefBytes.GetSize()), library->MergeFrom(funcLib)));
  }
  return SUCCESS;
}

Status ParserGraphOptimizer::UpdateGraph(vector<NodePtr> &nodes) {
  ComputeGraphPtr sub_graph = nullptr;
  GE_MAKE_SHARED(sub_graph = std::make_shared<ComputeGraph>("subGraph"), sub_graph = nullptr; return PARAM_INVALID);

  unordered_map<string, NodePtr> node_map;
  vector<InDataAnchorPtr> input_anchors;
  vector<OutDataAnchorPtr> output_anchors;
  map<OutDataAnchorPtr, vector<InDataAnchorPtr>> output_in_map;
  vector<InControlAnchorPtr> input_control_anchors;
  vector<OutControlAnchorPtr> output_control_anchors;

  GE_CHK_STATUS_RET(InsertNode(sub_graph, nodes, input_anchors, output_anchors, output_in_map, input_control_anchors,
                               output_control_anchors, node_map),
                    "insert node to sub_graph failed.");
  GE_CHK_STATUS_RET(LinkInnerAnchor(node_map), "Link inner anchor failed.");

  std::unique_ptr<NodeDef> node_def(new (std::nothrow) NodeDef());  // tensorflow NodeDef
  GE_CHECK_NOTNULL(node_def);
  std::unique_ptr<FunctionDefLibrary> func_def_lib(new (std::nothrow) FunctionDefLibrary());
  GE_CHECK_NOTNULL(func_def_lib);
  // convert graph to FunctionDef
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(nodes.size() == 0,
                                 REPORT_INNER_ERROR("E19999", "Param nodes size must greater than 0");
                                 return PARAM_INVALID, "node size must greater than 0 .");
  GE_CHK_STATUS_RET(CollectNodeFuncs(nodes, func_def_lib.get()), "Collect functionDef in nodes failed.");
  GE_CHK_STATUS_RET(GraphToFunctionDef::BuildFunctionDef(sub_graph, nodes[0]->GetName(), func_def_lib.get(),
                                                         node_def.get(), input_anchors, output_anchors),
                    "Build functiondef failed.");
  string nodefStr;
  string funcdefStr;

  GE_IF_BOOL_EXEC(!node_def->SerializeToString(&nodefStr),
                  REPORT_CALL_ERROR("E19999", "Serialize nodedef to string failed");
                  GELOGE(PARAM_INVALID, "Serialize nodedef to string failed.");
                  return PARAM_INVALID);

  GE_IF_BOOL_EXEC(!func_def_lib->SerializeToString(&funcdefStr),
                  REPORT_CALL_ERROR("E19999", "Serialize func_def to string failed, ");
                  GELOGE(PARAM_INVALID, "Serialize func_def to string failed.");
                  return PARAM_INVALID);

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(nodes.size() == 0, return PARAM_INVALID, "nodes is empty.");

  OpDescPtr fusion_node_opdef = nullptr;
  GE_MAKE_SHARED(
      fusion_node_opdef = std::make_shared<OpDesc>(nodes[0]->GetOpDesc()->GetName(), nodes[0]->GetOpDesc()->GetType()),
      fusion_node_opdef = nullptr;
      return FAILED);

  std::string type = "";
  GE_CHK_STATUS_RET(ge::parser::GetOriginalType(nodes[0], type));
  (void)AttrUtils::SetStr(fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);

  (void)AttrUtils::SetZeroCopyBytes(
      fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF,
      Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(funcdefStr.data()), funcdefStr.length()));
  (void)AttrUtils::SetZeroCopyBytes(
      fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_NODE_DEF,
      Buffer::CopyFrom(reinterpret_cast<const uint8_t *>(nodefStr.data()), nodefStr.length()));

  (void)AttrUtils::SetInt(fusion_node_opdef, ge::ATTR_NAME_FRAMEWORK_FWK_TYPE, ge::GetParserContext().type);

  // reconstruct fusion_node and edges
  GE_CHK_STATUS_RET(RebuildOutputAnchors(output_anchors, fusion_node_opdef),
                    "rebuild output edges to fusion node failed.")
  GE_CHK_STATUS_RET(RebuildInputAnchors(input_anchors, fusion_node_opdef),
                    "rebuild input edges to fusion node failed.");
  NodePtr fusion_node = graph_->AddNode(fusion_node_opdef);

  // add Anchors
  GE_CHK_STATUS_RET(RebuildFusionNode(input_anchors, output_anchors, output_in_map, input_control_anchors,
                                      output_control_anchors, fusion_node),
                    "rebuild node failed!");

  return SUCCESS;
}

Status ParserGraphOptimizer::InsertNode(ge::ComputeGraphPtr sub_graph, vector<ge::NodePtr> &nodes,
                                        vector<ge::InDataAnchorPtr> &input_anchors,
                                        vector<ge::OutDataAnchorPtr> &output_anchors,
                                        map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                                        vector<ge::InControlAnchorPtr> &input_control_anchors,
                                        vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                        unordered_map<string, ge::NodePtr> &node_map) {
  GE_CHECK_NOTNULL(sub_graph);
  for (NodePtr node : nodes) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr op_def = node->GetOpDesc();
    NodePtr new_node = sub_graph->AddNode(op_def);
    node_map[node->GetName()] = new_node;

    // Input
    for (auto in_anchor : node->GetAllInDataAnchors()) {  // data
      OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_out_anchor->GetOwnerNode());
      GE_IF_BOOL_EXEC(iter == nodes.end(), input_anchors.emplace_back(in_anchor));
    }
    // Output
    for (auto out_anchor : node->GetAllOutDataAnchors()) {
      bool hasOutNode = false;
      // data anchor
      for (auto peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_in_anchor->GetOwnerNode());
        GE_IF_BOOL_EXEC(iter == nodes.end(), output_in_map[out_anchor].emplace_back(peer_in_anchor); hasOutNode = true);
      }
      GE_IF_BOOL_EXEC(hasOutNode == true, output_anchors.emplace_back(out_anchor));
    }

    InControlAnchorPtr node_in_control = node->GetInControlAnchor();
    GE_IF_BOOL_EXEC(
        node_in_control != nullptr, for (auto peer_out_anchor
                                         : node_in_control->GetPeerOutControlAnchors()) {
          vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_out_anchor->GetOwnerNode());
          GE_IF_BOOL_EXEC(iter == nodes.end(), input_control_anchors.emplace_back(node_in_control));
        });
    OutControlAnchorPtr node_out_control = node->GetOutControlAnchor();
    GE_IF_BOOL_EXEC(
        node_out_control != nullptr, for (auto peer_in_control_anchor
                                          : node_out_control->GetPeerInControlAnchors()) {
          vector<ge::NodePtr>::iterator iter = find(nodes.begin(), nodes.end(), peer_in_control_anchor->GetOwnerNode());
          GE_IF_BOOL_EXEC(iter == nodes.end(), output_control_anchors.emplace_back(node_out_control));
        });
  }
  return SUCCESS;
}

Status ParserGraphOptimizer::LinkInnerAnchor(unordered_map<string, ge::NodePtr> &node_map) {
  for (auto node : graph_->GetDirectNode()) {
    GE_IF_BOOL_EXEC(node_map.count(node->GetName()) == 0, continue);
    NodePtr dst = node_map[node->GetName()];
    for (auto in_anchor : node->GetAllInDataAnchors()) {
      OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      GE_IF_BOOL_EXEC(node_map.count(peer_out_anchor->GetOwnerNode()->GetName()) == 0, continue);
      NodePtr src = node_map[peer_out_anchor->GetOwnerNode()->GetName()];

      GE_IF_BOOL_EXEC(ge::GraphUtils::AddEdge(src->GetOutDataAnchor(peer_out_anchor->GetIdx()),
                                              dst->GetInDataAnchor(in_anchor->GetIdx())) != GRAPH_SUCCESS,
                      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                        src->GetName().c_str(), src->GetType().c_str(), peer_out_anchor->GetIdx(),
                                        dst->GetName().c_str(), dst->GetType().c_str(), in_anchor->GetIdx());
                      GELOGE(FAILED,
                             "LinkInnerAnchor Link data anchor failed, src node: %s, "
                             "dst node: %s.",
                             src->GetName().c_str(), dst->GetName().c_str());
                      return FAILED);
    }

    InControlAnchorPtr node_in_control = node->GetInControlAnchor();
    GE_IF_BOOL_EXEC(
        node_in_control != nullptr, for (auto peer_out_ctl_anchor
                                         : node_in_control->GetPeerOutControlAnchors()) {
          GE_IF_BOOL_EXEC(node_map.count(peer_out_ctl_anchor->GetOwnerNode()->GetName()) == 0, continue);
          NodePtr src_ctrl = node_map[peer_out_ctl_anchor->GetOwnerNode()->GetName()];
          GE_IF_BOOL_EXEC(
              ge::GraphUtils::AddEdge(src_ctrl->GetOutControlAnchor(), dst->GetInControlAnchor()) != GRAPH_SUCCESS,
              REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                                src_ctrl->GetName().c_str(), src_ctrl->GetType().c_str(),
                                dst->GetName().c_str(), dst->GetType().c_str());
              GELOGE(FAILED,
                     "LinkInnerAnchor Link control anchor failed, src node: "
                     "%s, dst node: %s.",
                     src_ctrl->GetName().c_str(), dst->GetName().c_str());
              return FAILED);
        });
  }
  return SUCCESS;
}

// rebuild output anchor
Status ParserGraphOptimizer::RebuildOutputAnchors(vector<ge::OutDataAnchorPtr> &output_anchors,
                                                  ge::OpDescPtr fusion_op_desc) {
  ge::GeAttrValue::LIST_INT output_list;
  GE_CHECK_NOTNULL(fusion_op_desc);

  // create input desc
  for (auto out_anchor : output_anchors) {
    NodePtr src_node = out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(src_node);

    GeTensorDesc src_out_desc = src_node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
    GE_CHK_BOOL_EXEC(fusion_op_desc->AddOutputDesc(src_out_desc) == ge::GRAPH_SUCCESS, return FAILED);

    ge::DataType data_type = src_out_desc.GetDataType();
    auto iter = GE_TENSORFLOW_DATA_TYPE_MAP.find((int32_t)data_type);
    GE_IF_BOOL_EXEC(
        iter == GE_TENSORFLOW_DATA_TYPE_MAP.end(),
        REPORT_INNER_ERROR("E19999", "datatype:%d of output:%d in node:%s:%s is not supported",
                           data_type, out_anchor->GetIdx(), src_node->GetName().c_str(), src_node->GetName().c_str());
        GELOGE(PARAM_INVALID, "data_type %s not supported", ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
        return PARAM_INVALID);

    int32_t dtype = iter->second;
    output_list.push_back((int64_t)dtype);
    GELOGI("FUNCDEF: output_list push_back  %d.", dtype);
  }
  GE_IF_BOOL_EXEC(!output_list.empty(), (void)AttrUtils::SetListInt(fusion_op_desc, ge::T_OUT_DATATYPE, output_list));

  return SUCCESS;
}
// rebuild input desc
Status ParserGraphOptimizer::RebuildInputAnchors(vector<ge::InDataAnchorPtr> &input_anchors,
                                                 ge::OpDescPtr fusion_op_desc) {
  ge::GeAttrValue::LIST_INT input_list;
  GE_CHECK_NOTNULL(fusion_op_desc);
  // add input desc
  for (auto in_anchor : input_anchors) {
    NodePtr dst_node = in_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(dst_node);

    auto tensorDescPtr = dst_node->GetOpDesc()->GetInputDescPtr(in_anchor->GetIdx());
    GE_CHECK_NOTNULL_EXEC(tensorDescPtr, return domi::FAILED);

    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((fusion_op_desc->AddInputDesc(*tensorDescPtr)) != GRAPH_SUCCESS,
                                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                                     fusion_op_desc->GetName().c_str(),
                                                     fusion_op_desc->GetType().c_str());
                                   return FAILED,
                                   "Add fusion_op_desc AddInputDesc failed");
    ge::DataType data_type = tensorDescPtr->GetDataType();
    auto iter = GE_TENSORFLOW_DATA_TYPE_MAP.find((int32_t)data_type);
    GE_IF_BOOL_EXEC(
        iter == GE_TENSORFLOW_DATA_TYPE_MAP.end(),
        REPORT_INNER_ERROR("E19999", "datatype:%d of input:%d in node:%s:%s is not supported",
                           data_type, in_anchor->GetIdx(), dst_node->GetName().c_str(), dst_node->GetName().c_str());
        GELOGE(PARAM_INVALID, "data_type %s not supported", ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
        return PARAM_INVALID);

    int32_t dtype = iter->second;
    input_list.push_back((int64_t)dtype);
    GELOGI("FUNCDEF: input_list push_back  %d.", dtype);
  }
  GE_IF_BOOL_EXEC(!input_list.empty(), (void)AttrUtils::SetListInt(fusion_op_desc, ge::T_IN_DATATYPE, input_list));

  return SUCCESS;
}

Status ParserGraphOptimizer::RebuildFusionNode(vector<ge::InDataAnchorPtr> &input_anchors,
                                               vector<ge::OutDataAnchorPtr> &output_anchors,
                                               map<ge::OutDataAnchorPtr, vector<ge::InDataAnchorPtr>> &output_in_map,
                                               vector<ge::InControlAnchorPtr> &input_control_anchors,
                                               vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                               ge::NodePtr fusion_node) {
  int32_t src_index = 0;

  for (auto out_anchor : output_anchors) {
    for (auto in_anchor : output_in_map[out_anchor]) {
      (void)in_anchor->Unlink(out_anchor);
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(fusion_node->GetOutDataAnchor(src_index), in_anchor),
                                  "Add anchor between fusion node and in anchor node!");
    }
    src_index++;
  }
  src_index = 0;
  for (auto in_anchor : input_anchors) {
    OutDataAnchorPtr out_anchor = in_anchor->GetPeerOutAnchor();
    out_anchor->Unlink(in_anchor);
    GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(out_anchor, fusion_node->GetInDataAnchor(src_index)),
                                "Add anchor between out anchor node and fusion node!");
    src_index++;
  }

  for (auto out_control_anchor : output_control_anchors) {
    for (auto in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
      in_control_anchor->Unlink(out_control_anchor);
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(fusion_node->GetOutControlAnchor(), in_control_anchor),
                                  "Add anchor between fusion node and in control anchor node!");
    }
  }
  for (auto in_control_anchor : input_control_anchors) {
    for (auto out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
      out_control_anchor->Unlink(in_control_anchor);
      GE_RETURN_WITH_LOG_IF_ERROR(GraphUtils::AddEdge(out_control_anchor, fusion_node->GetInControlAnchor()),
                                  "Add anchor between out control anchor node and fusion node!");
    }
  }
  return SUCCESS;
}

Status ParserGraphOptimizer::Insert4DTo5DTransOp(OutDataAnchorPtr src_anchor, InDataAnchorPtr dst_anchor,
                                                 enum ge::Format src_out_format, enum ge::DataType src_out_data_type,
                                                 enum ge::Format dst_in_format, enum ge::DataType dst_in_data_type) {
  bool isNCHWFP32To5DFP16 = (src_out_format == ge::FORMAT_NCHW && dst_in_format == ge::FORMAT_NC1HWC0);
  if (isNCHWFP32To5DFP16) {
    NodePtr cast_node = nullptr;

    if (src_out_data_type != dst_in_data_type) {
      OpDescPtr cast_opdesc = CreateCastOp(src_out_data_type, dst_in_data_type, ge::FORMAT_NCHW);
      cast_node = graph_->AddNode(cast_opdesc);
      GE_CHK_BOOL_EXEC(cast_node != nullptr,
                       REPORT_CALL_ERROR("E19999", "Add Cast node to graph:%s failed",
                                         graph_->GetName().c_str());
                       return INTERNAL_ERROR, "graph add cast node fail.");
    }

    OpDescPtr trans_data_opdesc = CreateTransDataOp(FORMAT_NCHW);
    NodePtr trans_data_node = graph_->AddNode(trans_data_opdesc);
    GE_CHK_BOOL_EXEC(trans_data_node != nullptr,
                     REPORT_CALL_ERROR("E19999", "Add Transdata node to graph:%s failed",
                                       graph_->GetName().c_str());
                     return INTERNAL_ERROR, "graph add TransData node node fail.");
    GE_CHK_STATUS_RET(NewNodeAddEdges(src_anchor, dst_anchor, nullptr, cast_node, trans_data_node),
                      "NewNodeAddEdges ret fail.");

    return SUCCESS;
  }

  OpDescPtr translateto5D = CreateTranslateOp(src_out_format, src_out_data_type, dst_in_format, dst_in_data_type);
  GE_CHECK_NOTNULL(translateto5D);
  NodePtr transNode = graph_->AddNode(translateto5D);
  GE_CHECK_NOTNULL(transNode);
  GELOGI("Create 4D To 5D fp32 node susscess!");

  GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, transNode->GetInDataAnchor(0)),
                  REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                                    src_anchor->GetOwnerNode()->GetName().c_str(),
                                    src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                                    transNode->GetName().c_str(), transNode->GetType().c_str());
                  return INTERNAL_ERROR);
  GE_IF_BOOL_EXEC(GraphUtils::AddEdge(transNode->GetOutDataAnchor(0), dst_anchor),
                  REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                                    transNode->GetName().c_str(), transNode->GetType().c_str(),
                                    dst_anchor->GetOwnerNode()->GetName().c_str(),
                                    dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                  return INTERNAL_ERROR);

  GELOGI("Create 4D To 5D susscess!");
  return SUCCESS;
}

Status ParserGraphOptimizer::InsertFZ2HWCK(OutDataAnchorPtr src_anchor, InDataAnchorPtr dst_anchor,
                                           enum ge::Format srcOutFormat, enum ge::DataType srcOutDatatype,
                                           enum ge::Format dstInFormat, enum ge::DataType dstInDatatype) {
  GELOGI("In InsertFZ2HWCK  !");
  GE_IF_BOOL_EXEC(
      srcOutFormat == ge::FORMAT_FRACTAL_Z, NodePtr transHalfNode = nullptr;
      if (srcOutDatatype == ge::DT_FLOAT) {
        // create FZ fp32->FZ  fp16 node
        OpDescPtr translatetoHalf = CreateTranslateOp(srcOutFormat, srcOutDatatype, srcOutFormat, ge::DT_FLOAT16);
        transHalfNode = graph_->AddNode(translatetoHalf);
        GE_CHECK_NOTNULL(transHalfNode);
        GELOGI("Create FZ fp32 to FZ fp16 node susscess!");
        // create FZ fp16->HWCK  fp32 node
      }

      OpDescPtr translatetoHWCK = CreateTranslateOp(srcOutFormat, ge::DT_FLOAT16, dstInFormat, dstInDatatype);
      NodePtr transHWCKNode = graph_->AddNode(translatetoHWCK); GELOGI("Create FZ 16 to HWCK fp32 node susscess!");
      GE_CHECK_NOTNULL(transHWCKNode); if (transHalfNode) {
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, transHalfNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            transHalfNode->GetName().c_str(), transHalfNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(transHalfNode->GetOutDataAnchor(0), transHWCKNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                                          transHalfNode->GetName().c_str(), transHalfNode->GetType().c_str(),
                                          transHWCKNode->GetName().c_str(), transHWCKNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(transHWCKNode->GetOutDataAnchor(0), dst_anchor) != SUCCESS,
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            transHWCKNode->GetName().c_str(), transHWCKNode->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                        return INTERNAL_ERROR);
      } else {
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, transHWCKNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            transHWCKNode->GetName().c_str(), transHWCKNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(transHWCKNode->GetOutDataAnchor(0), dst_anchor) != SUCCESS,
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            transHWCKNode->GetName().c_str(), transHWCKNode->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                        return INTERNAL_ERROR);
      } GELOGI("Create InsertFZ2HWCK success!");)
  return SUCCESS;
}

Status ParserGraphOptimizer::InsertVar5DTo4D(ge::OutDataAnchorPtr src_anchor, ge::InDataAnchorPtr dst_anchor,
                                             enum ge::Format srcOutFormat, enum ge::DataType srcOutDatatype,
                                             enum ge::Format dstInFormat, enum ge::DataType dstInDatatype) {
  GELOGI("In Insert 5D To 4D  !");
  GE_IF_BOOL_EXEC(
      srcOutFormat == ge::FORMAT_NC1HWC0, NodePtr cast_node = nullptr;
      if (srcOutDatatype == ge::DT_FLOAT && dstInDatatype == ge::DT_FLOAT) {
        auto cast_opdesc = CreateCastOp(ge::DT_FLOAT, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);
        cast_node = graph_->AddNode(cast_opdesc);

        srcOutDatatype = ge::DT_FLOAT16;
      } NodePtr transHalfNode = nullptr;
      OpDescPtr translateto4D = CreateTranslateOp(srcOutFormat, srcOutDatatype, dstInFormat, dstInDatatype);
      NodePtr trans4DNode = graph_->AddNode(translateto4D); GELOGI("Create 5D To 4D fp32 node susscess!");
      GE_CHECK_NOTNULL(trans4DNode);

      if (cast_node) {
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, cast_node->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            cast_node->GetName().c_str(), cast_node->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), trans4DNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                            cast_node->GetName().c_str(), cast_node->GetType().c_str(),
                            trans4DNode->GetName().c_str(), trans4DNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(trans4DNode->GetOutDataAnchor(0), dst_anchor) != SUCCESS,
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            trans4DNode->GetName().c_str(), trans4DNode->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                        return INTERNAL_ERROR);
      } else {
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, trans4DNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            trans4DNode->GetName().c_str(), trans4DNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(trans4DNode->GetOutDataAnchor(0), dst_anchor) != SUCCESS,
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            trans4DNode->GetName().c_str(), trans4DNode->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                        return INTERNAL_ERROR);
      } GELOGI("Create 5D To 4D susscess!");)
  return SUCCESS;
}

Status ParserGraphOptimizer::InsertHWCK2FZ(OutDataAnchorPtr src_anchor, InDataAnchorPtr dst_anchor,
                                           enum ge::Format srcOutFormat, enum ge::DataType srcOutDatatype,
                                           enum ge::Format dstInFormat, enum ge::DataType dstInDatatype) {
  GELOGI("In InsertHWCK2FZ  !");
  GE_IF_BOOL_EXEC(
      srcOutFormat == ge::FORMAT_HWCN, NodePtr transHalfNode = nullptr;
      OpDescPtr translatetoFZ = CreateTranslateOp(srcOutFormat, srcOutDatatype, dstInFormat, ge::DT_FLOAT16);
      NodePtr transHWCK2FZNode = graph_->AddNode(translatetoFZ); GELOGI("Create HWCK fp32 to FZ 16 node susscess!");
      GE_CHECK_NOTNULL(transHWCK2FZNode);

      ge::NodePtr translateHalftoFp32Node = nullptr; if (dstInDatatype == ge::DT_FLOAT) {
        // create FZ fp16 ->FZ  fp32 node
        ge::OpDescPtr translateHalftoFp32 = CreateTranslateOp(dstInFormat, ge::DT_FLOAT16, dstInFormat, dstInDatatype);
        translateHalftoFp32Node = graph_->AddNode(translateHalftoFp32);
        GE_CHECK_NOTNULL(translateHalftoFp32Node);
        GELOGI("Create FZ fp32 to FZ fp16 node susscess!");
        // create FZ fp16->HWCK  fp32 node
      }

      if (translateHalftoFp32Node) {
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, transHWCK2FZNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            transHWCK2FZNode->GetName().c_str(), transHWCK2FZNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(
            GraphUtils::AddEdge(transHWCK2FZNode->GetOutDataAnchor(0), translateHalftoFp32Node->GetInDataAnchor(0)),
            REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                              transHWCK2FZNode->GetName().c_str(), transHWCK2FZNode->GetType().c_str(),
                              translateHalftoFp32Node->GetName().c_str(), translateHalftoFp32Node->GetType().c_str());
            return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(translateHalftoFp32Node->GetOutDataAnchor(0), dst_anchor) != SUCCESS,
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            translateHalftoFp32Node->GetName().c_str(), translateHalftoFp32Node->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                        return INTERNAL_ERROR);
      } else {
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(src_anchor, transHWCK2FZNode->GetInDataAnchor(0)),
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            transHWCK2FZNode->GetName().c_str(), transHWCK2FZNode->GetType().c_str());
                        return INTERNAL_ERROR);
        GE_IF_BOOL_EXEC(GraphUtils::AddEdge(transHWCK2FZNode->GetOutDataAnchor(0), dst_anchor) != SUCCESS,
                        REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            transHWCK2FZNode->GetName().c_str(), transHWCK2FZNode->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                        return INTERNAL_ERROR);
      } GELOGI("Create InsertHWCK2FZ success!");)
  return SUCCESS;
}

Status ParserGraphOptimizer::Insert5DTo4DTransOp(OutDataAnchorPtr src_anchor, InDataAnchorPtr dst_anchor,
                                                 enum ge::Format src_out_format, enum ge::DataType src_out_data_type,
                                                 enum ge::Format dst_in_format, enum ge::DataType dst_in_data_type) {
  // Status ret;
  NodePtr permute_node = nullptr;
  NodePtr cast_node = nullptr;

  OpDescPtr trans_data_opdesc = CreateTransDataOp(FORMAT_NC1HWC0);
  NodePtr trans_data_node = graph_->AddNode(trans_data_opdesc);
  GE_CHK_BOOL_EXEC(trans_data_node != nullptr,
                   REPORT_CALL_ERROR("E19999", "Add Transdata node to graph:%s failed",
                                     graph_->GetName().c_str());
                   return INTERNAL_ERROR, "graph add TransData node node fail.");

  if (src_out_data_type != dst_in_data_type) {
    OpDescPtr cast_opdesc = CreateCastOp(src_out_data_type, dst_in_data_type, ge::FORMAT_NCHW);
    cast_node = graph_->AddNode(cast_opdesc);
    GE_CHK_BOOL_EXEC(cast_node != nullptr,
                     REPORT_CALL_ERROR("E19999", "Add Cast node to graph:%s failed",
                                     graph_->GetName().c_str());
                     return INTERNAL_ERROR, "graph add cast node fail.");
  }

  if (dst_in_format == FORMAT_NHWC) {
    OpDescPtr permute_opdec = CreatePermuteOp(FORMAT_NCHW, dst_in_format);
    permute_node = graph_->AddNode(permute_opdec);
    GE_CHK_BOOL_EXEC(permute_node != nullptr,
                     REPORT_CALL_ERROR("E19999", "Add Permute node to graph:%s failed",
                                       graph_->GetName().c_str());
                     return INTERNAL_ERROR, "graph add permute node fail.");
  }

  GE_CHK_STATUS_RET(NewNodeAddEdges(src_anchor, dst_anchor, trans_data_node, cast_node, permute_node),
                    "NewNodeAddEdges ret fail.");

  return SUCCESS;
}

Status ParserGraphOptimizer::NewNodeAddEdges(OutDataAnchorPtr src_anchor, InDataAnchorPtr dst_anchor, NodePtr first,
                                             NodePtr second, NodePtr third) {
  GE_CHECK_NOTNULL(src_anchor);
  GE_CHECK_NOTNULL(dst_anchor);
  OutDataAnchorPtr add_in_anchor = nullptr;
  InDataAnchorPtr add_out_anchor = nullptr;
  NodePtr src_node = src_anchor->GetOwnerNode();
  NodePtr dst_node = dst_anchor->GetOwnerNode();

  if (first != nullptr) {
    Status status = GraphUtils::AddEdge(src_anchor, first->GetInDataAnchor(0));
    GE_CHK_BOOL_EXEC(status == SUCCESS,
                     REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            first->GetName().c_str(), first->GetType().c_str());
                     return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                     src_anchor->GetIdx(), 0);
    if (second != nullptr) {
      status = GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0));
      GE_CHK_BOOL_EXEC(status == SUCCESS,
                       REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                            first->GetName().c_str(), first->GetType().c_str(),
                            second->GetName().c_str(), second->GetType().c_str());
                       return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.", 0, 0);
      if (third != nullptr) {
        status = GraphUtils::AddEdge(second->GetOutDataAnchor(0), third->GetInDataAnchor(0));
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                            second->GetName().c_str(), second->GetType().c_str(),
                            third->GetName().c_str(), third->GetType().c_str());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.", 0, 0);
        status = GraphUtils::AddEdge(third->GetOutDataAnchor(0), dst_anchor);
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            third->GetName().c_str(), third->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                         0, dst_anchor->GetIdx());
      } else {
        status = GraphUtils::AddEdge(second->GetOutDataAnchor(0), dst_anchor);
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            second->GetName().c_str(), second->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                         0, dst_anchor->GetIdx());
      }
    } else {
      if (third != nullptr) {
        status = GraphUtils::AddEdge(first->GetOutDataAnchor(0), third->GetInDataAnchor(0));
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                            first->GetName().c_str(), first->GetType().c_str(),
                            third->GetName().c_str(), third->GetType().c_str());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.", 0, 0);
        status = GraphUtils::AddEdge(third->GetOutDataAnchor(0), dst_anchor);
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            third->GetName().c_str(), third->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                         0, dst_anchor->GetIdx());
      } else {
        status = GraphUtils::AddEdge(first->GetOutDataAnchor(0), dst_anchor);
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                            first->GetName().c_str(), first->GetType().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                         0, dst_anchor->GetIdx());
      }
    }
  } else {
    if (second != nullptr) {
      Status status = GraphUtils::AddEdge(src_anchor, second->GetInDataAnchor(0));
      GE_CHK_BOOL_EXEC(status == SUCCESS,
                       REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            second->GetName().c_str(), second->GetType().c_str());
                       return INTERNAL_ERROR,
                       "graph add src to cast edge fail, src index:%d, dst index:%d.", src_anchor->GetIdx(), 0);
      GE_IF_BOOL_EXEC(
          third != nullptr, status = GraphUtils::AddEdge(second->GetOutDataAnchor(0), third->GetInDataAnchor(0));
          GE_CHK_BOOL_EXEC(status == SUCCESS,
                           REPORT_CALL_ERROR(
                               "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:0) failed",
                               second->GetName().c_str(), second->GetType().c_str(),
                               third->GetName().c_str(), third->GetType().c_str());
                           return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.", 0, 0);
          status = GraphUtils::AddEdge(third->GetOutDataAnchor(0), dst_anchor);
          GE_CHK_BOOL_EXEC(status == SUCCESS,
                           REPORT_CALL_ERROR(
                              "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                              third->GetName().c_str(), third->GetType().c_str(),
                              dst_anchor->GetOwnerNode()->GetName().c_str(),
                              dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                           return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                           0, dst_anchor->GetIdx()););
      GE_IF_BOOL_EXEC(third == nullptr, status = GraphUtils::AddEdge(second->GetOutDataAnchor(0), dst_anchor);
                      GE_CHK_BOOL_EXEC(
                          status == SUCCESS,
                          REPORT_CALL_ERROR(
                              "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                              second->GetName().c_str(), second->GetType().c_str(),
                              dst_anchor->GetOwnerNode()->GetName().c_str(),
                              dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                          return INTERNAL_ERROR,
                          "graph add edge fail, src index:%d, dst index:%d.", 0, 0););
    } else {
      if (third != nullptr) {
        Status status = GraphUtils::AddEdge(src_anchor, third->GetInDataAnchor(0));
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                            "E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            src_anchor->GetOwnerNode()->GetType().c_str(), src_anchor->GetIdx(),
                            third->GetName().c_str(), third->GetType().c_str());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.", 0, 0);
        status = GraphUtils::AddEdge(third->GetOutDataAnchor(0), dst_anchor);
        GE_CHK_BOOL_EXEC(status == SUCCESS,
                         REPORT_CALL_ERROR(
                              "E19999", "Add edge between op:%s(%s)(index:0) and op:%s(%s)(index:%d) failed",
                              third->GetName().c_str(), third->GetType().c_str(),
                              dst_anchor->GetOwnerNode()->GetName().c_str(),
                              dst_anchor->GetOwnerNode()->GetType().c_str(), dst_anchor->GetIdx());
                         return INTERNAL_ERROR, "graph add edge fail, src index:%d, dst index:%d.",
                         0, dst_anchor->GetIdx());
      }
    }
  }
  return SUCCESS;
}

OpDescPtr ParserGraphOptimizer::CreateTranslateOp(enum ge::Format inFormat, enum ge::DataType inDatatype,
                                                  enum ge::Format outFormat, enum ge::DataType outDatatype) {
  /**
   * 0. FP32 <-> FP16
   * 1. from HWCK(FP32) to FracZ(FP16);
   * 2. from FracZ(FP16) to HWCK(FP32);
   * 3. from NHWC(FP32) to NC1HWC0(FP16);
   * 4. from NC1HWC0(FP32) to NHWC(FP32);
   * 5. from NC1HWC0(FP16) to NHWC(FP32)
   */
  static uint32_t transop_count = 0;
  OpDescPtr op_def = nullptr;
  std::stringstream sstmp;
  sstmp << "translate_" << ge::parser::TRANSDATA << "_" << transop_count++;
  GE_MAKE_SHARED(op_def = std::make_shared<OpDesc>(sstmp.str().c_str(), ge::parser::TRANSLATE), op_def = nullptr;
                 return op_def);
  GELOGI(
      "create translate op:%s, input format:%s, input datatype:%s, output "
      "format:%s, output datatype:%s.",
      op_def->GetName().c_str(), ge::TypeUtils::FormatToSerialString(inFormat).c_str(),
      ge::TypeUtils::DataTypeToSerialString(inDatatype).c_str(), ge::TypeUtils::FormatToSerialString(outFormat).c_str(),
      ge::TypeUtils::DataTypeToSerialString(outDatatype).c_str());

  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_def, ge::ATTR_NAME_INPUT_FORMAT, inFormat),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INPUT_FORMAT.c_str(),
                                     op_def->GetName().c_str(), op_def->GetType().c_str());
                   return nullptr,
                   "SetInt ATTR_NAME_INPUT_FORMAT failed.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_def, ATTR_NAME_INPUT_DATATYPE, inDatatype),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INPUT_DATATYPE.c_str(),
                                     op_def->GetName().c_str(), op_def->GetType().c_str());
                   return nullptr,
                   "SetInt ATTR_NAME_INPUT_DATATYPE failed.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_def, ge::ATTR_NAME_OUTPUT_FORMAT, outFormat),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_OUTPUT_FORMAT.c_str(),
                                     op_def->GetName().c_str(), op_def->GetType().c_str());
                   return nullptr,
                   "SetInt ATTR_NAME_INPUT_DATATYPE failed.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_def, ATTR_NAME_OUTPUT_DATATYPE, outDatatype),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_OUTPUT_DATATYPE.c_str(),
                                     op_def->GetName().c_str(), op_def->GetType().c_str());
                   return nullptr,
                   "SetInt ATTR_NAME_INPUT_DATATYPE failed.");
  if (inDatatype != ge::DT_FLOAT16) {
    GE_CHK_BOOL_EXEC(SUCCESS == op_def->AddInputDesc(GeTensorDesc(GeShape(), inFormat)),
                     REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                       op_def->GetName().c_str(), op_def->GetType().c_str());
                     return nullptr,
                     "create translate op:add input desc fail.");
  } else {
    GE_CHK_BOOL_EXEC(SUCCESS == op_def->AddInputDesc(GeTensorDesc(GeShape(), inFormat, ge::DT_FLOAT16)),
                     REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                       op_def->GetName().c_str(), op_def->GetType().c_str());
                     return nullptr,
                     "create translate op:add input desc fail.");
  }
  if (outDatatype != ge::DT_FLOAT16) {
    GE_CHK_BOOL_EXEC(SUCCESS == op_def->AddOutputDesc(GeTensorDesc(GeShape(), outFormat)),
                     REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                       op_def->GetName().c_str(), op_def->GetType().c_str());
                     return nullptr,
                     "create translate op:add output desc fail.");
  } else {
    GE_CHK_BOOL_EXEC(SUCCESS == op_def->AddOutputDesc(GeTensorDesc(GeShape(), outFormat, ge::DT_FLOAT16)),
                     REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                       op_def->GetName().c_str(), op_def->GetType().c_str());
                     return nullptr, "create translate op:add output desc fail.");
  }
  return op_def;
}

OpDescPtr ParserGraphOptimizer::CreatePermuteOp(enum ge::Format input_format, enum ge::Format output_format) {
  static uint32_t transop_count = 0;

  std::stringstream sstmp;
  sstmp << "transdata_" << ge::parser::PERMUTE << "_" << transop_count++;

  OpDescPtr op_desc = nullptr;
  GE_MAKE_SHARED(op_desc = std::make_shared<OpDesc>(sstmp.str().c_str(), ge::parser::PERMUTE), op_desc = nullptr;
                 return op_desc);
  GELOGI("create permute op:%s", op_desc->GetName().c_str());

  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INPUT_FORMAT, (int64_t)input_format),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INPUT_FORMAT.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "SetInt ATTR_NAME_INPUT_FORMAT failed.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc, ge::ATTR_NAME_OUTPUT_FORMAT, (int64_t)output_format),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_OUTPUT_FORMAT.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "SetInt ATTR_NAME_OUTPUT_FORMAT failed.");

  GE_IF_BOOL_EXEC(input_format == FORMAT_NCHW, (void)AttrUtils::SetInt(op_desc, "NCHW_to_NHWC", (int64_t)1));
  GE_IF_BOOL_EXEC(input_format == FORMAT_NHWC, (void)AttrUtils::SetInt(op_desc, "NHWC_to_NCHW", (int64_t)1));

  GE_CHK_BOOL_EXEC(SUCCESS == op_desc->AddInputDesc(GeTensorDesc(GeShape(), input_format)),
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "create permute op:add input desc fail.");
  GE_CHK_BOOL_EXEC(SUCCESS == op_desc->AddOutputDesc(GeTensorDesc(GeShape(), output_format)),
                   REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "create permute op:add output desc fail.");

  return op_desc;
}

OpDescPtr ParserGraphOptimizer::CreateCastOp(enum ge::DataType input_data_type, enum ge::DataType output_data_type,
                                             enum ge::Format format) {
  static uint32_t transop_count = 0;
  std::stringstream sstmp;
  sstmp << "transdata_" << ge::parser::CAST << "_" << transop_count++;

  OpDescPtr op_desc = nullptr;
  GE_MAKE_SHARED(op_desc = std::make_shared<OpDesc>(sstmp.str().c_str(), ge::parser::CAST), op_desc = nullptr;
                 return op_desc);
  GELOGI("create cast op:%s, input datatype:%s, out datatype:%s", op_desc->GetName().c_str(),
         ge::TypeUtils::DataTypeToSerialString(input_data_type).c_str(),
         ge::TypeUtils::DataTypeToSerialString(output_data_type).c_str());

  if (!(AttrUtils::SetInt(op_desc, ge::CAST_ATTR_SRCT, (int64_t)input_data_type) &&
        AttrUtils::SetInt(op_desc, ge::CAST_ATTR_DSTT, (int64_t)output_data_type) &&
        AttrUtils::SetInt(op_desc, ge::CAST_ATTR_DST_TYPE, (int64_t)output_data_type) &&
        AttrUtils::SetBool(op_desc, ge::CAST_ATTR_TRUNCATE, false))) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s or %s or %s or %s to op:%s(%s) failed",
                      CAST_ATTR_SRCT.c_str(), CAST_ATTR_DSTT.c_str(),
                      CAST_ATTR_DST_TYPE.c_str(), CAST_ATTR_TRUNCATE.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "Set CAST_ATTR_SRCT or CAST_ATTR_DSTT or CAST_ATTR_DST_TYPE or CAST_ATTR_TRUNCATE fail, node: %s.",
           op_desc->GetName().c_str());
    return nullptr;
  }

  GE_CHK_BOOL_EXEC(SUCCESS == op_desc->AddInputDesc(GeTensorDesc(GeShape(), format, input_data_type)),
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "create cast op:add input desc fail.");
  GE_CHK_BOOL_EXEC(SUCCESS == op_desc->AddOutputDesc(GeTensorDesc(GeShape(), format, output_data_type)),
                   REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "create cast op:add output desc fail.");

  return op_desc;
}
OpDescPtr ParserGraphOptimizer::CreateTransDataOp(enum ge::Format input_format) {
  static uint32_t transop_count = 0;
  std::stringstream sstmp;
  sstmp << "transdata_" << ge::parser::TRANSDATA << "_" << transop_count++;

  OpDescPtr op_desc = nullptr;
  GE_MAKE_SHARED(op_desc = std::make_shared<OpDesc>(sstmp.str().c_str(), ge::parser::TRANSDATA), op_desc = nullptr;
                 return op_desc);

  GELOGI("create transdata op:%s, input format:%s.", op_desc->GetName().c_str(),
         ge::TypeUtils::FormatToSerialString(input_format).c_str());
  enum ge::Format output_format = FORMAT_NC1HWC0;
  if (input_format != FORMAT_NCHW) {
    input_format = FORMAT_NC1HWC0;
    output_format = FORMAT_NCHW;
  }

  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INPUT_FORMAT, (int64_t)input_format),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_INPUT_FORMAT.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "SetInt of ATTR_NAME_INPUT_FORMAT failed.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc, ge::ATTR_NAME_OUTPUT_FORMAT, (int64_t)output_format),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_OUTPUT_FORMAT.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "SetInt of ATTR_NAME_OUTPUT_FORMAT failed.");
  GE_CHK_BOOL_EXEC(SUCCESS == op_desc->AddInputDesc(GeTensorDesc(GeShape(), input_format)),
                   REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "create transdata op:add input desc fail.");
  GE_CHK_BOOL_EXEC(SUCCESS == op_desc->AddOutputDesc(GeTensorDesc(GeShape(), output_format)),
                   REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return nullptr,
                   "create transdata op:add output desc fail.");

  return op_desc;
}
}  // namespace ge
