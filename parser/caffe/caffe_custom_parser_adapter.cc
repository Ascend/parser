/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020~2022. All rights reserved.
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

#include "parser/caffe/caffe_custom_parser_adapter.h"
#include <memory>
#include <vector>
#include "parser/common/acl_graph_parser_util.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/utils/graph_utils.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"
#include "parser/common/parser_utils.h"

using domi::ParseParamByOpFunc;
using domi::ParseParamFunc;
using domi::CAFFE;
using domi::caffe::LayerParameter;
using domi::caffe::InnerProductParameter;
using domi::caffe::ConvolutionParameter;
using std::vector;

namespace ge {
namespace {
const char *const kConvolution = "Convolution";
const char *const kInnerProduct = "InnerProduct";
const int64_t kDimDedaultValue = 1;
const int kBlobIndexOne = 1;
}  // namespace

Status CaffeCustomParserAdapter::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  GE_CHECK_NOTNULL(op_src);
  const LayerParameter *layer = PtrToPtr<const Message, const LayerParameter>(op_src);
  GELOGD("Caffe layer name = %s, layer type= %s, parse params", layer->name().c_str(), layer->type().c_str());
  GE_CHECK_NOTNULL(op_dest);

  ParseParamFunc customOpParser = domi::OpRegistry::Instance()->GetParseParamFunc(op_dest->GetType(), layer->type());
  GE_CHECK_NOTNULL(customOpParser);

  op_dest->SetName(layer->name());
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  GE_CHK_BOOL_RET_STATUS(customOpParser(op_src, op) == SUCCESS, FAILED,
                         "[Invoke][CustomOpParser] failed, layer name:%s, layer type:%s",
                         layer->name().c_str(), layer->type().c_str());
  return SUCCESS;
}

Status CaffeCustomParserAdapter::ParseParams(const Operator &op_src, const ge::OpDescPtr &op_dest) {
  GELOGI("Caffe custom op begin to params: layer name = %s, layer type= %s ",
         ParserUtils::GetOperatorName(op_src).c_str(), ParserUtils::GetOperatorType(op_src).c_str());
  GE_CHECK_NOTNULL(op_dest);

  ParseParamByOpFunc custom_op_parser = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(
      ParserUtils::GetOperatorType(op_src));
  GE_CHECK_NOTNULL(custom_op_parser);

  op_dest->SetName(ParserUtils::GetOperatorName(op_src));
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);

  GE_CHK_BOOL_RET_STATUS(custom_op_parser(op_src, op) == SUCCESS, FAILED,
                         "[Invoke][CustomOpParser] failed, layer name:%s, type:%s",
                         ParserUtils::GetOperatorName(op_src).c_str(), ParserUtils::GetOperatorType(op_src).c_str());
  return SUCCESS;
}

Status CaffeCustomParserAdapter::AddEdgeFromConstNode(const NodePtr &const_node, const int32_t index,
                                                      const bool update_in_turn, ge::NodePtr &node) const {
  GE_CHECK_NOTNULL(const_node);
  GE_CHECK_NOTNULL(node);
  auto op = node->GetOpDesc();
  GE_CHECK_NOTNULL(op);
  auto valid_input_name = op->GetValidInputNameByIndex(index);
  if (update_in_turn || valid_input_name.empty()) {
    if (node->AddLinkFrom(static_cast<const uint32_t &>(index), const_node) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddEdge failed of from Node %s output to Node %s input %d",
                        const_node->GetName().c_str(), node->GetName().c_str(), index);
      GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] AddEdge failed of from Node %s output to Node %s input %d",
             const_node->GetName().c_str(), node->GetName().c_str(), index);
    }
  } else {
    if (node->AddLinkFrom(valid_input_name, const_node) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddEdge failed of from Node %s output to Node %s input %s",
                        const_node->GetName().c_str(), node->GetName().c_str(), valid_input_name.c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] AddEdge failed of from Node %s output to Node %s input %s",
             const_node->GetName().c_str(), node->GetName().c_str(), valid_input_name.c_str());
    }
  }
  return SUCCESS;
}

Status CaffeCustomParserAdapter::ParseWeights(const Message *op_src, ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);
  const LayerParameter *layer = PtrToPtr<const Message, const LayerParameter>(op_src);

  GE_CHK_BOOL_RET_STATUS(layer != nullptr, FAILED, "[Convert][Type]Dynamic cast op_src to LayerParameter failed");
  GELOGI("layer: %s blobs_size: %d bottom_size: %d", layer->name().c_str(), layer->blobs_size(), layer->bottom_size());
  if (layer->blobs_size() == 0) {
    return SUCCESS;
  }

  bool bias_en = false;
  bool update_in_turn = (static_cast<int64_t >(op->GetAllInputsSize()) == (layer->bottom_size() + layer->blobs_size()));
  int start_pos = layer->bottom_size();
  for (int i = 0; i < layer->blobs_size(); ++i) {
    ge::GeTensorPtr weight = ge::parser::MakeShared<ge::GeTensor>();
    GE_CHECK_NOTNULL(weight);
    GE_CHK_STATUS_RET(ConvertWeight(layer->blobs(i), layer->name(), weight),
                      "[Convert][Blobs] (%d) for layer %s failed", i, layer->name().c_str());
    GE_IF_BOOL_EXEC(layer->type() == kConvolution && i == kBlobIndexOne,
                    bias_en = layer->convolution_param().bias_term();
                   );
    GE_IF_BOOL_EXEC(layer->type() == kInnerProduct && i == kBlobIndexOne,
                    bias_en = layer->inner_product_param().bias_term();
                    );
    auto bias_shape = weight->MutableTensorDesc().GetShape();
    // The num 0, 1, 2, 3 represet the dim index.
    bool matched = bias_en && bias_shape.GetDimNum() == static_cast<size_t>(ge::parser::DIM_DEFAULT_SIZE) &&
                   bias_shape.GetDim(0) == 1 && bias_shape.GetDim(1) == 1 && bias_shape.GetDim(2) == 1;
    if (matched) {
      weight->MutableTensorDesc().SetShape(ge::GeShape({bias_shape.GetDim(3)}));
    }
    matched = layer->type() == kInnerProduct && i == 0 &&
              bias_shape.GetDimNum() == static_cast<size_t>(ge::parser::DIM_DEFAULT_SIZE) &&
              bias_shape.GetDim(0) == 1 && bias_shape.GetDim(1) == 1;
    if (matched) {
      weight->MutableTensorDesc().SetShape(ge::GeShape({bias_shape.GetDim(2), bias_shape.GetDim(3)}));
    }

    // construct const node
    auto const_opdesc = ge::OpDescUtils::CreateConstOp(weight);  // use org weight before SetWeights Overwrite
    GE_CHECK_NOTNULL(const_opdesc);
    auto owner_graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);

    // add edge from const to current node
    auto const_node = owner_graph->AddNodeFront(const_opdesc);
    auto index = start_pos + i;
    GE_CHK_STATUS_RET_NOLOG(AddEdgeFromConstNode(const_node, static_cast<int32_t>(index), update_in_turn, node));

    std::vector<ge::NodePtr> original_nodes;
    ge::GraphUtils::RecordOriginalNames(original_nodes, const_node);
  }
  GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetInt(op, "tvm_origin_input_num", layer->bottom_size())),
                  GELOGW("SetInt failed for op %s.", op->GetName().c_str()););  // no need to return

  return SUCCESS;
}
REGISTER_CUSTOM_PARSER_ADAPTER_CREATOR(CAFFE, CaffeCustomParserAdapter);
}  // namespace ge
