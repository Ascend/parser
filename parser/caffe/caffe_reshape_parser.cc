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

#include "parser/caffe/caffe_reshape_parser.h"
#include <vector>
#include "parser/common/acl_graph_parser_util.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"
#include "proto/om.pb.h"

using namespace ge::parser;
using domi::CAFFE;

namespace ge {
namespace {
const int kAnchorIndexZero = 0;
const int kAnchorIndexOne = 1;
const int32_t RESHAPE_AXIS_DEFAULT_VALUE = 0;
const int32_t RESHAPE_NUM_AXES_DEFAULT_VALUE = -1;
}  // namespace

Status CaffeReshapeParser::ParseParams(const Message *op_src, ge::OpDescPtr &op) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);
  const LayerParameter *layer = DOMI_DYNAMIC_CAST<const LayerParameter *>(op_src);
  if (layer == nullptr) {
    REPORT_INNER_ERROR("E19999", "Reshape Dynamic cast op_src to LayerParameter failed");
    GELOGE(FAILED, "[Convert][DataType]Reshape Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  GELOGD("Caffe layer name = %s, layer type= %s, parse params", layer->name().c_str(), layer->type().c_str());
  const ReshapeParameter &reshape_parameter = layer->reshape_param();

  GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetInt(op, RESHAPE_ATTR_AXIS, RESHAPE_AXIS_DEFAULT_VALUE)),
                  GELOGW("SetInt failed for op %s.", op->GetName().c_str()););  // no need to return
  GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetInt(op, RESHAPE_ATTR_NUM_AXES, RESHAPE_NUM_AXES_DEFAULT_VALUE)),
                  GELOGW("SetInt failed for op %s.", op->GetName().c_str()););  // no need to return

  if (!reshape_parameter.has_shape()) {
    REPORT_INNER_ERROR("E19999", "Reshape has no shape info, ret fail, layer name = %s, layer type= %s",
                       layer->name().c_str(), layer->type().c_str());
    GELOGE(FAILED, "[Check][Param]Reshape has no shape info, ret fail, layer name = %s, layer type= %s",
           layer->name().c_str(), layer->type().c_str());
    return FAILED;
  }
  const BlobShape &blob_shape = reshape_parameter.shape();
  std::vector<int64_t> dims;
  for (int i = 0; i < blob_shape.dim_size(); i++) {
    dims.push_back(blob_shape.dim(i));
  }

  if (reshape_parameter.has_axis()) {
    GE_LOGW_IF(reshape_parameter.axis() == -1,
               "axis with -1 may lead to calculation errors when input less than 4 dims.");
    GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetInt(op, RESHAPE_ATTR_AXIS, reshape_parameter.axis())),
                    GELOGW("SetInt failed for op %s.", op->GetName().c_str()););  // no need to return
  }
  if (reshape_parameter.has_num_axes()) {
    GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetInt(op, RESHAPE_ATTR_NUM_AXES, reshape_parameter.num_axes())),
                    GELOGW("SetInt failed for op %s.", op->GetName().c_str()););  // no need to return
  }
  GE_IF_BOOL_EXEC(!(ge::AttrUtils::SetListInt(op, RESHAPE_ATTR_SHAPE, dims)),
                  GELOGW("SetListInt failed for op %s.", op->GetName().c_str()););  // no need to return
  return SUCCESS;
}

Status CaffeReshapeParser::ParseWeights(const Message *op_src, ge::OpDescPtr &op) {
  (void)op_src;
  (void)op;
  return SUCCESS;
}

Status CaffeReshapeParser::AddConstInput(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto owner_graph = node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "node's graph is empty, name: %s", node->GetName().c_str());
    GELOGE(FAILED, "[Get][OwnerComputeGraph]node's graph is empty, name: %s", node->GetName().c_str());
    return FAILED;
  }
  ge::OpDescPtr op = node->GetOpDesc();
  GE_CHECK_NOTNULL(op);
  vector<int64_t> attr_shape;
  GE_IF_BOOL_EXEC(!(ge::AttrUtils::GetListInt(op, RESHAPE_ATTR_SHAPE, attr_shape)),
                  GELOGW("GetListInt failed for op %s.", op->GetName().c_str()););  // no need to return
  size_t dims_size = attr_shape.size();

  // construct GeTensorDesc
  ge::GeTensorDesc const_desc = ge::GeTensorDesc();
  std::vector<int64_t> shape_vec = {static_cast<int64_t>(dims_size)};
  ge::GeShape shape(shape_vec);
  const_desc.Update(shape, ge::FORMAT_NCHW, ge::DT_INT64);
  ge::graphStatus state = op->UpdateInputDesc(RESHAPE_ATTR_SHAPE, const_desc);
  if (state != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "op:%s UpdateInputDesc failed.", op->GetName().c_str());
    GELOGE(FAILED, "[Update][InputDesc] failed for op:%s.", op->GetName().c_str());
    return FAILED;
  }

  // construct GeTensorPtr
  ge::GeTensorPtr constTensor = ge::parser::MakeShared<ge::GeTensor>();
  GE_CHECK_NOTNULL(constTensor);
  constTensor->SetTensorDesc(const_desc);

  std::unique_ptr<int64_t[]> data(new (std::nothrow) int64_t[dims_size]());
  GE_CHECK_NOTNULL(data);
  for (size_t i = 0; i < dims_size; ++i) {
    data[i] = attr_shape[i];
  }
  GE_IF_BOOL_EXEC(
      constTensor->SetData(reinterpret_cast<uint8_t *>(data.get()), dims_size * sizeof(int64_t)) != ge::GRAPH_SUCCESS,
      GELOGW("SetData failed for GeTensor."););  // no need to return

  // construct const node and add edge
  auto const_opdesc = ge::OpDescUtils::CreateConstOp(constTensor);
  GE_CHECK_NOTNULL(const_opdesc);
  auto const_node = owner_graph->AddNodeFront(const_opdesc);
  GE_CHECK_NOTNULL(const_node);
  ge::OutDataAnchorPtr out_archor_ptr = const_node->GetOutDataAnchor(kAnchorIndexZero);
  GE_CHECK_NOTNULL(out_archor_ptr);
  ge::InDataAnchorPtr in_archor_ptr = node->GetInDataAnchor(kAnchorIndexOne);
  GE_CHECK_NOTNULL(in_archor_ptr);
  state = ge::GraphUtils::AddEdge(out_archor_ptr, in_archor_ptr);
  if (state != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddEdge failed of from Node %s to Node %s",
                      const_node->GetName().c_str(), node->GetName().c_str());
    GELOGE(FAILED, "[Add][Edge] failed of from Node %s to Node %s",
           const_node->GetName().c_str(), node->GetName().c_str());
    return domi::FAILED;
  }
  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(CAFFE, RESHAPE, CaffeReshapeParser);
}  // namespace ge
