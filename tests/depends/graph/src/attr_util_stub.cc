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

#include "graph/ge_attr_value.h"
#include <set>
#include <google/protobuf/text_format.h>
#include "external/graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "graph/ge_tensor_impl.h"
#include "graph/buffer_impl.h"
#include "graph/op_desc_impl.h"
#include "proto/ge_ir.pb.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/debug/ge_attr_define.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "graph/serialization/tensor_desc_serializer.h"

using std::map;
using std::string;
using std::vector;
using std::set;

namespace ge {
void NamedAttrs::SetName(const std::string &name) {
  name_ = name;
}

string NamedAttrs::GetName() const {
  return name_;
}

AnyValue NamedAttrs::GetItem(const string &key) const {
  AnyValue value;
  (void)GetAttr(key, value);
  return value;
}

ProtoAttrMap &NamedAttrs::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &NamedAttrs::GetAttrMap() const {
  return attrs_;
}

bool AttrUtils::HasAttr(ConstAttrHolderAdapter &&obj, const string &name) {
  if (!obj) {
    return false;
  }
  return obj->HasAttr(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj, const string &name, int32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (int64_val > INT32_MAX) {
    REPORT_INNER_ERROR("E19999", "%ld int64_t value cannot cast to int32_t", int64_val);
    GELOGE(GRAPH_FAILED, "[Check][Param] %ld int64_t value cannot cast to int32_t", int64_val);
    return false;
  }
  value = static_cast<int32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj, const string &name, uint32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (int64_val > UINT32_MAX) {
    REPORT_INNER_ERROR("E19999", "%ld int64_t value cannot cast to uint32_t", int64_val);
    GELOGE(GRAPH_FAILED, "[Check][Param] %ld int64_t value cannot cast to uint32_t", int64_val);
    return false;
  }
  // 老版本中，只判断了上限，没有判断下限，因此小于0时，这里不会报错
  // 这里维持老版本的做法，在第一次上库做完后，补上小于0的判断
  value = static_cast<uint32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc) {
  if (org_op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "org_op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] org_op_desc is null");
    return nullptr;
  }
  std::shared_ptr<proto::OpDef> op_def;
  op_def = ComGraphMakeShared<proto::OpDef>();
  if (op_def == nullptr) {
    REPORT_CALL_ERROR("E19999", "create proto::OpDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][OpDef] proto::OpDef make shared failed");
    return nullptr;  // lint !e665
  }
  ModelSerializeImp imp;
  (void)imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  GE_CHK_BOOL_EXEC(imp.UnserializeOpDesc(op_desc, *op_def),
                   REPORT_CALL_ERROR("E19999", "UnserializeOpDesc failed");
                   return op_desc, "[Call][UnserializeOpDesc] op_desc unserialize failed");
  op_desc->ext_attrs_ = org_op_desc->ext_attrs_;

  // This function may be called by some passes of fusion engine, in this condition, do not need these attribute
  if (op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Op desc impl is nullptr.");
    return nullptr;
  }
  if (!op_desc->impl_->input_name_idx_.empty()) {
    op_desc->impl_->input_name_idx_.clear();
  }
  if (!op_desc->impl_->output_name_idx_.empty()) {
    op_desc->impl_->output_name_idx_.clear();
  }

  op_desc->impl_->MutableIRMeta() = IRMetaData(op_desc->GetName());
  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  if (org_op_desc == nullptr || org_op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "org_op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] org_op_desc is null");
    return nullptr;
  }
  std::shared_ptr<proto::OpDef> op_def = ComGraphMakeShared<proto::OpDef>();
  if (op_def == nullptr) {
    REPORT_CALL_ERROR("E19999", "create proto::OpDef failed");
    GELOGE(GRAPH_FAILED, "[Create][OpDef] proto::OpDef make shared failed");
    return nullptr;
  }
  ModelSerializeImp imp;
  (void)imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  if (!imp.UnserializeOpDesc(op_desc, *op_def)) {
    REPORT_CALL_ERROR("E19999", "UnserializeOpDesc failed.");
    return nullptr;
  }

  op_desc->ext_attrs_ = org_op_desc->ext_attrs_;

  if (op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "op desc impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] op desc impl is null.");
    return nullptr;
  }
  op_desc->impl_->input_name_idx_.insert(org_op_desc->impl_->input_name_idx_.begin(),
                                         org_op_desc->impl_->input_name_idx_.end());
  op_desc->impl_->MutableIRMeta() = org_op_desc->impl_->GetIRMeta();
  op_desc->impl_->output_name_idx_.insert(org_op_desc->impl_->output_name_idx_.begin(),
                                          org_op_desc->impl_->output_name_idx_.end());

  op_desc->impl_->infer_func_ = org_op_desc->impl_->infer_func_;
  op_desc->impl_->infer_format_func_ = org_op_desc->impl_->infer_format_func_;
  op_desc->impl_->verifier_func_ = org_op_desc->impl_->verifier_func_;

  return op_desc;
}

#define SET_ATTR_FUNC(type_name, type)  \
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY  \
    bool AttrUtils::Set##type_name(AttrHolderAdapter &&obj, const string &name, const type &value) {  \
      if (obj->HasAttr("test_fail")) {                                                                \
        return false;                                                                                 \
      }                                                                                               \
      return SetAttrValue(obj->MutableAttrMap(), name, value);  \
    }

#define GET_ATTR_FUNC(type_name, type) \
    GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY  \
    bool AttrUtils::Get##type_name(ConstAttrHolderAdapter &&obj, const string &name, type &value) { \
        return GetAttrValue(obj->GetAttrMap(), name, value);  \
    }

#define SET_GET_FUNC(type_name, type) \
    SET_ATTR_FUNC(type_name, type)  \
    GET_ATTR_FUNC(type_name, type)

bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const string &name, const vector<int64_t> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}
bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<int64_t> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}
SET_GET_FUNC(Int, int64_t)
SET_GET_FUNC(Float, float)
SET_GET_FUNC(ListFloat, vector<float>)
SET_GET_FUNC(Bool, bool)
SET_GET_FUNC(ListBool, vector<bool>)
SET_GET_FUNC(Str, string)
SET_GET_FUNC(ListStr, vector<string>)
SET_GET_FUNC(TensorDesc, GeTensorDesc)
SET_GET_FUNC(ListTensorDesc, vector<GeTensorDesc>)
SET_GET_FUNC(NamedAttrs, NamedAttrs)
SET_GET_FUNC(ListNamedAttrs, vector<NamedAttrs>)
SET_GET_FUNC(DataType, DataType)
SET_GET_FUNC(ListDataType, vector<DataType>)
SET_GET_FUNC(ListListInt, vector<vector<int64_t>>)
SET_GET_FUNC(ListListFloat, vector<vector<float>>)


bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const string &name, const vector<uint32_t> &value) {
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value.begin(), value.end()));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListInt(AttrUtils::AttrHolderAdapter &&obj, const string &name, const vector<int32_t> &value) {
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value.begin(), value.end()));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const string &name, std::initializer_list<int64_t> &&value) {
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<int32_t> &value) {
  value.clear();
  vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0; i < int64_list.size(); ++i) {
    if (int64_list[i] > INT32_MAX) {
      REPORT_INNER_ERROR("E19999", "index %zu %ld int64_t value cannot cast to int32_t", i, int64_list[i]);
      GELOGE(GRAPH_FAILED, "[Check][Param] index %zu %ld int64_t value cannot cast to int32_t", i, int64_list[i]);
      return false;
    }
  }
  value.insert(value.begin(), int64_list.begin(), int64_list.end());
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<uint32_t> &value) {
  value.clear();
  vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0; i < int64_list.size(); ++i) {
    if (int64_list[i] > UINT32_MAX) {
      REPORT_INNER_ERROR("E19999", "index %zu %ld int64_t value cannot cast to uint32_t", i, int64_list[i]);
      GELOGE(GRAPH_FAILED, "[Check][Param] index %zu %ld int64_t value cannot cast to uint32_t", i, int64_list[i]);
      return false;
    }
    // 老版本中，只判断了上限，没有判断下限，因此小于0时，这里不会报错
    // 这里维持老版本的做法，在第一次上库做完后，补上小于0的判断
  }
  value.insert(value.begin(), int64_list.begin(), int64_list.end());
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensor(AttrUtils::AttrHolderAdapter &&obj, const string &name, const GeTensor &value) {
  // 当前GeTensor的拷贝赋值、拷贝构造函数均不是深拷贝，因此无法使用默认的方法SetAttr
  if (!obj->MutableAttrMap().SetByName(name, GeTensor())) {
    return false;
  }
  auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  TensorUtils::CopyTensor(value, *tensor);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensor(AttrHolderAdapter &&obj, const string &name, const GeTensorPtr &value) {
  return SetTensor(std::move(obj), name, *value);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensor(AttrHolderAdapter &&obj, const string &name, const ConstGeTensorPtr &value) {
  return SetTensor(std::move(obj), name, *value);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrUtils::AttrHolderAdapter &&obj, const string &name, const vector<GeTensor> &value) {
  std::vector<GeTensor> tensors(value.size());
  if (!obj->MutableAttrMap().SetByName(name, tensors)) {
    return false;
  }
  auto attr_tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (attr_tensors == nullptr) {
    return false;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    TensorUtils::CopyTensor(value[i], (*attr_tensors)[i]);
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const string &name, const vector<GeTensorPtr> &value) {
  vector<ConstGeTensorPtr> tensors(value.size());
  std::copy(value.begin(), value.end(), tensors.begin());
  return SetListTensor(std::move(obj), name, tensors);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const string &name, const vector<ConstGeTensorPtr> &value) {
  std::vector<GeTensor> tensors(value.size());
  if (!obj->MutableAttrMap().SetByName(name, tensors)) {
    return false;
  }
  auto attr_tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (attr_tensors == nullptr) {
    return false;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    TensorUtils::CopyTensor(*(value[i]), (*attr_tensors)[i]);
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const string &name,
                              std::initializer_list<ConstGeTensorPtr> &&value) {
  return SetListTensor(std::move(obj), name, vector<ConstGeTensorPtr>(value));
}

// 所有权UT测试，不能把属性上的GeTensor给错误释放了
// 而且这里的行为与老版本是不一样的，老版本中，即使属性的owner生命周期结束析构了，通过本接口获取的value仍然是可用的
// 但是新接口中，owner没有转移，owner析构后，value指向的内存就被释放了，这里需要排查
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::MutableTensor(AttrHolderAdapter &&obj, const string &name, GeTensorPtr &value) {
  auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  value = std::shared_ptr<GeTensor>(tensor, [](GeTensor *){});
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetTensor(ConstAttrHolderAdapter &&obj, const string &name, ConstGeTensorPtr &value) {
  auto tensor = obj->GetAttrMap().GetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  value = std::shared_ptr<const GeTensor>(tensor, [](const GeTensor *){});
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListTensor(ConstAttrHolderAdapter &&obj, const string &name, vector<ConstGeTensorPtr> &value) {
  auto tensors = obj->GetAttrMap().GetByName<std::vector<GeTensor>>(name);
  if (tensors == nullptr) {
    return false;
  }
  value.resize(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    value[i] = std::shared_ptr<const GeTensor>(&(*tensors)[i], [](const GeTensor *){});
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::MutableListTensor(AttrHolderAdapter &&obj, const string &name, vector<GeTensorPtr> &value) {
  auto tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (tensors == nullptr) {
    return false;
  }
  value.resize(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    value[i] = std::shared_ptr<GeTensor>(&(*tensors)[i], [](GeTensor *){});
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetGraph(AttrUtils::AttrHolderAdapter &&obj, const string &name, const ComputeGraphPtr &value) {
  proto::GraphDef *graph_def = SetAndGetAttrValue(obj->MutableAttrMap(), name, proto::GraphDef());
  if (graph_def == nullptr) {
    return false;
  }
  ModelSerializeImp imp;
  if (!imp.SerializeGraph(value, graph_def)) {
    REPORT_CALL_ERROR("E19999", "SerializeGraph failed when add ComputeGraph to attr %s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Serialize][Graph] Failed when add ComputeGraph to attr %s", name.c_str());
    obj->MutableAttrMap().Delete(name);
    return false;
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListGraph(AttrUtils::AttrHolderAdapter &&obj, const string &name,
                             const vector<ComputeGraphPtr> &value) {
  std::vector<proto::GraphDef> graphs(value.size());
  if (!obj->MutableAttrMap().SetByName(name, graphs)) {
    return false;
  }
  auto attr_graphs = obj->MutableAttrMap().MutableGetByName<std::vector<proto::GraphDef>>(name);
  if (attr_graphs == nullptr) {
    return false;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    ModelSerializeImp imp;
    if (!imp.SerializeGraph(value[i], &attr_graphs->at(i))) {
          REPORT_CALL_ERROR("E19999", "SerializeGraph failed when add ComputeGraph to attr %s", name.c_str());
      GELOGE(GRAPH_FAILED, "[Serialize][Graph] Failed when add ComputeGraph to attr %s", name.c_str());
      obj->MutableAttrMap().Delete(name);
      return false;
    }
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetGraph(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, ComputeGraphPtr &value) {
  auto attr_graph_def = obj->GetAttrMap().GetByName<proto::GraphDef>(name);
  if (attr_graph_def == nullptr) {
    return false;
  }
  // 这里延续了老代码实现，先拷贝构造一个ComputeGraph，然后做反序列化，感觉直接把attr_graph_def传进去应该就可以了?
  // 下一步对这里做整改，直接传入attr_graph_def，避免这一次拷贝
  auto graph_def = ComGraphMakeShared<proto::GraphDef>(*attr_graph_def);
  if (graph_def == nullptr) {
    REPORT_CALL_ERROR("E19999", "create proto::GraphDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
    return false;
  }

  ModelSerializeImp imp;
  imp.SetProtobufOwner(graph_def);
  if (!imp.UnserializeGraph(value, *graph_def)) {
    REPORT_CALL_ERROR("E19999", "UnserializeGraph failed when get attr ComputeGraph by name %s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Unserialize][Graph] Failed when get attr ComputeGraph by name %s", name.c_str());
    return false;
  }

  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListGraph(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name,
                             vector<ComputeGraphPtr> &value) {
  auto graph_defs = obj->GetAttrMap().GetByName<std::vector<proto::GraphDef>>(name);
  if (graph_defs == nullptr) {
    return false;
  }

  value.resize(graph_defs->size());
  for (size_t i = 0; i < graph_defs->size(); ++i) {
    std::shared_ptr<proto::GraphDef> graph_def;
    graph_def = ComGraphMakeShared<proto::GraphDef>(graph_defs->at(i));
    if (graph_def == nullptr) {
      REPORT_CALL_ERROR("E19999", "create proto::GraphDef failed.");
      GELOGE(GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
      graph_def = nullptr;
      return false;  // lint !e665
    } else {
      ComputeGraphPtr graph = nullptr;
      ModelSerializeImp imp;
      imp.SetProtobufOwner(static_cast<const ProtoMsgOwner &>(graph_def));
      if (!imp.UnserializeGraph(graph, *graph_def)) {
        REPORT_CALL_ERROR("E19999", "UnserializeGraph failed.");
        GELOGE(GRAPH_FAILED, "[Unserialize][Graph] Failed");
        return false;
      }  // lint !e514
      value[i] = graph;
    }
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetBytes(AttrUtils::AttrHolderAdapter &&obj, const string &name, const Buffer &value) {
  auto buffer = SetAndGetAttrValue(obj->MutableAttrMap(), name, Buffer());
  if (buffer == nullptr) {
    return false;
  }
  BufferUtils::CopyFrom(value, *buffer);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetBytes(ConstAttrHolderAdapter &&obj, const string &name, Buffer &value) {
  auto buffer = obj->GetAttrMap().GetByName<Buffer>(name);
  if (buffer == nullptr) {
    return false;
  }
  BufferUtils::CopyFrom(*buffer, value);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListBytes(AttrUtils::AttrHolderAdapter &&obj, const string &name, const vector<Buffer> &value) {
  std::vector<Buffer> buffers(value.size());
  auto attr_buffers = SetAndGetAttrValue(obj->MutableAttrMap(), name, buffers);
  if (attr_buffers == nullptr) {
    return false;
  }

  for (size_t i = 0; i < value.size(); ++i) {
    BufferUtils::CopyFrom(value[i], (*attr_buffers)[i]);
  }

  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListBytes(AttrUtils::ConstAttrHolderAdapter &&obj, const string &name, vector<Buffer> &value) {
  auto buffers = obj->GetAttrMap().GetByName<std::vector<Buffer>>(name);
  if (buffers == nullptr) {
    return false;
  }
  value.resize(buffers->size());
  for (size_t i = 0; i < buffers->size(); ++i) {
    BufferUtils::CopyFrom(buffers->at(i), value[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetZeroCopyBytes(AttrHolderAdapter &&obj, const string &name, Buffer &&buffer) {
  // Value will be shared
  return SetAttrValue(obj->MutableAttrMap(), name, std::move(buffer));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetZeroCopyBytes(ConstAttrHolderAdapter &&obj, const string &name, Buffer &buffer) {
  // Value will be shared
  return GetAttrValue<Buffer>(obj->GetAttrMap(), name, buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetZeroCopyListBytes(AttrHolderAdapter &&obj, const string &name, vector<Buffer> &list_buffer) {
  // Value will be shared
  return SetAttrValue(obj->MutableAttrMap(), name, list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj, const string &name, vector<Buffer> &list_buffer) {
  // Value will be shared
  return GetAttrValue<vector<Buffer>>(obj->GetAttrMap(), name, list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::map<string, AnyValue> AttrUtils::GetAllAttrs(ConstAttrHolderAdapter &&obj) {
  auto holder = obj.get();
  if (holder == nullptr) {
    std::map<string, AnyValue> empty;
    return empty;
  }
  return holder->GetAllAttrs();
}


std::string AttrUtils::GetAttrsStrAfterRid(ConstAttrHolderAdapter &&obj, const set<string> &un_compute_attrs) {

  std::map<string, AnyValue> attr_map = GetAllAttrs(std::move(obj));
  if (attr_map.empty()) {
    return "";
  }
  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : attr_map) {
    proto::AttrDef attr_def;
    auto *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr.second.GetValueTypeId());
    if (serializer == nullptr || serializer->Serialize(attr.second, attr_def) != GRAPH_SUCCESS) {
      ordered_attrs[attr.first] = "";
      continue;
    }

    ordered_attrs[attr.first] = attr_def.SerializeAsString();
  }

  std::stringstream ss;
  for (auto &attr : ordered_attrs) {
    if (un_compute_attrs.find(attr.first) != un_compute_attrs.end()) {
      continue;
    }
    ss << attr.first << ":" << attr.second << ";";
  }
  return ss.str();
}
std::string AttrUtils::GetAllAttrsStr(ConstAttrHolderAdapter &&obj) {

  std::map<string, AnyValue> attr_map = GetAllAttrs(std::move(obj));
  if (attr_map.empty()) {
    return "";
  }
  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : attr_map) {
    proto::AttrDef attr_def;
    auto *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr.second.GetValueTypeId());
    if (serializer == nullptr || serializer->Serialize(attr.second, attr_def) != GRAPH_SUCCESS) {
      ordered_attrs[attr.first] = "";
      continue;
    }

    if (attr_def.has_t()) {
      // print tensor desc message as an ordered string.
      std::string ordered_tensor_desc;
      (void)google::protobuf::TextFormat::PrintToString(attr_def.t().desc(), &ordered_tensor_desc);
      ordered_attrs[attr.first] = ordered_tensor_desc + attr_def.t().data();
    } else if (attr_def.has_td()) {
      // print tensor desc message as an ordered string.
      string ordered_attr;
      (void)google::protobuf::TextFormat::PrintToString(attr_def.td(), &ordered_attr);
      ordered_attrs[attr.first] = ordered_attr;
    } else {
      ordered_attrs[attr.first] = attr_def.SerializeAsString();
    }
  }

  std::stringstream ss;
  for (auto &attr : ordered_attrs) {
    ss << attr.first << ":" << attr.second << ";";
  }
  return ss.str();
}
}  // namespace ge
