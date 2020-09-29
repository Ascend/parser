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

#ifndef GE_TYPES_MAP_H
#define GE_TYPES_MAP_H

#include "external/graph/types.h"
#include "proto/tensorflow/graph.pb.h"

namespace ge {
// Correspondence between data_type in GE and tensorflow
static map<int32_t, int32_t> GE_TENSORFLOW_DATA_TYPE_MAP = {
    {ge::DataType::DT_UNDEFINED, domi::tensorflow::DT_INVALID},
    {ge::DataType::DT_FLOAT, domi::tensorflow::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, domi::tensorflow::DT_HALF},
    {ge::DataType::DT_INT8, domi::tensorflow::DT_INT8},
    {ge::DataType::DT_INT16, domi::tensorflow::DT_INT16},
    {ge::DataType::DT_UINT16, domi::tensorflow::DT_UINT16},
    {ge::DataType::DT_UINT8, domi::tensorflow::DT_UINT8},
    {ge::DataType::DT_INT32, domi::tensorflow::DT_INT32},
    {ge::DataType::DT_INT64, domi::tensorflow::DT_INT64},
    {ge::DataType::DT_UINT32, domi::tensorflow::DT_UINT32},
    {ge::DataType::DT_UINT64, domi::tensorflow::DT_UINT64},
    {ge::DataType::DT_STRING, domi::tensorflow::DT_STRING},
    {ge::DataType::DT_RESOURCE, domi::tensorflow::DT_RESOURCE},
    {ge::DataType::DT_BOOL, domi::tensorflow::DT_BOOL},
    {ge::DataType::DT_DOUBLE, domi::tensorflow::DT_DOUBLE},
    {ge::DataType::DT_COMPLEX64, domi::tensorflow::DT_COMPLEX64},
    {ge::DataType::DT_COMPLEX128, domi::tensorflow::DT_COMPLEX128},
    {ge::DataType::DT_QINT8, domi::tensorflow::DT_QINT8},
    {ge::DataType::DT_QINT16, domi::tensorflow::DT_QINT16},
    {ge::DataType::DT_QINT32, domi::tensorflow::DT_QINT32},
    {ge::DataType::DT_QUINT8, domi::tensorflow::DT_QUINT8},
    {ge::DataType::DT_QUINT16, domi::tensorflow::DT_QUINT16},
    {ge::DataType::DT_DUAL, domi::tensorflow::DT_INVALID},
    {ge::DataType::DT_DUAL_SUB_INT8, domi::tensorflow::DT_INVALID},
    {ge::DataType::DT_DUAL_SUB_UINT8, domi::tensorflow::DT_INVALID},
};
}  // namespace ge
#endif  // GE_TYPES_MAP_H
