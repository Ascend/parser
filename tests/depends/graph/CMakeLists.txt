# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set(PARSER_PROTO_LIST
    "${PARSER_DIR}/metadef/proto/om.proto"
    "${PARSER_DIR}/metadef/proto/ge_ir.proto"
    "${PARSER_DIR}/metadef/proto/task.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/attr_value.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/function.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/graph.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/graph_library.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/node_def.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/op_def.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/resource_handle.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/tensor.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/tensor_shape.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/types.proto"
    "${PARSER_DIR}/metadef/proto/tensorflow/versions.proto"
    "${PARSER_DIR}/metadef/proto/caffe/caffe.proto"
    "${PARSER_DIR}/metadef/proto/onnx/ge_onnx.proto"
    #"${PARSER_DIR}/metadef/proto/proto_inner/ge_onnx.proto"
)

protobuf_generate(ge PARSER_PROTO_SRCS PARSER_PROTO_HDRS ${PARSER_PROTO_LIST})


file(GLOB_RECURSE SRCS RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        "src/attr_util_stub.cc"
)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)


add_library(attr_util_stub STATIC
    ${SRCS} ${PARSER_PROTO_HDRS}
)

target_compile_definitions(attr_util_stub PRIVATE
    google=ascend_private
)

target_compile_options(attr_util_stub PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(attr_util_stub PRIVATE
    intf_pub
    metadef_headers
    air_headers
    slog_headers
    ascend_protobuf
    c_sec
)
