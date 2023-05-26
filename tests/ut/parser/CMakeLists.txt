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
set(CMAKE_CXX_STANDARD 11)

################################################################################
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

############ libut_parser_proto.a ############
add_library(ut_parser_proto STATIC
    ${PARSER_PROTO_HDRS} ${PARSER_PROTO_SRCS}
)

target_compile_definitions(ut_parser_proto PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ut_parser_proto PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(ut_parser_proto PRIVATE
    intf_pub
    ascend_protobuf
)


################################################################################
set(DUPLICATE_PROTO_LIST
    "${PARSER_DIR}/metadef/proto/onnx/ge_onnx.proto"
)

protobuf_generate(ge DUP_PROTO_SRCS DUP_PROTO_HDRS ${DUPLICATE_PROTO_LIST})

################################################################################
set(MATEDEF_SRC_FILES
    "${PARSER_DIR}/metadef/graph/utils/constant_utils.cc"
    "${PARSER_DIR}/metadef/graph/resource_context_mgr.cc"
    "${PARSER_DIR}/metadef/graph/aligned_ptr.cc"
    "${PARSER_DIR}/metadef/graph/anchor.cc"
    "${PARSER_DIR}/metadef/graph/ascend_string.cc"
    "${PARSER_DIR}/metadef/graph/attr_value.cc"
    "${PARSER_DIR}/metadef/graph/buffer.cc"
    "${PARSER_DIR}/metadef/graph/compute_graph.cc"
    "${PARSER_DIR}/metadef/graph/debug/graph_debug.cc"
    "${PARSER_DIR}/metadef/graph/detail/attributes_holder.cc"
    "${PARSER_DIR}/metadef/graph/format_refiner.cc"
    "${PARSER_DIR}/metadef/graph/ge_attr_define.cc"
    "${PARSER_DIR}/metadef/graph/any_value.cc"
    "${PARSER_DIR}/metadef/graph/attr_store.cc"
    "${PARSER_DIR}/metadef/graph/serialization/attr_serializer_registry.cc"
    "${PARSER_DIR}/metadef/graph/serialization/attr_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/bool_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/buffer_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/data_type_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/float_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/graph_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/int_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/list_list_float_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/list_list_int_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/list_value_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/named_attrs_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/string_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/tensor_desc_serializer.cc"
    "${PARSER_DIR}/metadef/graph/serialization/tensor_serializer.cc"
    "${PARSER_DIR}/metadef/graph/ge_tensor.cc"
    "${PARSER_DIR}/metadef/graph/gnode.cc"
    "${PARSER_DIR}/metadef/graph/graph.cc"
    "${PARSER_DIR}/metadef/graph/inference_context.cc"
    "${PARSER_DIR}/metadef/graph/model.cc"
    "${PARSER_DIR}/metadef/graph/model_serialize.cc"
    "${PARSER_DIR}/metadef/graph/node.cc"
    "${PARSER_DIR}/metadef/graph/op_desc.cc"
    "${PARSER_DIR}/metadef/graph/ir_meta.cc"
    "${PARSER_DIR}/metadef/graph/ir_data_type_symbol_store.cc"
    "${PARSER_DIR}/metadef/graph/operator.cc"
    "${PARSER_DIR}/metadef/graph/operator_impl.cc"
    "${PARSER_DIR}/metadef/graph/operator_factory.cc"
    "${PARSER_DIR}/metadef/graph/operator_factory_impl.cc"
    "${PARSER_DIR}/metadef/graph/opsproto/opsproto_manager.cc"
    "${PARSER_DIR}/metadef/graph/option/ge_context.cc"
    "${PARSER_DIR}/metadef/graph/option/ge_local_context.cc"
    "${PARSER_DIR}/metadef/graph/ref_relation.cc"
    "${PARSER_DIR}/metadef/graph/runtime_inference_context.cc"
    "${PARSER_DIR}/metadef/graph/shape_refiner.cc"
    "${PARSER_DIR}/metadef/graph/tensor.cc"
    "${PARSER_DIR}/metadef/graph/types.cc"
    "${PARSER_DIR}/metadef/graph/utils/anchor_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/file_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/ge_ir_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/graph_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/graph_utils_ex.cc"
    "${PARSER_DIR}/metadef/graph/utils/connection_matrix.cc"
    "${PARSER_DIR}/metadef/graph/utils/cycle_detector.cc"
    "${PARSER_DIR}/metadef/graph/utils/node_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/node_utils_ex.cc"
    "${PARSER_DIR}/metadef/graph/utils/op_desc_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/op_desc_utils_ex.cc"
    "${PARSER_DIR}/metadef/graph/utils/tensor_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/transformer_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/tuning_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/type_utils.cc"
    "${PARSER_DIR}/metadef/graph/utils/trace/trace_manager.cc"
    "${PARSER_DIR}/metadef/graph/utils/op_type_utils.cc"
    "${PARSER_DIR}/metadef/graph/common/large_bm.cc"
    "${PARSER_DIR}/metadef/graph/common/plugin/plugin_manager.cc"
    "${PARSER_DIR}/metadef/ops/op_imp.cpp"
    "${PARSER_DIR}/metadef/third_party/transformer/src/axis_util.cc"
    "${PARSER_DIR}/metadef/third_party/transformer/src/expand_dimension.cc"
    "${PARSER_DIR}/metadef/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${PARSER_DIR}/metadef/third_party/transformer/src/transfer_shape_utils.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${PARSER_DIR}/metadef/inc)
include_directories(${PARSER_DIR}/metadef/inc/graph)
include_directories(${PARSER_DIR}/metadef/inc/external)
include_directories(${PARSER_DIR}/metadef/inc/external/graph)
include_directories(${PARSER_DIR}/metadef/graph)
include_directories(${PARSER_DIR}/metadef/register)
include_directories(${PARSER_DIR}/metadef)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

############ libut_parser_graph.a ############
add_library(ut_parser_graph STATIC
    ${MATEDEF_SRC_FILES} ${PARSER_PROTO_HDRS} ${DUP_PROTO_HDRS}
)

target_compile_definitions(ut_parser_graph PRIVATE
    google=ascend_private
)

target_compile_options(ut_parser_graph PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(ut_parser_graph PRIVATE
    intf_pub
    metadef_headers
    air_headers
    slog_headers
    c_sec ascend_protobuf
)


################################################################################
set(REGISTER_SRC_FILES
    "${PARSER_DIR}/metadef/register/auto_mapping_util.cpp"
    "${PARSER_DIR}/metadef/register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/fusion_statistic/fusion_statistic_recorder.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/graph_fusion/fusion_pass_registry.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/graph_fusion/fusion_pattern.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/graph_fusion/graph_fusion_pass_base.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/graph_fusion/pattern_fusion_base_pass.cc"
    "${PARSER_DIR}/metadef/register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cc"
    "${PARSER_DIR}/metadef/register/host_cpu_context.cc"
    "${PARSER_DIR}/metadef/register/infer_data_slice_registry.cc"
    "${PARSER_DIR}/metadef/register/ops_kernel_builder_registry.cc"
    "${PARSER_DIR}/metadef/register/op_kernel_registry.cpp"
    "${PARSER_DIR}/metadef/register/op_tiling/op_tiling.cc"
    "${PARSER_DIR}/metadef/register/op_tiling/op_tiling_registry.cc"
    "${PARSER_DIR}/metadef/register/register.cpp"
    "${PARSER_DIR}/metadef/register/register_pass.cpp"
    "${PARSER_DIR}/metadef/register/scope/scope_graph.cc"
    "${PARSER_DIR}/metadef/register/scope/scope_pass.cc"
    "${PARSER_DIR}/metadef/register/scope/scope_pass_registry.cc"
    "${PARSER_DIR}/metadef/register/scope/scope_pattern.cc"
    "${PARSER_DIR}/metadef/register/scope/scope_util.cc"
    "${PARSER_DIR}/metadef/register/tensor_assign.cpp"
    "${PARSER_DIR}/metadef/register/prototype_pass_registry.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${PARSER_DIR}/metadef)
include_directories(${PARSER_DIR}/metadef/graph)
include_directories(${PARSER_DIR}/metadef/register)
include_directories(${PARSER_DIR}/metadef/inc)
include_directories(${PARSER_DIR}/metadef/inc/external)
include_directories(${PARSER_DIR}/metadef/inc/register)

############ libut_parser_register.a ############
add_library(ut_parser_register STATIC
    ${REGISTER_SRC_FILES} ${PARSER_PROTO_HDRS}
)

target_compile_definitions(ut_parser_register PRIVATE
    google=ascend_private
)

target_compile_options(ut_parser_register PRIVATE
    -O2 -g -fno-common
)

target_link_libraries(ut_parser_register PRIVATE
    intf_pub
    air_headers
    slog_headers
    runtime_headers
    msprof_headers
    c_sec ascend_protobuf json
)


################################################################################
set(PARSER_SRC_FILES
    "${PARSER_DIR}/parser/caffe/caffe_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/caffe/caffe_data_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_op_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_parser.cc"
    "${PARSER_DIR}/parser/caffe/caffe_reshape_parser.cc"
    "${PARSER_DIR}/parser/common/acl_graph_parser_util.cc"
    "${PARSER_DIR}/parser/common/convert/pb2json.cc"
    "${PARSER_DIR}/parser/common/convert/message2operator.cc"
    "${PARSER_DIR}/parser/common/data_op_parser.cc"
    "${PARSER_DIR}/parser/common/model_saver.cc"
    "${PARSER_DIR}/parser/common/op_def/arg_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/constant_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/fill_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/framework_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/ir_pb_converter.cc"
    "${PARSER_DIR}/parser/common/op_def/no_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/operator.cc"
    "${PARSER_DIR}/parser/common/op_def/ref_switch_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/shape_n_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/variable_operator.cc"
    "${PARSER_DIR}/parser/common/op_def/var_is_initialized_op_operator.cc"
    "${PARSER_DIR}/parser/common/op_map.cc"
    "${PARSER_DIR}/parser/common/op_parser_factory.cc"
    "${PARSER_DIR}/parser/common/parser_api.cc"
    "${PARSER_DIR}/parser/common/parser_factory.cc"
    "${PARSER_DIR}/parser/common/parser_fp16_t.cc"
    "${PARSER_DIR}/parser/common/parser_inner_ctx.cc"
    "${PARSER_DIR}/parser/common/parser_types.cc"
    "${PARSER_DIR}/parser/common/parser_utils.cc"
    "${PARSER_DIR}/parser/common/pass_manager.cc"
    "${PARSER_DIR}/parser/common/pre_checker.cc"
    "${PARSER_DIR}/parser/common/proto_file_parser.cc"
    "${PARSER_DIR}/parser/common/prototype_pass_manager.cc"
    "${PARSER_DIR}/parser/common/op_registration_tbe.cc"
    "${PARSER_DIR}/parser/common/tbe_plugin_loader.cc"
    "${PARSER_DIR}/parser/common/thread_pool.cc"
    "${PARSER_DIR}/parser/common/auto_mapping_subgraph_io_index_func.cc"
    "${PARSER_DIR}/parser/onnx/onnx_constant_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_file_constant_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/onnx/onnx_data_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_parser.cc"
    "${PARSER_DIR}/parser/onnx/onnx_util.cc"
    "${PARSER_DIR}/parser/onnx/subgraph_adapter/if_subgraph_adapter.cc"
    "${PARSER_DIR}/parser/onnx/subgraph_adapter/subgraph_adapter_factory.cc"
    "${PARSER_DIR}/parser/tensorflow/graph_to_function_def.cc"
    "${PARSER_DIR}/parser/tensorflow/parser_graph_optimizer.cc"
    "${PARSER_DIR}/parser/tensorflow/iterator_fusion_pass.cc"
    "${PARSER_DIR}/parser/tensorflow/scope/scope_pass_manager.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_arg_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_auto_mapping_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_constant_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_data_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_enter_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fill_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_frameworkop_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fusion_custom_parser_adapter.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_fusion_op_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_identity_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_merge_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_no_op_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_ref_switch_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_reshape_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_shape_n_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_squeeze_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_util.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_variable_v2_parser.cc"
    "${PARSER_DIR}/parser/tensorflow/tensorflow_var_is_initialized_op_parser.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${PARSER_DIR})
include_directories(${PARSER_DIR}/inc)
include_directories(${PARSER_DIR}/parser)
include_directories(${PARSER_DIR}/parser/onnx)
include_directories(${PARSER_DIR}/tests)
include_directories(${PARSER_DIR}/metadef/inc)
include_directories(${PARSER_DIR}/metadef/inc/external)
include_directories(${PARSER_DIR}/metadef/inc/register)

set(PARSER_UT_FILES
    "parser_ut_utils.cc"
    "testcase/common/acl_graph_parser_unittest.cc"
    "testcase/onnx_parser_testcase/onnx_parser_unittest.cc"
    "testcase/caffe_parser_testcase/caffe_parser_unittest.cc"
    "testcase/onnx_parser_testcase/message2operator_unittest.cc"
    "testcase/tensorflow_parser_testcase/tensorflow_parser_unittest.cc"
    "testcase/tensorflow_parser_testcase/tensorflow_auto_mapping_parser_adapter_unittest.cc"
    "testcase/graph_optimizer_testcase/graph_optimizer_unittest.cc"
)

############ libut_parser_common.a ############
add_library(ut_parser_common STATIC
    ${PARSER_SRC_FILES} ${PARSER_PROTO_HDRS}
)

target_compile_definitions(ut_parser_common PRIVATE
    google=ascend_private
)

target_compile_options(ut_parser_common PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ut_parser_common PRIVATE
    intf_pub
    air_headers
    slog_headers
    ut_parser_proto ut_parser_graph c_sec
    ascend_protobuf
    json
)


################################################################################
add_executable(ut_parser
    ${PARSER_UT_FILES} ${PARSER_PROTO_SRCS}
)

target_compile_options(ut_parser PRIVATE
    -g
)

target_compile_definitions(ut_parser PRIVATE
    google=ascend_private
)

target_link_libraries(ut_parser
    intf_pub
    metadef_headers
    air_headers
    slog_headers
    cce_headers
    runtime_headers
    msprof_headers
    ut_parser_proto
    -Wl,--whole-archive ut_parser_common -Wl,--no-whole-archive
    ut_parser_graph ut_parser_register error_manager_stub parser_mmpa_stub attr_util_stub
    GTest::gtest GTest::gtest_main parser_slog_stub ascend_protobuf c_sec json -lrt -ldl -lgcov
)
