set(SRC_LIST
    "parser_factory.cc"
    "data_op_parser.cc"
    "op_parser_factory.cc"
    "pre_checker.cc"
    "register_tbe.cc"
    "parser_api.cc"
    "parser_inner_ctx.cc"
    "proto_file_parser.cc"
    "acl_graph_parser_util.cc"
    "tbe_plugin_loader.cc"
    "model_saver.cc"
    "../tensorflow/tensorflow_custom_parser_adapter.cc"
    "../tensorflow/tensorflow_fusion_custom_parser_adapter.cc"
    "../tensorflow/tensorflow_fusion_op_parser.cc"
    "../tensorflow/tensorflow_util.cc"
    "convert/pb2json.cc"
    "op_def/ir_pb_converter.cc"
    "op_def/defs.cc"
    "op_def/op_schema.cc"
    "op_def/operator.cc"
    "op_map.cc"
    "parser_types.cc"
    "pass_manager.cc"
    "parser_fp16_t.cc"
    "thread_pool.cc"
)

############ libparser_common.so ############
add_library(parser_common SHARED ${SRC_LIST})

target_compile_options(parser_common PRIVATE
    -Werror
)

target_compile_definitions(parser_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
)

target_include_directories(parser_common PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${TOP_DIR}/framework/domi
    ${TOP_DIR}/framework/domi/common
    ${TOP_DIR}/framework/domi/parser
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/common/util
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/external/graph
    ${TOP_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
)

target_link_libraries(parser_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    graph
    protobuf
    register
    c_sec
    slog
    mmpa
    error_manager
    -Wl,--as-needed
    json
    -lrt
    -ldl
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS parser_common OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
