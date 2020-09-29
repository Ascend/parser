set(PROTO_LIST
    "${TOP_DIR}/inc/register/proto/onnx/ge_onnx.proto"
    "${TOP_DIR}/inc/common/proto/om.proto"
)

set(SRC_LIST
    "onnx_custom_parser_adapter.cc"
    "onnx_parser.cc"
    "onnx_data_parser.cc"
    "onnx_util.cc"
    "onnx_constant_parser.cc" 
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

############ libfmk_onnx_parser.so ############
add_library(fmk_onnx_parser SHARED ${SRC_LIST} ${PROTO_HDRS})

target_compile_options(fmk_onnx_parser PRIVATE
    -Werror
)

target_compile_definitions(fmk_onnx_parser PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
)

target_include_directories(fmk_onnx_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${TOP_DIR}/framework/domi/parser
    ${TOP_DIR}/framework/domi
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/external/graph
    ${TOP_DIR}/inc/framework
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
)

target_link_libraries(fmk_onnx_parser PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    protobuf
    ge_common
    register
    c_sec
    parser_common 
    graph
    slog
    mmpa
    -Wl,--as-needed
    json
    -lrt
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS fmk_onnx_parser OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
