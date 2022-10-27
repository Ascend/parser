set(SRC_LIST
    "onnx_custom_parser_adapter.cc"
    "onnx_parser.cc"
    "onnx_data_parser.cc"
    "onnx_util.cc"
    "onnx_constant_parser.cc"
    "onnx_file_constant_parser.cc"
    "subgraph_adapter/if_subgraph_adapter.cc"
    "subgraph_adapter/subgraph_adapter_factory.cc"
)

############ libfmk_onnx_parser.so ############
add_library(fmk_onnx_parser SHARED ${SRC_LIST})

add_dependencies(fmk_onnx_parser
    parser_protos
)

target_compile_options(fmk_onnx_parser PRIVATE
    -Werror
    -fno-common
    -fvisibility=hidden
    -Wextra
    -Wfloat-equal
)

target_compile_definitions(fmk_onnx_parser PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(fmk_onnx_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/subgraph_adapter
    ${PARSER_DIR}
    ${PARSER_DIR}/inc
    ${PARSER_DIR}/parser
    ${PARSER_DIR}/parser/inc
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/parser_protos
)

target_link_options(fmk_onnx_parser PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(fmk_onnx_parser
    PRIVATE
        intf_pub
        air_headers
        static_mmpa
        -Wl,--no-as-needed
        ascend_protobuf
        register
        c_sec
        parser_common
        graph
        slog
        -Wl,--as-needed
        json
        -lrt
        -ldl
        error_manager
    PUBLIC
        parser_headers
)

##################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_onnx_parser.cc
    COMMAND echo "Generating stub files."
    && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/../stub/gen_stubapi.py ${PARSER_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
    && mv onnx_parser.cc stub_onnx_parser.cc
    &&  echo "Generating stub files end."
)
##################################################################

############ stub/libfmk_onnx_parser.so ############
add_library(fmk_onnx_parser_stub SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/stub_onnx_parser.cc
)

target_compile_options(fmk_onnx_parser_stub PRIVATE
    -O2
    -fno-common
)

target_compile_definitions(fmk_onnx_parser_stub PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_HOST_INFER
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(fmk_onnx_parser_stub PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${PARSER_DIR}
    ${PARSER_DIR}/inc
    ${PARSER_DIR}/inc/external
    ${PARSER_DIR}/parser
    ${PARSER_DIR}/../inc
)

target_link_libraries(fmk_onnx_parser_stub PRIVATE
    intf_pub
    metadef_headers
)

set_target_properties(fmk_onnx_parser_stub PROPERTIES
    OUTPUT_NAME fmk_onnx_parser
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ install ############
install(TARGETS fmk_onnx_parser_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)
