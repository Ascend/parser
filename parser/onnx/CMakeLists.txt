set(SRC_LIST
    "onnx_custom_parser_adapter.cc"
    "onnx_parser.cc"
    "onnx_data_parser.cc"
    "onnx_util.cc"
    "onnx_constant_parser.cc"
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
    -Wno-deprecated-declarations
    -fno-common
    -fvisibility=hidden
)

target_compile_definitions(fmk_onnx_parser PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    FUNC_VISIBILITY
)

target_include_directories(fmk_onnx_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/subgraph_adapter
    ${PARSER_DIR}
    ${PARSER_DIR}/inc
    ${PARSER_DIR}/parser
    ${PARSER_DIR}/parser/inc
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/graph
    ${METADEF_DIR}/inc/register
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
    ${METADEF_DIR}/inc/external/register
    #### temp ####
    ${PARSER_DIR}/../graphengine/inc/common/util
    ${PARSER_DIR}/../graphengine/inc/external
    ${PARSER_DIR}/../graphengine/inc/framework
    ${PARSER_DIR}/../graphengine/inc
    ${PARSER_DIR}/../graphengine/ge
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/parser_protos
    #### blue zone compile #####
    ${PARSER_DIR}/../ge
    ${PARSER_DIR}/../inc
    ${PARSER_DIR}/../inc/common/util
    ${PARSER_DIR}/../inc/framework
    ${PARSER_DIR}/../inc/external
    ${PARSER_DIR}/../third_party/fwkacllib/inc
    #### independent compile #####
    ${METADEF_DIR}/third_party/graphengine/ge
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/framework
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
)

target_link_options(fmk_onnx_parser PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(fmk_onnx_parser PRIVATE
    $<BUILD_INTERFACE:intf_pub>
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
)

##################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_onnx_parser.cc
    COMMAND echo "Generating stub files."
    && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/../stub/gen_stubapi.py ${PARSER_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
    && mv onnx_parser.cc stub_onnx_parser.cc
    &&  echo "Generating stub files end."
    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS ../stub/gen_stubapi.py ${TOP_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
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
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
)

target_link_libraries(fmk_onnx_parser_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(fmk_onnx_parser_stub PROPERTIES
    OUTPUT_NAME fmk_onnx_parser
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS fmk_onnx_parser OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS fmk_onnx_parser_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)
