set(SRC_LIST
    "tensorflow/tensorflow_arg_parser.cc"
    "tensorflow/tensorflow_auto_mapping_parser_adapter.cc"
    "tensorflow/tensorflow_constant_parser.cc"
    "tensorflow/tensorflow_data_parser.cc"
    "tensorflow/tensorflow_enter_parser.cc"
    "tensorflow/tensorflow_fill_parser.cc"
    "tensorflow/tensorflow_frameworkop_parser.cc"
    "tensorflow/tensorflow_fusionop_util.cc"
    "tensorflow/tensorflow_identity_parser.cc"
    "tensorflow/tensorflow_merge_parser.cc"
    "tensorflow/tensorflow_no_op_parser.cc"
    "tensorflow/tensorflow_parser.cc"
    "tensorflow/tensorflow_ref_switch_parser.cc"
    "tensorflow/tensorflow_reshape_parser.cc"
    "tensorflow/tensorflow_shape_n_parser.cc"
    "tensorflow/tensorflow_squeeze_parser.cc"
    "tensorflow/tensorflow_var_is_initialized_op_parser.cc"
    "tensorflow/tensorflow_variable_v2_parser.cc"
    "caffe/caffe_parser.cc"
    "caffe/caffe_data_parser.cc"
    "caffe/caffe_reshape_parser.cc"
    "caffe/caffe_custom_parser_adapter.cc"
    "caffe/caffe_op_parser.cc"
    "tensorflow/scope/scope_pass_manager.cc"
    "tensorflow/graph_functiondef.cc"
    "tensorflow/graph_optimizer.cc"
    "tensorflow/iterator_fusion_pass.cc"
    "common/op_def/arg_op.cc"
    "common/op_def/constant_op.cc"
    "common/op_def/fill_op.cc"
    "common/op_def/frameworkop_op.cc"
    "common/op_def/no_op_op.cc"
    "common/op_def/ref_switch_op.cc"
    "common/op_def/shape_n_op.cc"
    "common/op_def/var_is_initialized_op_op.cc"
    "common/op_def/variable_op.cc"
)

############ libfmk_parser.so ############
add_library(fmk_parser SHARED
    ${SRC_LIST}
    $<TARGET_OBJECTS:parser_graph_library_proto_obj>
)

target_compile_options(fmk_parser PRIVATE
    -Werror
    -Wno-deprecated-declarations
    -fno-common
    -fvisibility=hidden
)

target_compile_definitions(fmk_parser PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    FUNC_VISIBILITY
)

target_include_directories(fmk_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${PARSER_DIR}
    ${PARSER_DIR}/inc
    ${PARSER_DIR}/parser
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
    ${PARSER_DIR}/../inc/framework
    ${PARSER_DIR}/../inc/common/util
    ${PARSER_DIR}/../inc/external
    ${PARSER_DIR}/../third_party/fwkacllib/inc
    #### blue independent compile #####
    ${METADEF_DIR}/third_party/graphengine/ge
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/framework
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
)

target_link_options(fmk_parser PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(fmk_parser
    $<BUILD_INTERFACE:intf_pub>
    static_mmpa
    -Wl,--no-as-needed
    ascend_protobuf
    error_manager
    parser_common
    graph
    register
    _caffe_parser
    c_sec
    slog
    -Wl,--as-needed
    json
    -lrt
    -ldl
)

##################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_tensorflow_parser.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_caffe_parser.cc
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${PARSER_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && mv tensorflow_parser.cc stub_tensorflow_parser.cc
            && mv caffe_parser.cc stub_caffe_parser.cc
            &&  echo "Generating stub files end."
    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS ../stub/gen_stubapi.py ${TOP_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
)

##################################################################

############ stub/libfmk_parser.so ############
add_library(fmk_parser_stub SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/stub_tensorflow_parser.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_caffe_parser.cc
)

target_compile_options(fmk_parser_stub PRIVATE
    -O2
    -fno-common
)

target_compile_definitions(fmk_parser_stub PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_HOST_INFER
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(fmk_parser_stub PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${PARSER_DIR}
    ${PARSER_DIR}/inc
    ${PARSER_DIR}/inc/external
    ${PARSER_DIR}/parser
    ${PARSER_DIR}/../inc
    ${PARSER_DIR}/../inc/common/util
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
)

target_link_libraries(fmk_parser_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(fmk_parser_stub PROPERTIES
    OUTPUT_NAME fmk_parser
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS fmk_parser OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS fmk_parser_stub OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)
