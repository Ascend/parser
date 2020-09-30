set(PROTO_LIST
    "${METADEF_DIR}/proto/tensorflow/graph.proto"
    "${METADEF_DIR}/proto/tensorflow/node_def.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
    "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
    "${METADEF_DIR}/proto/tensorflow/function.proto"
    "${METADEF_DIR}/proto/tensorflow/op_def.proto"
    "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor.proto"
    "${METADEF_DIR}/proto/tensorflow/types.proto"
    "${METADEF_DIR}/proto/tensorflow/versions.proto"
    "${METADEF_DIR}/proto/tensorflow/graph_library.proto"
)

protobuf_generate_py(ge PROTO_SRCS ${PROTO_LIST})

include_directories(${CMAKE_CURRENT_LIST_DIR})

############ func2graph/util ############
add_custom_target(util ALL
        DEPENDS ${PROTO_SRCS}
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/util
                && cp -r ${PROTO_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/util
        )

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/util OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}/func2graph
)

