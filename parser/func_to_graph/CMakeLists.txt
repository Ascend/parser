set(PROTO_LIST
    "${TOP_DIR}/inc/register/proto/tensorflow/graph.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/node_def.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/tensor_shape.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/attr_value.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/function.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/op_def.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/resource_handle.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/tensor.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/types.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/versions.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/graph_library.proto"
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

