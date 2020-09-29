set(PROTO_LIST
    "${TOP_DIR}/inc/register/proto/caffe/caffe.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

############ lib_caffe_parser.so ############
add_library(_caffe_parser SHARED ${PROTO_SRCS})

target_include_directories(_caffe_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(_caffe_parser PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    protobuf
    -Wl,--as-needed
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS _caffe_parser OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
