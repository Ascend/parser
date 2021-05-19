############ lib_caffe_parser.so ############
add_library(_caffe_parser SHARED
    $<TARGET_OBJECTS:parser_caffe_proto_obj>
)

target_compile_definitions(_caffe_parser PRIVATE
    google=ascend_private
)

target_include_directories(_caffe_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_options(_caffe_parser PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(_caffe_parser PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    ascend_protobuf
    -Wl,--as-needed
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS _caffe_parser OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
