function(protobuf_generate comp c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})
    set(_add_target FALSE)

    set(extra_option "")
    foreach(arg ${ARGN})
        if("${arg}" MATCHES "--proto_path")
            set(extra_option ${arg})
        endif()
    endforeach()


    foreach(file ${ARGN})
        if("${file}" MATCHES "--proto_path")
            continue()
        endif()

        if("${file}" STREQUAL "TARGET")
            set(_add_target TRUE)
            continue()
        endif()

        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        get_filename_component(parent_subdir ${file_dir} NAME)

        if("${parent_subdir}" STREQUAL "proto")
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
        else()
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
        endif()
        list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
                COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
                COMMAND ${PROTOC_PROGRAM} -I${file_dir} ${extra_option} --cpp_out=${proto_output_path} ${abs_file}
                DEPENDS ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    if(_add_target)
        add_custom_target(
            ${comp} DEPENDS ${${c_var}} ${${h_var}}
        )
    endif()
    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)

endfunction()

function(protobuf_generate_py comp py_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: protobuf_generate_py() called without any proto files")
        return()
    endif()

    set(${py_var})
    set(_add_target FALSE)

    foreach(file ${ARGN})
        if("${file}" STREQUAL "TARGET")
            set(_add_target TRUE)
            continue()
        endif()

        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        get_filename_component(parent_subdir ${file_dir} NAME)

        if("${parent_subdir}" STREQUAL "proto")
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
        else()
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
        endif()
        list(APPEND ${py_var} "${proto_output_path}/${file_name}_pb2.py")

        add_custom_command(
                OUTPUT "${proto_output_path}/${file_name}_pb2.py"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
                COMMAND ${CMAKE_COMMAND} -E echo "generate proto python_out ${comp} by ${abs_file}"
                COMMAND ${PROTOC_PROGRAM} -I${file_dir} --python_out=${proto_output_path} ${abs_file}
                DEPENDS ${abs_file}
                COMMENT "Running PYTHON protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    if(_add_target)
        add_custom_target(
            ${comp} DEPENDS ${${py_var}}
        )
    endif()
    set_source_files_properties(${${py_var}} PROPERTIES GENERATED TRUE)
    set(${py_var} ${${py_var}} PARENT_SCOPE)

endfunction()
