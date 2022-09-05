
LOCAL_PATH := $(call my-dir)
include $(LOCAL_PATH)/stub/Makefile
COMMON_LOCAL_C_INCLUDES := \
    proto/om.proto \
    proto/insert_op.proto \
    proto/ge_ir.proto \
    proto/task.proto \
    proto/tensorflow/graph.proto \
    proto/tensorflow/node_def.proto \
    proto/tensorflow/tensor_shape.proto \
    proto/tensorflow/attr_value.proto \
    proto/tensorflow/function.proto \
    proto/tensorflow/op_def.proto \
    proto/tensorflow/resource_handle.proto \
    proto/tensorflow/tensor.proto \
    proto/tensorflow/types.proto \
    proto/tensorflow/versions.proto \
    proto/tensorflow/graph_library.proto \
    proto/caffe/caffe.proto \
    tensorflow/proto/tensorflow/graph.proto \
    tensorflow/proto/tensorflow/node_def.proto \
    tensorflow/proto/tensorflow/tensor_shape.proto \
    tensorflow/proto/tensorflow/attr_value.proto \
    tensorflow/proto/tensorflow/function.proto \
    tensorflow/proto/tensorflow/op_def.proto \
    tensorflow/proto/tensorflow/resource_handle.proto \
    tensorflow/proto/tensorflow/tensor.proto \
    tensorflow/proto/tensorflow/types.proto \
    tensorflow/proto/tensorflow/versions.proto \
    tensorflow/proto/tensorflow/graph_library.proto \
    caffe/proto/caffe/caffe.proto \
    $(LOCAL_PATH) \
    $(LOCAL_PATH)/tensorflow \
    $(LOCAL_PATH)/caffe \
    $(LOCAL_PATH)/../ \
    $(TOPDIR)inc \
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)parser/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)parser/inc/external \
    $(TOPDIR)metadef/inc/external/graph \
    $(TOPDIR)parser/inc/external/parser \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)parser/parser \
    $(TOPDIR)parser        \
    $(TOPDIR)graphengine/ge \
    libc_sec/include \
    third_party/protobuf/include \
    third_party/json/include \
    third_party/openssl/include/x86/include \

include $(CLEAR_VARS)

LOCAL_MODULE := libfmk_parser

LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0
LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

PARSER_TENSORFLOW_SRC_FILES := \
    tensorflow/tensorflow_arg_parser.cc \
    tensorflow/tensorflow_auto_mapping_parser_adapter.cc \
    tensorflow/tensorflow_constant_parser.cc \
    tensorflow/tensorflow_data_parser.cc \
    tensorflow/tensorflow_enter_parser.cc \
    tensorflow/tensorflow_fill_parser.cc \
    tensorflow/tensorflow_frameworkop_parser.cc \
    tensorflow/tensorflow_identity_parser.cc \
    tensorflow/tensorflow_merge_parser.cc \
    tensorflow/tensorflow_no_op_parser.cc \
    tensorflow/tensorflow_parser.cc \
    tensorflow/tensorflow_ref_switch_parser.cc \
    tensorflow/tensorflow_reshape_parser.cc \
    tensorflow/tensorflow_shape_n_parser.cc \
    tensorflow/tensorflow_squeeze_parser.cc \
    tensorflow/tensorflow_var_is_initialized_op_parser.cc \
    tensorflow/tensorflow_variable_v2_parser.cc \
    tensorflow/proto/tensorflow/graph_library.proto \
	caffe/caffe_parser.cc \
    caffe/caffe_data_parser.cc \
    caffe/caffe_reshape_parser.cc \
    caffe/caffe_custom_parser_adapter.cc \
    caffe/caffe_op_parser.cc \

PARSER_SCOPE_SRC_FILES := \
    tensorflow/scope/scope_pass_manager.cc \

FMK_COMMON_SRC_FILES := \
    tensorflow/graph_to_function_def.cc \
    tensorflow/parser_graph_optimizer.cc \
    tensorflow/iterator_fusion_pass.cc \
    common/op_def/arg_op_operator.cc \
    common/op_def/constant_operator.cc \
    common/op_def/fill_operator.cc \
    common/op_def/framework_op_operator.cc \
    common/op_def/no_op_operator.cc \
    common/op_def/ref_switch_operator.cc \
    common/op_def/shape_n_operator.cc \
    common/op_def/var_is_initialized_op_operator.cc \
    common/op_def/variable_operator.cc \

LOCAL_SRC_FILES := $(PARSER_TENSORFLOW_SRC_FILES)
LOCAL_SRC_FILES += $(PARSER_SCOPE_SRC_FILES)
LOCAL_SRC_FILES += $(FMK_COMMON_SRC_FILES)

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SHARED_LIBRARIES := \
    libascend_protobuf \
    libslog \
    libc_sec \
    liberror_manager \
    libparser_common \
    libgraph \
    libregister \
    lib_caffe_parser \

LOCAL_STATIC_LIBRARIES += libmmpa

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_SHARED_LIBRARY)

#compiler for host parser
include $(CLEAR_VARS)

LOCAL_MODULE := stub/libfmk_parser

LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DREUSE_MEMORY=1 -O2
LOCAL_CFLAGS += -DFMK_HOST_INFER -DFMK_SUPPORT_DUMP
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_C_INCLUDES := $(COMMON_LOCAL_C_INCLUDES)

LOCAL_SRC_FILES := ../../out/parser/lib64/stub/tensorflow_parser.cc
LOCAL_SRC_FILES += ../../out/parser/lib64/stub/caffe_parser.cc


LOCAL_SHARED_LIBRARIES :=

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_SHARED_LIBRARY)
