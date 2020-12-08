LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libparser_common

LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0
LOCAL_CFLAGS += -Werror
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

COMMON_LOCAL_SRC_FILES := \
    parser_factory.cc \
    data_op_parser.cc \
    op_parser_factory.cc \
    pre_checker.cc \
    register_tbe.cc \
    parser_api.cc \
    parser_inner_ctx.cc \
    acl_graph_parser_util.cc\
    proto_file_parser.cc \
    ../../graph/passes/pass_manager.cc \
    ../../graph/common/omg_util.cc \
    ../tensorflow/tensorflow_custom_parser_adapter.cc \
    ../tensorflow/tensorflow_fusion_custom_parser_adapter.cc \
    ../tensorflow/tensorflow_fusion_op_parser.cc \
    ../tensorflow/tensorflow_util.cc \
    ../../common/convert/pb2json.cc \
    op_def/ir_pb_converter.cc \
    op_def/defs.cc \
    op_def/op_schema.cc \
    op_def/operator.cc \
    op_map.cc \
    parser_utils.cc \

FMK_COMMON_SRC_FILES := \
    ../../common/types.cc \
    ../../common/util.cc \
    ../../common/model_saver.cc \
    ../../common/fmk_error_codes.cc \
    ../../common/fp16_t.cc \
    ../../common/thread_pool.cc \
    ../../common/auth/file_saver.cc \
    ../../common/auth/cipher.cc \
    ../../common/context/ctx.cc \

LOCAL_SRC_FILES := $(COMMON_LOCAL_SRC_FILES)
#LOCAL_SRC_FILES += $(FMK_COMMON_SRC_FILES)

LOCAL_C_INCLUDES := \
    proto/om.proto \
    proto/insert_op.proto \
    proto/ge_ir.proto \
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
    $(LOCAL_PATH) \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/common/util \
    $(TOPDIR)framework/domi \
    $(TOPDIR)framework/domi/common \
    $(TOPDIR)framework/domi/parser \
    $(TOPDIR)third_party/json/include \
    $(TOPDIR)third_party/protobuf/include \
    libc_sec/include \
    third_party/openssl/include/x86/include \

LOCAL_SHARED_LIBRARIES := \
    libprotobuf \
    libslog \
    libgraph \
    libmmpa \
    libc_sec \
    liberror_manager \
    libregister \
    libge_common \

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_SHARED_LIBRARY)
