LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libparser_common

LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0
LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

COMMON_LOCAL_SRC_FILES := \
    parser_factory.cc \
    data_op_parser.cc \
    op_parser_factory.cc \
    pre_checker.cc \
    op_registration_tbe.cc \
    parser_api.cc \
    parser_inner_ctx.cc \
    proto_file_parser.cc \
    acl_graph_parser_util.cc \
    tbe_plugin_loader.cc \
    model_saver.cc \
    ../tensorflow/tensorflow_custom_parser_adapter.cc \
    ../tensorflow/tensorflow_fusion_custom_parser_adapter.cc \
    ../tensorflow/tensorflow_fusion_op_parser.cc \
    ../tensorflow/tensorflow_util.cc \
    convert/pb2json.cc \
    op_def/ir_pb_converter.cc \
    op_def/operator.cc \
    op_map.cc \
    parser_types.cc \
    pass_manager.cc \
    parser_fp16_t.cc \
    thread_pool.cc \
    parser_utils.cc \

FMK_COMMON_SRC_FILES := \
#     ../../common/fmk_error_codes.cc \
    ../../common/auth/cipher.cc \
    ../../common/context/ctx.cc \
    ../../graph/passes/pass_manager.cc \
    ../../graph/common/omg_util.cc \
    ../../common/types.cc \
    ../../common/auth/file_saver.cc \
    ../../common/util.cc \
    ../../common/model_saver.cc \
    ../../common/fp16_t.cc \
    ../../common/thread_pool.cc \

LOCAL_SRC_FILES := $(COMMON_LOCAL_SRC_FILES)
LOCAL_SRC_FILES += $(FMK_COMMON_SRC_FILES)

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
    $(TOPDIR)metadef/inc \
    $(TOPDIR)graphengine/inc \
    $(TOPDIR)parser/inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)metadef/inc/external \
    $(TOPDIR)graphengine/inc/external \
    $(TOPDIR)parser/inc/external \
    $(TOPDIR)metadef/inc/external/graph \
    $(TOPDIR)graphengine/inc/framework \
    $(TOPDIR)metadef/inc/common/util \
    $(TOPDIR)graphengine/ge \
    $(TOPDIR)graphengine/ge/common \
    $(TOPDIR)parser/parser \
    $(TOPDIR)parser   \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)third_party/json/include \
    $(TOPDIR)third_party/protobuf/include \
    libc_sec/include \
    third_party/openssl/include/x86/include \

LOCAL_SHARED_LIBRARIES := \
    libascend_protobuf \
    libslog \
    libgraph \
    libc_sec \
    liberror_manager \
    libregister \

LOCAL_STATIC_LIBRARIES += libmmpa

LOCAL_LDFLAGS := -lrt -ldl

include $(BUILD_HOST_SHARED_LIBRARY)
