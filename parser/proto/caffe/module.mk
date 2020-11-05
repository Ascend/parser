LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := lib_caffe_parser

LOCAL_CFLAGS += -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_SRC_FILES := \
    caffe.proto \

LOCAL_C_INCLUDES := \
    third_party/protobuf/include \

LOCAL_SHARED_LIBRARIES := \
    libascend_protobuf \

include $(BUILD_HOST_SHARED_LIBRARY)
