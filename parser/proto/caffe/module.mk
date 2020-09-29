LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := lib_caffe_parser

ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_SRC_FILES := \
    caffe.proto \

LOCAL_C_INCLUDES := \
    third_party/protobuf/include \

LOCAL_SHARED_LIBRARIES := \
    libprotobuf \

include $(BUILD_HOST_SHARED_LIBRARY)
