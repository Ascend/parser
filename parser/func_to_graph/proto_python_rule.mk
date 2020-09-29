include $(BUILD_SYSTEM)/base_rules.mk

FUNCTION_TO_GRAPH_OUT_TIMESTAMP := $(HOST_OUT_ROOT)/func_to_graph/.timestamp

PROTO_SRC_DIR = framework/domi/parser/func_to_graph/proto
PY_PROTO_BUILD_DIR = $(HOST_OUT_ROOT)/tmp/function_to_graph/proto

$(warning PRIVATE_PROTOC is $(PRIVATE_PROTOC))
$(warning protobuf_lib_dir is $(protobuf_lib_dir))

$(FUNCTION_TO_GRAPH_OUT_TIMESTAMP): $(PRIVATE_PROTOC)
	mkdir -p $(PY_PROTO_BUILD_DIR)
	LD_LIBRARY_PATH=$(protobuf_lib_dir):$$LD_LIBRARY_PATH $(PRIVATE_PROTOC) -I=$(PROTO_SRC_DIR) --python_out=$(PY_PROTO_BUILD_DIR) $(PROTO_SRC_DIR)/*.proto

$(LOCAL_BUILT_MODULE): $(FUNCTION_TO_GRAPH_OUT_TIMESTAMP)
	mkdir -p $@
	cp -rf $(PY_PROTO_BUILD_DIR)/* $@