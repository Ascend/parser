inc_path := $(shell pwd)/parser/inc/external/
out_path := $(shell pwd)/out/parser/lib64/stub/
stub_path := $(shell pwd)/parser/parser/stub/

mkdir_stub := $(shell mkdir -p $(out_path))
local_stub := $(shell $(HI_PYTHON) $(stub_path)/gen_stubapi.py $(inc_path) $(out_path))
