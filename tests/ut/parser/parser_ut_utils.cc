/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ut/parser/parser_ut_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace ge {
void ParerUTestsUtils::ClearParserInnerCtx() {
  ge::GetParserContext().input_nodes_format_map.clear();
  ge::GetParserContext().output_formats.clear();
  ge::GetParserContext().user_input_dims.clear();
  ge::GetParserContext().input_dims.clear();
  ge::GetParserContext().op_conf_map.clear();
  ge::GetParserContext().user_out_nodes.clear();
  ge::GetParserContext().default_out_nodes.clear();
  ge::GetParserContext().out_nodes_map.clear();
  ge::GetParserContext().user_out_tensors.clear();
  ge::GetParserContext().net_out_nodes.clear();
  ge::GetParserContext().out_tensor_names.clear();
  ge::GetParserContext().data_tensor_names.clear();
  ge::GetParserContext().is_dynamic_input = false;
  ge::GetParserContext().train_flag = false;
  ge::GetParserContext().format = domi::DOMI_TENSOR_ND;
  ge::GetParserContext().type = domi::FRAMEWORK_RESERVED;
  ge::GetParserContext().run_mode = GEN_OM_MODEL;
  ge::GetParserContext().custom_proto_path = "";
  ge::GetParserContext().caffe_proto_path = "";
  ge::GetParserContext().enable_scope_fusion_passes = "";
  GELOGI("Clear parser inner context successfully.");
}

MemBuffer* ParerUTestsUtils::MemBufferFromFile(const char *path) {
  char path_temp[PATH_MAX + 1] = {0x00};
  if(strlen(path) > PATH_MAX || nullptr == realpath(path, path_temp)) {
      return nullptr;
  }
  FILE *fp = fopen(path_temp, "r+");
  if (fp == nullptr) {
      return nullptr;
  }

  // get model file length
  if (0 != fseek(fp, 0, SEEK_END)) {
      fclose(fp);
      return nullptr;
  }
  long file_length = ftell(fp);
  if (fseek(fp, 0, SEEK_SET)) {
      fclose(fp);
      return nullptr;
  }
  if (file_length <= 0) {
      fclose(fp);
      return nullptr;
  }

  // alloc model buffer
  void *data = malloc((unsigned int)file_length);
  if (!data) {
      fclose(fp);
      return nullptr;
  }

  // read file into memory
  uint32_t read_size = (uint32_t)fread(data, 1, (unsigned int)file_length, fp);

  // check if read success
  if ((long)read_size != file_length) {
      free(data);
      data = nullptr;
      fclose(fp);
      return nullptr;
  }

  // close model file
  fclose(fp);

  // create an MemBuffer
  MemBuffer* membuf = new MemBuffer();
  if (!membuf) {
      free(data);
      data = nullptr;
      return nullptr;
  }
  membuf->data = malloc((unsigned int)read_size);

  // set size && data
  membuf->size = (uint32_t)read_size;
  memcpy((char*)membuf->data, (char*)data, read_size);
  free(data);
  return membuf;
}

bool ParerUTestsUtils::ReadProtoFromText(const char *file, google::protobuf::Message *message) {
  std::ifstream fs(file);
  if (!fs.is_open()) {
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);

  fs.close();
  return ret;
}

void ParerUTestsUtils::WriteProtoToBinaryFile(const google::protobuf::Message &proto, const char *filename) {
  size_t size = proto.ByteSizeLong();
  char *buf = new char[size];
  proto.SerializeToArray(buf, size);
  std::ofstream out(filename);
  out.write(buf, size);
  out.close();
  delete[] buf;
}

void ParerUTestsUtils::WriteProtoToTextFile(const google::protobuf::Message &proto, const char *filename) {
  const int32_t fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 384U);
  if (fd >= 0) {
    google::protobuf::io::FileOutputStream output(fd);
    google::protobuf::TextFormat::Print(proto, &output);
    output.Close();
    close(fd);
  }
}

namespace ut {
NodePtr GraphBuilder::AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt, Format format,
                              DataType data_type, std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(std::move(shape)));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }

  return graph_->AddNode(op_desc);
}
void GraphBuilder::AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
}
void GraphBuilder::AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
  GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
}
}  // namespace ut
}  // namespace ge
