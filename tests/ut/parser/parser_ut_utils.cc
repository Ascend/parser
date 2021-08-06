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
}  // namespace ge
